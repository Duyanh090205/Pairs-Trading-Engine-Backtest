"""
Phase 3 — Fold-Level Metrics Runner

Iterates all pairs in a fold's engine output, runs PnL per pair,
aggregates into a single fold metrics dict consumed by audit_log and orchestrator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

_TOTAL_CAPITAL    = 1_000_000.0
_N_OPEN_PAIRS_MAX = 50
_TC_BPS           = 30.0
_BORROW_BPS_YR    = 50.0


def run_fold_pnl(
    fold_results: dict,
    trading_1min: dict[str, pd.DataFrame],
    pairs_df: pd.DataFrame,
    delta: float,
    delta_metrics: dict,
    total_capital: float = _TOTAL_CAPITAL,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    tc_bps: float = _TC_BPS,
    borrow_bps_yr: float = _BORROW_BPS_YR,
) -> dict:
    """
    Compute PnL and metrics for all pairs in a fold.

    Parameters
    ----------
    fold_results   : {(ta, tb): pair_df} from engine.run_fold_execution
    trading_1min   : {ticker: DataFrame} with 'close' column
    pairs_df       : Phase 1 surviving pairs (for metadata)
    delta          : selected Kalman delta for this fold
    delta_metrics  : {delta: {metric_kurt, median_HL, median_ACF78}} from select_delta

    Returns
    -------
    dict with: bar_pnl, bar_equity, daily_returns, sharpe, max_dd, cagr, calmar,
               win_rate, n_trades, avg_holding_bars, n_pairs_traded,
               cost_decomp, trade_log, pair_pnls, delta, delta_metrics,
               spread_prior_map (for Kalman degenerate check)
    """
    per_pair_dollar = total_capital / n_open_pairs_max

    all_bar_pnl       = []
    all_trade_log     = []
    all_rebalance_log = []
    pair_pnls         = {}
    per_pair_metrics  = {}
    cost_commission   = 0.0
    cost_borrow       = 0.0
    cost_rebalance    = 0.0
    spread_prior_map  = {}

    # Global counter for trade_id assignment (unique within a fold; runner promotes
    # to globally-unique by combining with fold_id downstream).
    _next_trade_id = 0

    for (ta, tb), pair_df in fold_results.items():
        if ta not in trading_1min or tb not in trading_1min:
            log.debug("Skip PnL for %s/%s: 1min data missing", ta, tb)
            continue

        close_a = trading_1min[ta]["close"].reindex(pair_df.index)
        close_b = trading_1min[tb]["close"].reindex(pair_df.index)

        result = compute_pair_pnl(
            pair_df, close_a, close_b,
            per_pair_dollar=per_pair_dollar,
            tc_bps=tc_bps,
            borrow_bps_yr=borrow_bps_yr,
        )

        bar_pnl = result["bar_pnl"]
        all_bar_pnl.append(bar_pnl)
        pair_tl  = result["trade_log"]
        pair_rbl = result.get("rebalance_log", [])
        hl_row = pairs_df[(pairs_df["ticker_a"] == ta) & (pairs_df["ticker_b"] == tb)]
        hl_days = float(hl_row["half_life_days"].iloc[0]) if not hl_row.empty else float("nan")
        pair_id = f"{ta}_{tb}"

        # Map pair-local trade indices to fold-unique trade_ids, then enrich rows.
        local_to_global = {}
        for t in pair_tl:
            local_idx = t.pop("pair_trade_idx")
            global_id = _next_trade_id
            _next_trade_id += 1
            local_to_global[local_idx] = global_id
            t["trade_id"]       = global_id
            t["pair_id"]        = pair_id
            t["ticker_a"]       = ta
            t["ticker_b"]       = tb
            t["half_life_days"] = hl_days
        all_trade_log.extend(pair_tl)

        # Rebalance events: tag with global trade_id, ticker (B leg only — engine
        # rebalances the hedge side), pair_id, and fold metadata.
        for r in pair_rbl:
            local_idx = r.pop("pair_trade_idx")
            r["trade_id"]            = local_to_global.get(local_idx, -1)
            r["pair_id"]             = pair_id
            r["ticker"]              = tb   # only the B leg is rebalanced
            r["ticker_a"]            = ta
            r["ticker_b"]            = tb
        all_rebalance_log.extend(pair_rbl)

        pair_pnls[(ta, tb)] = bar_pnl
        cost_commission += result["cost_commission"]
        cost_borrow     += result["cost_borrow"]
        cost_rebalance  += result["cost_rebalance"]
        spread_prior_map[(ta, tb)] = pair_df["spread_prior"].dropna().values
        # Per-pair trade metrics (for bucket-level volume stratification)
        _n_tl = len(pair_tl)
        per_pair_metrics[(ta, tb)] = {
            "n_trades":    _n_tl,
            "win_rate":    sum(1 for t in pair_tl if t["net_pnl"] > 0) / _n_tl if _n_tl else 0.0,
            "avg_net_bps": sum(t["net_bps"] for t in pair_tl) / _n_tl if _n_tl else 0.0,
            "total_pnl":   float(bar_pnl.sum()),
        }

    if not all_bar_pnl:
        log.warning("No pairs produced PnL — fold has no executable trades")
        return _empty_fold_metrics(delta, delta_metrics)

    # Aggregate: sum bar PnL across all pairs, union index
    total_bar_pnl = _sum_series(all_bar_pnl)

    metrics = compute_metrics(total_bar_pnl, total_capital, all_trade_log)

    # Lookahead assertion: exec_ts > signal_ts for every trade
    lookahead_ok = _verify_no_lookahead(fold_results)

    # Kalman degenerate check: var(prior) vs static PCA spread
    kalman_degenerate = _check_kalman_degenerate(fold_results, pairs_df)

    # Exit reason breakdown
    exit_breakdown = _exit_reason_breakdown(all_trade_log)

    return {
        # Core metrics
        "sharpe":           metrics["sharpe"],
        "max_dd":           metrics["max_dd"],
        "cagr":             metrics["cagr"],
        "calmar":           metrics["calmar"],
        "win_rate":         metrics["win_rate"],
        "n_trades":         metrics["n_trades"],
        "avg_holding_bars": metrics["avg_holding_bars"],
        "avg_net_bps":      metrics["avg_net_bps"],
        "total_return":     metrics["total_return"],
        # Time series
        "bar_pnl":          total_bar_pnl,
        "bar_equity":       metrics["bar_equity"],
        "daily_returns":    metrics["daily_returns"],
        # Cost decomposition
        "cost_decomp": {
            "commission":  cost_commission,
            "borrow":      cost_borrow,
            "rebalance":   cost_rebalance,
        },
        # Trade log + rebalance log (Week-5 hand-off)
        "trade_log":        all_trade_log,
        "rebalance_log":    all_rebalance_log,
        "exit_breakdown":   exit_breakdown,
        # Per-pair data (for volume stratification in Phase 4)
        "pair_pnls":        pair_pnls,
        "n_pairs_traded":   len(pair_pnls),
        "per_pair_metrics":  per_pair_metrics,
        # Audit fields
        "lookahead_ok":      lookahead_ok,
        "kalman_degenerate": kalman_degenerate,
        "delta":             delta,
        "delta_metrics":     delta_metrics,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_series(series_list: list[pd.Series]) -> pd.Series:
    """Sum a list of bar PnL series on their union index (fills 0 for missing bars)."""
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1).fillna(0.0)
    return df.sum(axis=1)


def _verify_no_lookahead(fold_results: dict) -> bool:
    """
    Verify position[t] == signal[t-1] for all t in all pairs.
    Returns True if no lookahead detected.
    """
    for (ta, tb), df in fold_results.items():
        pos = df["position"].values
        sig = df["signal"].values
        for i in range(1, len(pos)):
            if pos[i] != sig[i - 1]:
                log.error("Lookahead detected at pair %s/%s bar %d", ta, tb, i)
                return False
    return True


def _check_kalman_degenerate(fold_results: dict, pairs_df: pd.DataFrame) -> bool:  # noqa: ARG001
    """
    Red flag: Kalman prior spread has collapsed or has extreme kurtosis.
    Avoids cross-scale comparison: R is from 5-min formation, prior is from
    1-min trading — their variances are not comparable.
    Returns True if degenerate (flag).
    """
    degenerate = False
    for (ta, tb), df in fold_results.items():
        prior = df["spread_prior"].dropna().values
        if len(prior) < 10:
            continue

        var_prior = float(np.var(prior))
        if var_prior < 1e-14:
            log.warning("Kalman degenerate: %s/%s prior spread collapsed (var=%.2e)", ta, tb, var_prior)
            degenerate = True
            continue

        std_p = float(np.std(prior))
        if std_p < 1e-10:
            continue
        z = (prior - float(np.mean(prior))) / std_p
        kurt = float(np.mean(z ** 4)) - 3.0
        if abs(kurt) > 10.0:
            log.warning("Kalman degenerate: %s/%s excess kurtosis=%.1f", ta, tb, kurt)
            degenerate = True

    return degenerate


def _exit_reason_breakdown(trade_log: list[dict]) -> dict:
    reasons = {"zero_cross": 0, "eos": 0, "end_of_window": 0, "sl": 0, "max_holding": 0}
    for t in trade_log:
        r = t.get("exit_reason", "zero_cross")
        reasons[r] = reasons.get(r, 0) + 1
    total = len(trade_log)
    pct = {k: v / total if total > 0 else 0.0 for k, v in reasons.items()}
    return {"counts": reasons, "pct": pct}


def _empty_fold_metrics(delta: float, delta_metrics: dict) -> dict:
    return {
        "sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "calmar": 0.0,
        "win_rate": 0.0, "n_trades": 0, "avg_holding_bars": 0.0,
        "avg_net_bps": 0.0, "total_return": 0.0,
        "bar_pnl": pd.Series(dtype=float),
        "bar_equity": pd.Series(dtype=float),
        "daily_returns": pd.Series(dtype=float),
        "cost_decomp": {"commission": 0.0, "borrow": 0.0, "rebalance": 0.0},
        "trade_log": [], "rebalance_log": [], "exit_breakdown": {},
        "pair_pnls": {}, "n_pairs_traded": 0, "per_pair_metrics": {},
        "lookahead_ok": True, "kalman_degenerate": False,
        "delta": delta, "delta_metrics": delta_metrics,
    }
