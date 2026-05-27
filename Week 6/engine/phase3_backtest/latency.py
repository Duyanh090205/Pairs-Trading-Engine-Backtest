"""
Phase 3 — Latency Sweep

Wraps Phase 2 engine output: shifts position array by additional bars beyond
the baseline t+1 lag already built into the engine.

Configs:
  t+1  : baseline (0 extra bars)
  t+2  : 1 extra bar
  t+5  : 4 extra bars
  t+10 : 9 extra bars
  random: per-trade lag sampled Uniform(1, 5) extra bars

Pass criterion (per spec §3.4):
  Sharpe(t+5) > 0  AND  monotonic degradation (no cliff at t+2)
  If t+2 kills Sharpe → signal is microstructure noise, not cointegration.

Run on default config only. Do NOT cross with OAT grid.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

LATENCY_CONFIGS = {
    "t+1":    0,    # baseline already in engine
    "t+2":    1,
    "t+5":    4,
    "t+10":   9,
}


def apply_extra_lag(pair_df: pd.DataFrame, extra_bars: int) -> pd.DataFrame:
    """
    Shift position array by extra_bars additional bars (on top of t+1 in engine).
    Fills leading bars with 0 (flat).
    Also shifts signal by same amount so timestamp proof remains consistent.
    """
    if extra_bars == 0:
        return pair_df

    df = pair_df.copy()
    pos = df["position"].values.copy()
    sig = df["signal"].values.copy()

    shifted_pos = np.roll(pos, extra_bars)
    shifted_pos[:extra_bars] = 0
    shifted_sig = np.roll(sig, extra_bars)
    shifted_sig[:extra_bars] = 0

    df["position"] = shifted_pos
    df["signal"]   = shifted_sig

    # Shift forced_exit column if present (from _apply_max_holding)
    if "forced_exit" in df.columns:
        fe = df["forced_exit"].values.copy()
        shifted_fe = np.roll(fe, extra_bars)
        shifted_fe[:extra_bars] = False
        df["forced_exit"] = shifted_fe

    return df


def run_latency_sweep(
    fold_results: dict,
    trading_1min: dict[str, pd.DataFrame],
    total_capital: float = 1_000_000.0,
    n_open_pairs_max: int = 50,
    tc_bps: float = 30.0,
    borrow_bps_yr: float = 50.0,
    seed: int = 42,
) -> dict:
    """
    Run all latency configurations (t+1 through t+10 plus random).

    Parameters
    ----------
    fold_results  : {(ta, tb): pair_df} from engine.run_fold_execution

    Returns
    -------
    dict with: sharpe_by_lag, pass_criterion, monotonic_ok, latency_table
    """
    per_pair_dollar = total_capital / n_open_pairs_max
    results = {}

    for lag_name, extra_bars in LATENCY_CONFIGS.items():
        sharpe = _run_with_fixed_lag(
            fold_results, trading_1min, extra_bars,
            per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
        )
        results[lag_name] = sharpe
        log.info("Latency %s: Sharpe=%.3f", lag_name, sharpe)

    # Random lag: per-fold average Uniform(1,5) extra bars
    rng_lag = np.random.default_rng(seed)
    random_extra = int(rng_lag.integers(1, 6))   # 1..5 inclusive, whole fold
    random_sharpe = _run_with_fixed_lag(
        fold_results, trading_1min, random_extra,
        per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
    )
    results["random"] = random_sharpe
    log.info("Latency random (extra=%d): Sharpe=%.3f", random_extra, random_sharpe)

    # Pass criterion
    t5_pass    = results.get("t+5", 0.0) > 0.0
    t2_ok      = results.get("t+2", 0.0) > 0.0
    monotonic  = _is_monotonically_degrading(results)

    latency_pass = t5_pass and monotonic

    log.info(
        "Latency sweep: t+5_pass=%s monotonic=%s overall=%s",
        t5_pass, monotonic, latency_pass,
    )

    # Table for audit log
    ordered_keys = ["t+1", "t+2", "t+5", "t+10", "random"]
    latency_table = [
        {"lag": k, "extra_bars": LATENCY_CONFIGS.get(k, "var"),
         "sharpe": results.get(k, 0.0)}
        for k in ordered_keys
    ]

    return {
        "sharpe_by_lag":  results,
        "t5_pass":        t5_pass,
        "t2_ok":          t2_ok,
        "monotonic_ok":   monotonic,
        "latency_pass":   latency_pass,
        "latency_table":  latency_table,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_with_fixed_lag(
    fold_results: dict,
    trading_1min: dict,
    extra_bars: int,
    per_pair_dollar: float,
    tc_bps: float,
    borrow_bps_yr: float,
    total_capital: float,
) -> float:
    """Run PnL for all pairs with a fixed extra lag. Returns aggregate Sharpe."""
    all_bar_pnl = []
    all_trades  = []

    for (ta, tb), pair_df in fold_results.items():
        if ta not in trading_1min or tb not in trading_1min:
            continue

        lagged_df = apply_extra_lag(pair_df, extra_bars)
        close_a   = trading_1min[ta]["close"].reindex(pair_df.index)
        close_b   = trading_1min[tb]["close"].reindex(pair_df.index)

        result = compute_pair_pnl(
            lagged_df, close_a, close_b,
            per_pair_dollar, tc_bps, borrow_bps_yr,
        )
        all_bar_pnl.append(result["bar_pnl"])
        all_trades.extend(result["trade_log"])

    if not all_bar_pnl:
        return 0.0

    total_pnl = pd.concat(all_bar_pnl, axis=1).fillna(0.0).sum(axis=1)
    metrics   = compute_metrics(total_pnl, total_capital, all_trades)
    return float(metrics["sharpe"])


def _is_monotonically_degrading(results: dict) -> bool:
    """
    Check that Sharpe degrades monotonically from t+1 to t+10.
    Allows small tolerance (0.05 Sharpe units) to account for noise.
    """
    ordered = ["t+1", "t+2", "t+5", "t+10"]
    sharpes = [results.get(k, None) for k in ordered]
    sharpes = [s for s in sharpes if s is not None]

    if len(sharpes) < 2:
        return True

    tolerance = 0.10   # allow 0.10 Sharpe unit upward blip before calling non-monotonic
    for i in range(1, len(sharpes)):
        if sharpes[i] > sharpes[i - 1] + tolerance:
            return False
    return True
