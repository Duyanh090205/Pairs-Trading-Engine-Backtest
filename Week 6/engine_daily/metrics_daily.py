"""
Phase 3 — Daily metrics aggregation (V4)

Aggregates per-pair daily P&L from engine_daily.run_fold_daily into:
  - fold-level Sharpe, total_return, max_drawdown, calmar
  - exit_breakdown (counts of zero_cross / hard_sl / open_at_eom)
  - per-pair P&L distribution stats
  - daily portfolio equity curve (sum across pairs; vol-target sizing per pair)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _per_pair_stats(pair_df: pd.DataFrame) -> dict:
    """Per-pair stats: n_trades, win_rate, avg_pnl, exit_breakdown."""
    pos = pair_df["position"].values
    pos_prev = np.zeros_like(pos)
    pos_prev[1:] = pos[:-1]
    is_entry = (pos != 0) & (pos_prev == 0)
    is_exit_close = (pos == 0) & (pos_prev != 0)
    n_trades = int(is_entry.sum())

    # Trade outcomes: PnL between entry and exit
    pnl_net = pair_df["daily_pnl_net"].values
    trade_pnls = []
    cur_entry_idx = -1
    for i in range(len(pos)):
        if is_entry[i]:
            cur_entry_idx = i
        elif is_exit_close[i] and cur_entry_idx >= 0:
            trade_pnls.append(pnl_net[cur_entry_idx:i + 1].sum())
            cur_entry_idx = -1
    # Trade still open at end of window
    open_at_end = cur_entry_idx >= 0
    if open_at_end:
        trade_pnls.append(pnl_net[cur_entry_idx:].sum())

    trade_pnls = np.array(trade_pnls)
    win_rate = float((trade_pnls > 0).mean()) if len(trade_pnls) else 0.0
    avg_pnl = float(trade_pnls.mean()) if len(trade_pnls) else 0.0

    # Exit breakdown: single source of truth = position-level transitions.
    # When position transitions to 0 at bar i, the state-machine signal
    # fired at bar i-1 (1-bar execution lag), so look up exit_code[i-1].
    # This avoids double-counting a last-bar exit (whose position transition
    # is outside the window) as BOTH zero_cross AND open_at_eom.
    exit_codes = pair_df["exit_code"].values
    n_zero_cross = 0
    n_hard_sl = 0
    for i in range(len(pos)):
        if is_exit_close[i] and i >= 1:
            code = exit_codes[i - 1]
            if code == 1:
                n_zero_cross += 1
            elif code == 2:
                n_hard_sl += 1
    n_open_at_eom = int(open_at_end)

    notional = float(pair_df.attrs.get("notional", 0.0))
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_pnl_per_trade": avg_pnl,
        "total_pnl_net": float(pnl_net.sum()),
        "n_zero_cross": n_zero_cross,
        "n_hard_sl": n_hard_sl,
        "n_open_at_eom": n_open_at_eom,
        "notional": notional,
    }


def aggregate_fold_metrics(
    pair_results: dict[tuple[str, str], pd.DataFrame],
    total_capital: float = 1_000_000.0,
) -> dict:
    """
    Roll up per-pair results into one fold's metrics.

    Returns dict with keys: sharpe, total_return, max_dd, calmar, n_trades,
    win_rate, avg_net_bps, exit_breakdown, per_pair_metrics, daily_equity.
    """
    if not pair_results:
        return {"sharpe": 0.0, "total_return": 0.0, "n_trades": 0,
                "max_dd": 0.0, "calmar": 0.0, "win_rate": 0.0,
                "avg_net_bps": 0.0, "exit_breakdown": {},
                "per_pair_metrics": {}, "daily_equity": pd.Series(dtype=float)}

    # ---- Build daily portfolio P&L (sum across all pairs; outer-join on dates) ----
    per_pair_pnl = {f"{ta}/{tb}": df["daily_pnl_net"] for (ta, tb), df in pair_results.items()}
    pnl_df = pd.DataFrame(per_pair_pnl).fillna(0.0)
    daily_portfolio_pnl = pnl_df.sum(axis=1)
    daily_returns = daily_portfolio_pnl / total_capital
    daily_equity = total_capital + daily_portfolio_pnl.cumsum()

    # ---- Sharpe ----
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 1e-12:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1)
                       * np.sqrt(_TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0
    total_return = float(daily_portfolio_pnl.sum() / total_capital)

    # ---- Max DD ----
    eq = daily_equity.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = sharpe / abs(max_dd) if max_dd < -1e-6 else 0.0

    # ---- Per-pair stats + exit breakdown rollup ----
    per_pair: dict[str, dict] = {}
    exit_breakdown = {"zero_cross": 0, "hard_sl": 0, "open_at_eom": 0}
    total_trade_pnls: list[float] = []
    total_notional_traded = 0.0
    for (ta, tb), df in pair_results.items():
        s = _per_pair_stats(df)
        per_pair[f"{ta}/{tb}"] = s
        exit_breakdown["zero_cross"] += s["n_zero_cross"]
        exit_breakdown["hard_sl"] += s["n_hard_sl"]
        exit_breakdown["open_at_eom"] += s["n_open_at_eom"]
        if s["n_trades"]:
            total_trade_pnls.append(s["total_pnl_net"] / s["n_trades"])
            total_notional_traded += s["n_trades"] * s["notional"]

    n_total_trades = sum(s["n_trades"] for s in per_pair.values())
    win_trades = sum(s["n_trades"] * s["win_rate"] for s in per_pair.values())
    overall_win_rate = float(win_trades / n_total_trades) if n_total_trades else 0.0

    # avg_net_bps: total net P&L / total notional * 10000
    if total_notional_traded > 0:
        total_net = sum(s["total_pnl_net"] for s in per_pair.values())
        avg_net_bps = float(total_net / total_notional_traded * 10000.0)
    else:
        avg_net_bps = 0.0

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_dd": max_dd,
        "calmar": calmar,
        "n_trades": int(n_total_trades),
        "win_rate": overall_win_rate,
        "avg_net_bps": avg_net_bps,
        "exit_breakdown": exit_breakdown,
        "n_pairs_traded": len(pair_results),
        "per_pair_metrics": per_pair,
        "daily_equity": daily_equity,
        "daily_returns": daily_returns,
    }
