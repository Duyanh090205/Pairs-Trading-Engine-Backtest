"""Rolling-window comparison: live actuals vs backtest predicted reference values."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Backtest reference values (from results/v4/z30_composite/fold_metrics.csv).
# IMPORTANT semantics:
#   - sharpe_per_fold:     mean of per-fold annualized Sharpe across active folds = 1.279
#   - win_rate:            TRADE-LEVEL (winning_trades / total_trades), averaged across
#                          active folds = 0.592. NOT the fold-level "17/26 folds with
#                          positive Sharpe = 65%" — that is a different metric and would
#                          systematically misalign with live's trade-level computation.
#   - halt_rate:           12 of 39 folds halted by composite filter = 0.308
#   - avg_cost_bps:        APPROXIMATE — no aggregate column in fold_metrics.csv; estimated
#                          from V4 dynamic-cost model (spread + impact + commission + borrow).
BACKTEST_REFERENCE = {
    "sharpe_per_fold": 1.279,
    "monthly_sharpe_annualized": 0.660,
    "win_rate": 0.592,                  # trade-level, matches live aggregation
    "avg_cost_bps_per_trade": 22.0,     # APPROXIMATE — see comment
    "halt_rate": 0.308,
}


@dataclass
class MetricRow:
    metric: str
    backtest: float | None
    live: float | None
    delta: float | None


def rolling_comparison(conn: sqlite3.Connection, window_days: int = 21) -> list[MetricRow]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        "SELECT realized_pnl, realized_cost_bps FROM positions "
        "WHERE exit_ts IS NOT NULL AND exit_ts >= ?",
        (cutoff,),
    ).fetchall()

    out: list[MetricRow] = []
    n = len(rows)

    if n < 5:
        # Not enough data — return reference rows with live=None
        for k, v in BACKTEST_REFERENCE.items():
            out.append(MetricRow(metric=k, backtest=v, live=None, delta=None))
        out.append(MetricRow("n_trades", None, n, None))
        return out

    pnls = [float(r["realized_pnl"]) for r in rows if r["realized_pnl"] is not None]
    wins = sum(1 for p in pnls if p > 0)
    win_rate_live = wins / len(pnls) if pnls else None
    mean = sum(pnls) / len(pnls) if pnls else 0.0
    var = sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    sharpe_live = (mean / math.sqrt(var) * math.sqrt(252)) if var > 0 else None

    cost_bps = [float(r["realized_cost_bps"]) for r in rows if r["realized_cost_bps"] is not None]
    avg_cost = (sum(cost_bps) / len(cost_bps)) if cost_bps else None

    def _row(k, live):
        bt = BACKTEST_REFERENCE.get(k)
        d = (live - bt) if (live is not None and bt is not None) else None
        out.append(MetricRow(metric=k, backtest=bt, live=live, delta=d))

    _row("sharpe_per_fold", sharpe_live)
    _row("win_rate", win_rate_live)
    _row("avg_cost_bps_per_trade", avg_cost)
    out.append(MetricRow(metric="n_trades", backtest=None, live=n, delta=None))
    return out
