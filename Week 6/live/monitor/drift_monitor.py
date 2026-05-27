"""Realized vs predicted cost in bps — rolling window."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def compute_drift_bps(conn: sqlite3.Connection, lookback_days: int = 21) -> dict[str, float]:
    """Returns {'mean_drift_bps', 'max_abs_drift_bps', 'n_trades', 'total_cost_drift_usd'}.

    Drift = realized_cost_bps - predicted_cost_bps. Positive = real cost worse than predicted.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute(
        "SELECT realized_cost_bps, predicted_cost_bps, notional_a, notional_b "
        "FROM positions WHERE exit_ts IS NOT NULL AND exit_ts >= ? "
        "AND realized_cost_bps IS NOT NULL AND predicted_cost_bps IS NOT NULL",
        (cutoff,),
    ).fetchall()
    if not rows:
        return {
            "mean_drift_bps": None, "max_abs_drift_bps": None,
            "n_trades": 0, "total_cost_drift_usd": 0.0,
        }
    diffs = []
    cost_usd = 0.0
    for r in rows:
        d = float(r["realized_cost_bps"]) - float(r["predicted_cost_bps"])
        diffs.append(d)
        notional = float(r["notional_a"] or 0) + float(r["notional_b"] or 0)
        cost_usd += notional * d / 10_000
    return {
        "mean_drift_bps": sum(diffs) / len(diffs),
        "max_abs_drift_bps": max(abs(d) for d in diffs),
        "n_trades": len(diffs),
        "total_cost_drift_usd": cost_usd,
    }
