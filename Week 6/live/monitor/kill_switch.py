"""Auto-halt conditions evaluation. Distinct from safety/hardstop:
  - kill_switch: operator-clearable via clear_halt() once root cause is fixed
  - hardstop:    file-based, non-overridable until file manually removed

Triggers checked in evaluate():
  1. cumulative drift loss > 25% of cumulative gross PnL  → "drift_loss"
  2. session drawdown > 3% of session start equity         → "drawdown"
  3. > N consecutive losing trades                          → "consecutive_losses"
  4. unexplained reconcile mismatch > tolerance             → handled by reconcile.trip_halt_on_mismatch
  5. data stale > 5 minutes (bar gap > 300s)                → "stale_data" (engine sets)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class KillSwitchThresholds:
    drift_loss_pct_of_pnl: float = 0.25
    daily_drawdown_pct: float = 0.03
    consecutive_losses_max: int = 10
    stale_data_seconds: int = 300


def _cum_realized(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) AS p FROM positions "
        "WHERE exit_ts IS NOT NULL"
    ).fetchone()
    return float(row["p"]) if row["p"] is not None else 0.0


def _gross_pnl_abs(conn: sqlite3.Connection) -> float:
    """|sum of profits| + |sum of losses|, used as denominator for drift_loss_pct."""
    row = conn.execute(
        "SELECT COALESCE(SUM(ABS(realized_pnl)), 0) AS p FROM positions "
        "WHERE exit_ts IS NOT NULL"
    ).fetchone()
    return float(row["p"]) if row["p"] is not None else 0.0


def _consecutive_losses(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT realized_pnl FROM positions WHERE exit_ts IS NOT NULL "
        "ORDER BY exit_ts DESC LIMIT 50"
    ).fetchall()
    n = 0
    for r in rows:
        if r["realized_pnl"] is None or r["realized_pnl"] >= 0:
            break
        n += 1
    return n


def evaluate(conn: sqlite3.Connection,
             session_start_equity: float,
             current_equity: float,
             thresholds: KillSwitchThresholds | None = None) -> str | None:
    """Return halt-reason string if any trigger trips, else None."""
    th = thresholds or KillSwitchThresholds()
    # 2. session drawdown
    if session_start_equity > 0:
        dd = (session_start_equity - current_equity) / session_start_equity
        if dd >= th.daily_drawdown_pct:
            return f"daily_drawdown_{dd:.4f}"
    # 3. consecutive losses
    cl = _consecutive_losses(conn)
    if cl >= th.consecutive_losses_max:
        return f"consecutive_losses_{cl}"
    # 1. drift loss (requires comparison to predicted; placeholder until trade_journal
    # tracks predicted vs realized cost). Use realized cost overshooting predicted
    # cost as drift signal once trade_journal is populated.
    gross = _gross_pnl_abs(conn)
    drift_loss = _cum_realized_drift(conn)
    if gross > 0 and drift_loss / gross >= th.drift_loss_pct_of_pnl:
        return f"drift_loss_pct_{drift_loss / gross:.4f}"
    return None


def _cum_realized_drift(conn: sqlite3.Connection) -> float:
    """Sum of |realized_cost_bps - predicted_cost_bps| in $ across closed trades."""
    rows = conn.execute(
        "SELECT realized_cost_bps, predicted_cost_bps, notional_a, notional_b "
        "FROM positions WHERE exit_ts IS NOT NULL "
        "AND realized_cost_bps IS NOT NULL AND predicted_cost_bps IS NOT NULL"
    ).fetchall()
    total = 0.0
    for r in rows:
        notional = float(r["notional_a"] or 0) + float(r["notional_b"] or 0)
        bps_diff = abs(float(r["realized_cost_bps"]) - float(r["predicted_cost_bps"]))
        total += notional * bps_diff / 10_000
    return total
