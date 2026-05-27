"""Per-trade attribution: predicted vs realized cost per leg + total."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class TradeAttribution:
    pair_id: str
    entry_ts: str
    exit_ts: str | None
    entry_z: float
    exit_z: float | None
    realized_pnl: float | None
    predicted_cost_bps: float | None
    realized_cost_bps: float | None
    drift_bps: float | None


def record_entry(conn: sqlite3.Connection, pair_id: str,
                 predicted_cost_bps: float) -> None:
    """Update predicted_cost_bps on the currently-open position for `pair_id`."""
    conn.execute(
        "UPDATE positions SET predicted_cost_bps = ? "
        "WHERE pair_id = ? AND exit_ts IS NULL",
        (predicted_cost_bps, pair_id),
    )


def record_exit(conn: sqlite3.Connection, pair_id: str, exit_ts: str,
                exit_z: float, exit_reason: str, realized_pnl: float,
                realized_cost_bps: float) -> None:
    """Close the open position row for `pair_id`."""
    conn.execute(
        "UPDATE positions SET exit_ts = ?, exit_z = ?, exit_reason = ?, "
        "realized_pnl = ?, realized_cost_bps = ? "
        "WHERE pair_id = ? AND exit_ts IS NULL",
        (exit_ts, exit_z, exit_reason, realized_pnl, realized_cost_bps, pair_id),
    )


def list_recent(conn: sqlite3.Connection, limit: int = 50) -> list[TradeAttribution]:
    rows = conn.execute(
        "SELECT pair_id, entry_ts, exit_ts, entry_z, exit_z, realized_pnl, "
        "predicted_cost_bps, realized_cost_bps FROM positions "
        "ORDER BY entry_ts DESC LIMIT ?", (limit,),
    ).fetchall()
    out: list[TradeAttribution] = []
    for r in rows:
        drift = None
        if r["predicted_cost_bps"] is not None and r["realized_cost_bps"] is not None:
            drift = float(r["realized_cost_bps"]) - float(r["predicted_cost_bps"])
        out.append(TradeAttribution(
            pair_id=r["pair_id"],
            entry_ts=r["entry_ts"],
            exit_ts=r["exit_ts"],
            entry_z=float(r["entry_z"]),
            exit_z=float(r["exit_z"]) if r["exit_z"] is not None else None,
            realized_pnl=float(r["realized_pnl"]) if r["realized_pnl"] is not None else None,
            predicted_cost_bps=float(r["predicted_cost_bps"]) if r["predicted_cost_bps"] is not None else None,
            realized_cost_bps=float(r["realized_cost_bps"]) if r["realized_cost_bps"] is not None else None,
            drift_bps=drift,
        ))
    return out
