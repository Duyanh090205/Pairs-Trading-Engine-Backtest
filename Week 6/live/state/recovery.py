"""Restore engine state from SQLite after restart."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class OpenPosition:
    pair_id: str
    side_a: str
    side_b: str
    beta: float
    direction: int
    notional_a: float
    notional_b: float
    entry_ts: str
    entry_z: float


def load_open_positions(conn: sqlite3.Connection) -> list[OpenPosition]:
    rows = conn.execute(
        "SELECT pair_id, side_a, side_b, beta, direction, notional_a, notional_b, "
        "entry_ts, entry_z FROM positions WHERE exit_ts IS NULL"
    ).fetchall()
    return [OpenPosition(**dict(r)) for r in rows]


def load_pending_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE status IN ('submitted', 'partial')"
    ).fetchall()
    return [dict(r) for r in rows]
