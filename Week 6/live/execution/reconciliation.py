"""Broker positions vs local SQLite reconciliation."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class ReconcileMismatch:
    ticker: str
    broker_qty: float
    local_qty: float
    diff: float


def _expected_local_qty(conn: sqlite3.Connection, ticker: str) -> float:
    """Sum signed shares for `ticker` across open positions.

    Authoritative source = positions table (written by TradingStream when both
    legs of a pair fill). For open positions:
      direction=+1 → long side_a, short side_b  → +qty on side_a, -qty on side_b
      direction=-1 → short side_a, long side_b  → -qty on side_a, +qty on side_b
    Quantity per leg is reconstructed from fill_qty of the corresponding orders row.
    """
    rows = conn.execute(
        "SELECT pair_id, side_a, side_b, direction, notional_a, notional_b "
        "FROM positions WHERE exit_ts IS NULL "
        "AND (side_a = ? OR side_b = ?)",
        (ticker, ticker),
    ).fetchall()
    if not rows:
        return 0.0
    qty = 0.0
    for r in rows:
        # Determine shares filled for THIS ticker via orders.fill_qty (more precise than notional/price)
        # Use LOWER + REPLACE for back-compat with rows written before normalize_status() was added.
        leg_qty_row = conn.execute(
            "SELECT SUM(fill_qty) AS q FROM orders "
            "WHERE pair_id = ? AND ticker = ? "
            "AND LOWER(REPLACE(status, 'OrderStatus.', '')) = 'filled'",
            (r["pair_id"], ticker),
        ).fetchone()
        leg_qty = float(leg_qty_row["q"] or 0.0)
        # Sign by which leg this ticker is + the pair direction
        if ticker == r["side_a"]:
            sign = 1 if r["direction"] == 1 else -1
        else:  # side_b
            sign = -1 if r["direction"] == 1 else 1
        qty += sign * leg_qty
    return qty


def reconcile(trading_client, conn: sqlite3.Connection,
              tolerance_shares: float = 1.0) -> list[ReconcileMismatch]:
    """Return mismatches > tolerance_shares. Empty = clean."""
    from live.state.persist import log_event
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        log_event(conn, "reconcile_error", "ERROR",
                  f"could not fetch broker positions: {type(e).__name__}: {e}")
        return []
    broker_map: dict[str, float] = {}
    for p in positions:
        broker_map[str(p.symbol)] = float(p.qty)
    seen = set()
    out: list[ReconcileMismatch] = []
    for ticker, b_qty in broker_map.items():
        seen.add(ticker)
        local = _expected_local_qty(conn, ticker)
        diff = b_qty - local
        if abs(diff) > tolerance_shares:
            out.append(ReconcileMismatch(ticker, b_qty, local, diff))
    # Tickers we hold locally but not at broker
    # Use LOWER + REPLACE for back-compat with rows pre-normalize_status().
    local_tickers = {r["ticker"] for r in conn.execute(
        "SELECT DISTINCT ticker FROM orders "
        "WHERE LOWER(REPLACE(status, 'OrderStatus.', '')) IN "
        "('filled', 'partial', 'partially_filled')"
    )}
    for ticker in local_tickers - seen:
        local = _expected_local_qty(conn, ticker)
        if abs(local) > tolerance_shares:
            out.append(ReconcileMismatch(ticker, 0.0, local, -local))
    return out


def trip_halt_on_mismatch(conn: sqlite3.Connection,
                          mismatches: list[ReconcileMismatch]) -> bool:
    """If any mismatch > tolerance, trip the kill_switch (DB) and return True."""
    if not mismatches:
        return False
    from live.state.persist import log_event, set_halt
    reason = f"reconcile_mismatch:{len(mismatches)}_tickers"
    set_halt(conn, reason)
    log_event(conn, "reconcile_halt", "CRITICAL", reason,
              {"mismatches": [m.__dict__ for m in mismatches]})
    return True
