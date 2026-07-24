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
        # Net signed qty across all filled orders for this pair+ticker —
        # buys contribute positively, sells negatively. Prior lifecycles
        # cancel out (entry buy + exit sell = 0). Current open cycle gives
        # the unmatched residual (matches broker).
        # Use LOWER + REPLACE for back-compat with rows written before
        # normalize_status() was added.
        leg_qty_row = conn.execute(
            "SELECT SUM(CASE WHEN side = 'buy' THEN fill_qty "
            "             WHEN side = 'sell' THEN -fill_qty ELSE 0 END) AS q "
            "FROM orders WHERE pair_id = ? AND ticker = ? "
            "AND LOWER(REPLACE(status, 'OrderStatus.', '')) = 'filled'",
            (r["pair_id"], ticker),
        ).fetchone()
        net_leg_qty = float(leg_qty_row["q"] or 0.0)
        # Sign convention from position.direction. For NEE_PFE dir=+1:
        #   side_a NEE is LONG  -> +net_leg_qty (already correct: buy=+, sell=-)
        #   side_b PFE is SHORT -> the entry was a SELL, so net_leg_qty is
        #                          already negative; sign multiplier keeps it
        #                          aligned with broker's negative qty for shorts.
        # Therefore: just use net_leg_qty directly — the buy/sell signing
        # already encodes the long/short convention.
        qty += net_leg_qty
    return qty


def _recent_fill_activity(conn: sqlite3.Connection, grace_s: float) -> bool:
    """True if any order filled within the last `grace_s` seconds.

    Guards the mid-fill race (2026-07-06 incident): broker already shows the
    position but the positions row is only INSERTed by TradingStream after
    BOTH legs fill — reconcile running inside that window sees a false
    mismatch (local qty counts only orders of OPEN positions) and trips the
    kill switch. ISO-8601 UTC strings compare lexicographically.
    """
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=grace_s)).isoformat()
    row = conn.execute(
        "SELECT 1 FROM orders WHERE filled_ts IS NOT NULL AND filled_ts >= ? "
        "LIMIT 1", (cutoff,),
    ).fetchone()
    return row is not None


def reconcile(trading_client, conn: sqlite3.Connection,
              tolerance_shares: float = 1.0,
              grace_s: float = 180.0) -> list[ReconcileMismatch]:
    """Return mismatches > tolerance_shares. Empty = clean.

    Skips the round entirely (returns []) if a fill happened within the last
    `grace_s` seconds — the positions-row insert may still be in flight.
    Pass grace_s=0 to disable (tests).
    """
    from live.state.persist import log_event
    if grace_s > 0 and _recent_fill_activity(conn, grace_s):
        log_event(conn, "reconcile_skipped", "INFO",
                  f"fill within last {grace_s:.0f}s — skipping round "
                  "(mid-fill grace window)")
        return []
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
