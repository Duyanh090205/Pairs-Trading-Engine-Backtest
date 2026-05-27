"""SQLite read/write for live state. Single-file ACID store."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: str | Path) -> None:
    """Create tables from schema.sql if not present + apply additive migrations.

    Order matters:
      1. _apply_migrations on existing tables FIRST — bring columns up to date
         before any CREATE INDEX touches those columns.
      2. executescript to create any tables that don't exist yet.
      3. _apply_migrations again to create indexes that depend on newly-added
         columns (idempotent CREATE INDEX IF NOT EXISTS).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _apply_migrations(conn)
        conn.executescript(SCHEMA_PATH.read_text())
        _apply_migrations(conn)
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Forward-only ALTER TABLE migrations.

    SQLite's CREATE TABLE IF NOT EXISTS doesn't add columns to existing tables,
    so a backup restored from an older deploy may be missing columns that
    schema.sql now requires. We ensure every required column exists with a safe
    default. Each column declaration here MUST be ADDABLE without backfill
    (i.e. nullable OR has a literal default).
    """
    # Only run column-adds if `orders` table already exists; otherwise the
    # initial executescript will create it from scratch (no migration needed).
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='orders'"
    ).fetchone() is not None:
        _ensure_columns(conn, "orders", [
            ("broker_order_id", "TEXT"),
            ("bar_ts",          "TEXT NOT NULL DEFAULT ''"),
            ("limit_price",     "REAL"),
            ("fill_price",      "REAL"),
            ("fill_qty",        "REAL"),
            ("decision_price",  "REAL"),
            ("filled_ts",       "TEXT"),
            ("raw_response",    "TEXT"),
        ])
        # Indexes that depend on added columns must be created AFTER the ALTERs.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_pair_bar ON orders(pair_id, bar_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_broker ON orders(broker_order_id)")


def _ensure_columns(conn: sqlite3.Connection, table: str,
                    columns: list[tuple[str, str]]) -> None:
    """For each (name, type_decl), ALTER TABLE ADD COLUMN if missing."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, type_decl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {type_decl}")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def log_event(conn: sqlite3.Connection, event: str, level: str, message: str,
              payload: dict[str, Any] | None = None) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "INSERT INTO audit_log (ts, event, level, message, payload) VALUES (?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), event, level, message,
         json.dumps(payload) if payload else None),
    )


def is_halted(conn: sqlite3.Connection) -> tuple[bool, str | None]:
    row = conn.execute("SELECT halted, reason FROM kill_switch WHERE id = 1").fetchone()
    return bool(row["halted"]), row["reason"]


def set_halt(conn: sqlite3.Connection, reason: str) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "UPDATE kill_switch SET halted = 1, reason = ?, triggered_ts = ? WHERE id = 1",
        (reason, datetime.now(timezone.utc).isoformat()),
    )


def clear_halt(conn: sqlite3.Connection) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "UPDATE kill_switch SET halted = 0, cleared_ts = ? WHERE id = 1",
        (datetime.now(timezone.utc).isoformat(),),
    )


def get_or_set_session_equity(conn: sqlite3.Connection, current_equity: float) -> float:
    """Persist session_start_equity across restarts. Returns the original value.

    On first call: stores current_equity as the session baseline.
    Subsequent calls (after restart): returns the stored baseline (NOT current).
    Reason: kill_switch drawdown should be measured from FIRST deploy equity,
    not from each restart's current equity (which may already be depleted).
    """
    from datetime import datetime, timezone
    row = conn.execute(
        "SELECT session_start_equity FROM engine_session WHERE id = 1"
    ).fetchone()
    if row is not None and row["session_start_equity"] is not None:
        return float(row["session_start_equity"])
    conn.execute(
        "UPDATE engine_session SET session_start_equity = ?, session_started_ts = ? "
        "WHERE id = 1",
        (current_equity, datetime.now(timezone.utc).isoformat()),
    )
    return current_equity


def get_last_decision_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT last_decision_date FROM engine_session WHERE id = 1"
    ).fetchone()
    return row["last_decision_date"] if row else None


def set_last_decision_date(conn: sqlite3.Connection, date_iso: str) -> None:
    conn.execute(
        "UPDATE engine_session SET last_decision_date = ? WHERE id = 1",
        (date_iso,),
    )
