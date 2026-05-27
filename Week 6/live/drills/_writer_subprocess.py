"""Subprocess helper for drill_restart_recovery.

Writes positions + orders + audit log to a passed DB path, then immediately
exits via os._exit(0) WITHOUT cleanup. Simulates a kill -9 from the OS.

Args (sys.argv): db_path  hardstop_flag_path
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live.state.persist import init_db


def main() -> None:
    db_path = Path(sys.argv[1])
    flag_path = Path(sys.argv[2])
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = datetime.now(timezone.utc).isoformat()

    # 2 open positions
    cur.execute(
        "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
        "notional_a, notional_b, entry_ts, entry_z) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("JPM_BAC", "JPM", "BAC", 1.2, 1, 7500.0, 9000.0, ts, 3.15),
    )
    cur.execute(
        "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
        "notional_a, notional_b, entry_ts, entry_z) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("XOM_CVX", "XOM", "CVX", 0.85, -1, 5000.0, 4250.0, ts, -3.30),
    )

    # 1 pending (submitted, unfilled) order
    cur.execute(
        "INSERT INTO orders (client_order_id, broker_order_id, pair_id, bar_ts, ticker, "
        "side, qty, order_type, status, submitted_ts, decision_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("coid_test_001", "alpaca_test_001", "MSFT_GOOG", "2026-06-01T20:00:00Z",
         "MSFT", "buy", 50.0, "market", "submitted", ts, 389.18),
    )

    # 3 audit log entries
    for i in range(3):
        cur.execute(
            "INSERT INTO audit_log (ts, event, level, message) VALUES (?, ?, ?, ?)",
            (ts, "stream", "INFO", f"writer iteration {i}"),
        )

    conn.commit()
    # Write hardstop flag to simulate "had been tripped pre-crash"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(f"{ts}\nsubprocess_test_trip\n")

    # Simulate ungraceful exit — no conn.close(), no flushing of Python finalizers
    os._exit(0)


if __name__ == "__main__":
    main()
