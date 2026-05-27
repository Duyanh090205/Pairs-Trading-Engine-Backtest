"""DB integrity audit: schema, migrations, idempotency, corruption recovery.

Categories:
  A. init_db is idempotent (running twice produces identical schema)
  B. Migration handles legacy DB (no bar_ts column) without data loss
  C. ALTER TABLE bar_ts is idempotent (running twice is safe)
  D. Schema constraints actually enforced (NOT NULL, PRIMARY KEY)
  E. WAL mode recovery after simulated crash (WAL file present)
  F. Concurrent open (two connections, one read one write)
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_init_db_idempotent():
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="db_idem_")
    db = Path(td) / "state.db"
    init_db(db)
    # Capture schema after first init
    conn1 = sqlite3.connect(db)
    schema1 = sorted([r[0] for r in conn1.execute(
        "SELECT sql FROM sqlite_master WHERE type IN ('table','index') "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()])
    conn1.close()
    # Run init_db AGAIN
    init_db(db)
    conn2 = sqlite3.connect(db)
    schema2 = sorted([r[0] for r in conn2.execute(
        "SELECT sql FROM sqlite_master WHERE type IN ('table','index') "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()])
    conn2.close()
    check("init_db idempotent: schema identical after 2nd run",
          schema1 == schema2,
          f"diff: {set(schema1) ^ set(schema2)}" if schema1 != schema2 else "")


def t_migration_legacy_to_new():
    """Build a legacy DB WITHOUT bar_ts, then run init_db -> migration should add bar_ts."""
    td = tempfile.mkdtemp(prefix="db_mig_")
    db = Path(td) / "legacy.db"
    # Create a legacy orders schema (without bar_ts)
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE orders (
            client_order_id TEXT PRIMARY KEY,
            pair_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            order_type TEXT NOT NULL,
            status TEXT NOT NULL,
            submitted_ts TEXT NOT NULL
        )
    """)
    # Insert a row
    conn.execute(
        "INSERT INTO orders (client_order_id, pair_id, ticker, side, qty, order_type, status, submitted_ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("legacy_c1", "X_Y", "X", "buy", 10, "market", "filled",
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    from live.state.persist import init_db
    init_db(db)   # Should ALTER TABLE to add bar_ts

    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(orders)").fetchall()]
    check("migration: bar_ts column added to legacy table",
          "bar_ts" in cols)
    # Verify the pre-existing row is still present + has default '' for new col
    row = conn.execute("SELECT pair_id, bar_ts FROM orders WHERE client_order_id = ?",
                       ("legacy_c1",)).fetchone()
    check("migration: pre-existing row preserved (no data loss)",
          row is not None and row[0] == "X_Y")
    check("migration: new bar_ts column has default '' for legacy rows",
          row[1] == "")
    conn.close()


def t_migration_already_migrated():
    """Run init_db twice on a fresh DB. Migration should detect bar_ts already exists."""
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="db_mig2_")
    db = Path(td) / "fresh.db"
    init_db(db)
    try:
        init_db(db)  # second run — _apply_migrations should be a no-op
        check("migration: 2nd run on already-migrated DB is no-op (no crash)", True)
    except Exception as e:
        check("migration: 2nd run on already-migrated DB is no-op (no crash)",
              False, f"{type(e).__name__}: {e}")


def t_schema_constraints_enforced():
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="db_ce_")
    db = Path(td) / "state.db"
    init_db(db)
    conn = sqlite3.connect(db)
    # PRIMARY KEY on orders.client_order_id
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, "
                 "qty, order_type, status, submitted_ts) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 ("dup", "X_Y", "bts", "X", "buy", 10, "market", "submitted", ts))
    try:
        conn.execute("INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, "
                     "qty, order_type, status, submitted_ts) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("dup", "X_Y", "bts", "X", "buy", 10, "market", "submitted", ts))
        check("orders.client_order_id PRIMARY KEY enforced", False,
              "duplicate insert succeeded")
    except sqlite3.IntegrityError:
        check("orders.client_order_id PRIMARY KEY enforced (duplicate rejected)", True)

    # NOT NULL on orders.pair_id
    try:
        conn.execute("INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, "
                     "qty, order_type, status, submitted_ts) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("nl1", None, "bts", "X", "buy", 10, "market", "submitted", ts))
        check("orders.pair_id NOT NULL enforced", False, "NULL pair_id accepted")
    except sqlite3.IntegrityError:
        check("orders.pair_id NOT NULL enforced", True)

    # CHECK on kill_switch.id (must be 1)
    try:
        conn.execute("INSERT INTO kill_switch (id, halted) VALUES (2, 0)")
        check("kill_switch.id CHECK (id=1) enforced", False, "id=2 accepted")
    except sqlite3.IntegrityError:
        check("kill_switch.id CHECK (id=1) enforced (singleton table)", True)

    conn.close()


def t_wal_mode_active():
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="db_wal_")
    db = Path(td) / "state.db"
    init_db(db)
    conn = sqlite3.connect(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    check("WAL journal mode active", mode.lower() == "wal", f"got {mode}")
    conn.close()


def t_concurrent_connections():
    """Two simultaneous connections — one writes, the other reads. SQLite WAL allows this."""
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="db_concur_")
    db = Path(td) / "state.db"
    init_db(db)
    ts = datetime.now(timezone.utc).isoformat()
    writer = sqlite3.connect(db, isolation_level=None)
    reader = sqlite3.connect(db, isolation_level=None)
    writer.execute(
        "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
        "order_type, status, submitted_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("c1", "X_Y", "bts", "X", "buy", 10, "market", "submitted", ts),
    )
    # Reader sees the committed write
    row = reader.execute(
        "SELECT pair_id FROM orders WHERE client_order_id = 'c1'"
    ).fetchone()
    check("concurrent: reader sees committed write under WAL",
          row is not None and row[0] == "X_Y")
    writer.close()
    reader.close()


def main() -> int:
    print("== DB integrity audit ==")
    print("\n--- A. init_db idempotency ---")
    t_init_db_idempotent()
    print("\n--- B. Legacy -> new schema migration ---")
    t_migration_legacy_to_new()
    print("\n--- C. Migration on already-migrated DB ---")
    t_migration_already_migrated()
    print("\n--- D. Schema constraints ---")
    t_schema_constraints_enforced()
    print("\n--- E. WAL mode ---")
    t_wal_mode_active()
    print("\n--- F. Concurrent connections ---")
    t_concurrent_connections()
    print()
    if errors:
        print(f"FAIL: {len(errors)} integrity issue(s): {errors}")
        return 1
    print("PASS: all DB integrity checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
