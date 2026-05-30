"""Smoketest: ZTracker buffer persists across simulated restart.

Audit finding 2026-05-28: ZTracker.buf is in-memory only. Every Render
restart re-seeds it from formation spreads, so live Z values 'restart'
and don't accumulate trading-window observations the way backtest does.

Fix: save buf to pair_z_buffer table after every push; restore on
engine init.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live.engine_live.z_tracker import ZTracker
from live.state.persist import (
    connect, init_db, load_z_buffer, save_z_buffer,
)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    color = GREEN if cond else RED
    print(f"  {color}[{mark}]{RESET} {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_schema_has_pair_z_buffer():
    td = tempfile.mkdtemp(prefix="z_schema_")
    db = Path(td) / "x.db"
    init_db(db)
    with connect(db) as conn:
        cols = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='pair_z_buffer'"
        ).fetchone()
    check("pair_z_buffer table created on init_db", cols is not None)


def t_save_then_load_roundtrip():
    td = tempfile.mkdtemp(prefix="z_round_")
    db = Path(td) / "x.db"
    init_db(db)
    vals = [0.1, -0.2, 0.05, 0.0, -0.15, 0.3]
    with connect(db) as conn:
        save_z_buffer(conn, "A_B", vals)
        loaded = load_z_buffer(conn, "A_B")
    check("save->load returns same list",
          loaded == vals, f"got {loaded}")


def t_load_missing_returns_none():
    td = tempfile.mkdtemp(prefix="z_miss_")
    db = Path(td) / "x.db"
    init_db(db)
    with connect(db) as conn:
        loaded = load_z_buffer(conn, "NEVER_SAVED")
    check("load of unknown pair returns None", loaded is None,
          f"got {loaded}")


def t_overwrite_existing():
    td = tempfile.mkdtemp(prefix="z_over_")
    db = Path(td) / "x.db"
    init_db(db)
    with connect(db) as conn:
        save_z_buffer(conn, "C_D", [1.0, 2.0, 3.0])
        save_z_buffer(conn, "C_D", [9.0, 8.0])
        loaded = load_z_buffer(conn, "C_D")
    check("second save overwrites first (UPSERT)",
          loaded == [9.0, 8.0], f"got {loaded}")


def t_restore_into_ztracker():
    """End-to-end: build ZTracker, push some values, save, restore into new
    ZTracker, verify next push gives same Z as if no restart had happened."""
    td = tempfile.mkdtemp(prefix="z_e2e_")
    db = Path(td) / "x.db"
    init_db(db)

    # Build first tracker, seed with 60 values, push 5 trading bars
    seed = [0.01 * (i - 30) for i in range(60)]  # span -0.3..+0.29
    zt1 = ZTracker(window=60, seed=seed)
    for x in [0.1, 0.2, 0.3, 0.4, 0.5]:
        zt1.push(x)
    buf_before = zt1.to_list()
    z_next_no_restart = zt1.push(0.6)
    buf_after = zt1.to_list()

    # Save BEFORE the last push, restore into fresh tracker
    with connect(db) as conn:
        save_z_buffer(conn, "P_Q", buf_before)
        restored_buf = load_z_buffer(conn, "P_Q")

    zt2 = ZTracker(window=60, seed=None)
    zt2.restore_from(restored_buf)
    z_next_after_restart = zt2.push(0.6)

    check("restored ZTracker buf matches saved",
          restored_buf == buf_before, f"len before={len(buf_before)}, after={len(restored_buf)}")
    check("Z value identical after save/restore/push",
          abs(z_next_after_restart - z_next_no_restart) < 1e-12,
          f"no_restart={z_next_no_restart:.6f}, after_restart={z_next_after_restart:.6f}")
    check("buffer state same after push in both",
          zt2.to_list() == buf_after, f"diverged")


def t_restart_without_save_falls_back_to_seed():
    """If pair has no persisted buf, restore_from is NOT called and the
    formation seed is used. Verify ZTracker behaves like a clean seed."""
    td = tempfile.mkdtemp(prefix="z_legacy_")
    db = Path(td) / "x.db"
    init_db(db)
    seed = [0.0] * 60
    zt = ZTracker(window=60, seed=seed)
    with connect(db) as conn:
        buf = load_z_buffer(conn, "UNKNOWN")
    check("no persisted buf -> load returns None", buf is None)
    # ZTracker still works with formation seed
    z = zt.push(0.5)
    check("ZTracker push works without restoration",
          z is not None and abs(z) > 0, f"got {z}")


def main() -> int:
    print("== Smoketest: ZTracker persistence across restart ==\n")
    print("--- 1. Schema migration creates pair_z_buffer ---")
    t_schema_has_pair_z_buffer()
    print("\n--- 2. save -> load roundtrip ---")
    t_save_then_load_roundtrip()
    print("\n--- 3. load missing returns None ---")
    t_load_missing_returns_none()
    print("\n--- 4. overwrite via UPSERT ---")
    t_overwrite_existing()
    print("\n--- 5. End-to-end restart: Z value identical ---")
    t_restore_into_ztracker()
    print("\n--- 6. No persisted buf -> formation seed used ---")
    t_restart_without_save_falls_back_to_seed()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
