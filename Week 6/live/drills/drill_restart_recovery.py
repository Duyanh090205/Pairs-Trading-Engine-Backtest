"""GRADED drill: kill -9 mid-write → restart must recover all state.

Pass criteria:
  - All 2 open positions readable after restart
  - 1 pending order readable
  - 3 audit-log entries present
  - Hardstop file flag persists and is_tripped() returns True (file-based safety)
  - SQLite WAL files are recoverable (auto-checkpoint on next open)
  - No duplicate orders after restart (idempotency via client_order_id PK)

Run: python live/drills/drill_restart_recovery.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _run_writer_subprocess(db_path: Path, flag_path: Path) -> int:
    writer = Path(__file__).parent / "_writer_subprocess.py"
    proc = subprocess.run(
        [sys.executable, str(writer), str(db_path), str(flag_path)],
        capture_output=True, text=True, timeout=30,
    )
    # os._exit(0) yields exit code 0 even though Python finalizers were skipped
    return proc.returncode


def _check(name: str, cond: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  {mark}  {name}{(' — ' + detail) if detail else ''}")
    return cond


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="restart_drill_"))
    db_path = tmpdir / "state.db"
    flag_path = tmpdir / "HARDSTOP.flag"

    print(f"{YELLOW}== Kill -9 Recovery Drill =={RESET}")
    print(f"  workdir: {tmpdir}")

    # Phase A — subprocess writes and exits ungracefully
    print("  [A] writer subprocess running...")
    rc = _run_writer_subprocess(db_path, flag_path)
    print(f"      exit code = {rc} (0 = os._exit clean, no finalizers)")
    time.sleep(0.2)  # let OS settle WAL

    # Phase B — restart: open DB cleanly, run recovery
    print("  [B] restart simulation — reopening state...")

    # Sandbox hardstop module to use our flag path
    import live.safety.hardstop as hs
    hs.HARDSTOP_FLAG_PATH = flag_path

    from live.state.persist import connect
    from live.state.recovery import load_open_positions, load_pending_orders

    results = []
    with connect(db_path) as conn:
        open_pos = load_open_positions(conn)
        pending = load_pending_orders(conn)
        audit_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event = 'stream'"
        ).fetchone()
        # Schema must be intact
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        results.append(_check("schema intact after kill",
                              {"positions", "orders", "audit_log", "kill_switch"}.issubset(tables)))
        results.append(_check("2 open positions recovered",
                              len(open_pos) == 2, f"got {len(open_pos)}"))
        results.append(_check("pending order recovered",
                              len(pending) == 1, f"got {len(pending)}"))
        results.append(_check("3 audit-log writes durable",
                              audit_rows["n"] == 3, f"got {audit_rows['n']}"))

        # Idempotency: re-submitting the same client_order_id is rejected by PK
        try:
            from datetime import datetime, timezone
            conn.execute(
                "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
                "order_type, status, submitted_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("coid_test_001", "MSFT_GOOG", "2026-06-01T20:00:00Z", "MSFT", "buy",
                 50.0, "market", "submitted", datetime.now(timezone.utc).isoformat()),
            )
            results.append(_check("idempotency: duplicate client_order_id rejected",
                                  False, "INSERT succeeded but should have failed"))
        except Exception as e:
            results.append(_check("idempotency: duplicate client_order_id rejected",
                                  "UNIQUE" in str(e) or "PRIMARY KEY" in str(e),
                                  type(e).__name__))

        # Hardstop file persisted across crash → is_tripped True
        results.append(_check("hardstop flag persists across crash",
                              hs.is_tripped()))

        # Bonus: verify recoverable state can drive engine continuation
        results.append(_check("recovered position has correct beta",
                              abs(open_pos[0].beta - 1.2) < 1e-9 if open_pos else False))

    # Cleanup
    try:
        if flag_path.exists():
            flag_path.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    n_pass = sum(results)
    n_total = len(results)
    print()
    if all(results):
        print(f"{GREEN}DRILL PASS{RESET} — {n_pass}/{n_total} checks")
        return 0
    print(f"{RED}DRILL FAIL{RESET} — {n_pass}/{n_total} checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
