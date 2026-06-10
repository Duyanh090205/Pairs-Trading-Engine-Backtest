"""Drill: order rejection (insufficient buying power, etc.) → log + alert + continue.

Pass criteria:
  - Broker exception during submit_order is caught
  - orders.status set to 'rejected'
  - audit_log gets 'order_error' entry
  - Engine continues processing other pairs (no crash, no hardstop trip)
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def _check(name, cond, detail=""):
    mark = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  {mark}  {name}{(' — ' + detail) if detail else ''}")
    return cond


class _RejectingBroker:
    """Simulates Alpaca rejection on submit."""
    def submit_order(self, _):
        raise RuntimeError("403 insufficient buying power")


class _WorkingBroker:
    def submit_order(self, _):
        return type("R", (), {"id": "broker_id_99", "status": "accepted"})()


def main() -> int:
    print("== Order rejection drill ==")
    from live.execution.order_manager import OrderRequest, submit_order
    from live.safety import hardstop
    from live.state.persist import connect, init_db, is_halted

    td = tempfile.mkdtemp(prefix="rejection_drill_")
    db = Path(td) / "state.db"
    init_db(db)
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"

    results = []
    bar_ts = "2026-06-01T20:00:00Z"

    # Order 1 — broker rejects
    req_reject = OrderRequest("REJ_PAIR", bar_ts, "A", "REJ", "buy", 10, "market")
    with connect(db) as conn:
        out = submit_order(_RejectingBroker(), conn, req_reject, decision_price=100.0)
        row = conn.execute(
            "SELECT status FROM orders WHERE client_order_id = ?",
            (out.client_order_id,),
        ).fetchone()
        err_log = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event = 'order_error'"
        ).fetchone()
        halted, _ = is_halted(conn)

    results.append(_check("rejected order persisted with status='rejected'",
                          row is not None and row["status"] == "rejected",
                          row["status"] if row else "missing"))
    results.append(_check("audit_log has order_error entry",
                          err_log["n"] >= 1, f"count={err_log['n']}"))
    results.append(_check("kill_switch NOT tripped on single rejection",
                          halted is False))

    # Order 2 — different pair, broker accepts (engine continues)
    req_ok = OrderRequest("OK_PAIR", bar_ts, "A", "OK", "buy", 5, "market")
    with connect(db) as conn:
        out_ok = submit_order(_WorkingBroker(), conn, req_ok, decision_price=200.0)
    # normalize_status canonicalizes broker 'accepted' -> 'submitted'
    results.append(_check("engine continues processing other pairs after rejection",
                          out_ok.status == "submitted" and out_ok.submitted))

    print()
    if all(results):
        print(f"{GREEN}DRILL PASS{RESET} — {sum(results)}/{len(results)}")
        return 0
    print(f"{RED}DRILL FAIL{RESET} — {sum(results)}/{len(results)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
