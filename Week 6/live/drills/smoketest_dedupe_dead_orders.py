"""Smoketest: dedupe must IGNORE canceled/rejected/expired orders.

Bug discovered live 2026-05-27: after user cancelled orders on broker,
the engine's submit_order silently deduped on the next attempt because
the DB row still existed (status=canceled). Result: one-legged position.

Fix: dedupe only blocks if the existing order is still alive (or filled).
For dead statuses, walk an attempt counter to get a fresh client_order_id.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live.execution.order_manager import OrderRequest, submit_order
from live.state.persist import connect, init_db

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


def _mock_client():
    """Mock TradingClient.submit_order that returns a fake broker response."""
    client = MagicMock()
    counter = {"n": 0}

    def _submit(req):
        counter["n"] += 1
        r = MagicMock()
        r.id = f"broker_{counter['n']}"
        r.status = "OrderStatus.ACCEPTED"
        return r
    client.submit_order = _submit
    client._n_calls = counter
    return client


def _make_req(side: str = "buy", qty: float = 10.0) -> OrderRequest:
    return OrderRequest(
        pair_id="CRWD_EQT", bar_ts="2026-05-27T20:00:00Z", leg="A",
        ticker="CRWD", side=side, qty=qty, order_type="market",
    )


def t_active_order_still_deduped():
    """A previously-submitted order in ACCEPTED status MUST dedupe."""
    td = tempfile.mkdtemp(prefix="dedupe_active_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_client()
    req = _make_req()
    with connect(db) as conn:
        r1 = submit_order(client, conn, req, decision_price=50.0)
        r2 = submit_order(client, conn, req, decision_price=50.0)
    check("first submit succeeded", r1.submitted is True)
    check("second submit (same params, still active) hit dedupe",
          r2.submitted is False and r2.client_order_id == r1.client_order_id)
    check("broker.submit_order called exactly once",
          client._n_calls["n"] == 1, f"got {client._n_calls['n']}")


def t_canceled_order_allows_resubmit():
    """A previously-CANCELED order MUST NOT block re-submission."""
    td = tempfile.mkdtemp(prefix="dedupe_cancel_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_client()
    req = _make_req()
    with connect(db) as conn:
        r1 = submit_order(client, conn, req, decision_price=50.0)
        # Simulate user cancel: WebSocket trade_update flipped status to canceled
        conn.execute(
            "UPDATE orders SET status = 'OrderStatus.CANCELED' WHERE client_order_id = ?",
            (r1.client_order_id,),
        )
        r2 = submit_order(client, conn, req, decision_price=50.0)
    check("first submit succeeded", r1.submitted is True)
    check("second submit AFTER cancel actually submitted (no dedupe)",
          r2.submitted is True, f"got submitted={r2.submitted}")
    check("second submit got DIFFERENT client_order_id",
          r1.client_order_id != r2.client_order_id,
          f"r1={r1.client_order_id[:8]}, r2={r2.client_order_id[:8]}")
    check("broker.submit_order called twice (once per real submit)",
          client._n_calls["n"] == 2, f"got {client._n_calls['n']}")


def t_rejected_order_allows_resubmit():
    """REJECTED status also allows re-submit."""
    td = tempfile.mkdtemp(prefix="dedupe_reject_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_client()
    req = _make_req()
    with connect(db) as conn:
        r1 = submit_order(client, conn, req, decision_price=50.0)
        conn.execute(
            "UPDATE orders SET status = 'rejected' WHERE client_order_id = ?",
            (r1.client_order_id,),
        )
        r2 = submit_order(client, conn, req, decision_price=50.0)
    check("retry after rejection submitted fresh", r2.submitted is True)
    check("rejected retry got new coid", r1.client_order_id != r2.client_order_id)


def t_filled_order_still_deduped():
    """FILLED status MUST keep dedupe (don't accidentally double-fill)."""
    td = tempfile.mkdtemp(prefix="dedupe_filled_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_client()
    req = _make_req()
    with connect(db) as conn:
        r1 = submit_order(client, conn, req, decision_price=50.0)
        conn.execute(
            "UPDATE orders SET status = 'OrderStatus.FILLED' WHERE client_order_id = ?",
            (r1.client_order_id,),
        )
        r2 = submit_order(client, conn, req, decision_price=50.0)
    check("filled order DOES still dedupe (don't double-fill)",
          r2.submitted is False, f"got submitted={r2.submitted}")
    check("broker.submit_order NOT called second time",
          client._n_calls["n"] == 1, f"got {client._n_calls['n']}")


def t_multiple_dead_retries():
    """Multiple consecutive cancellations should keep generating fresh coids."""
    td = tempfile.mkdtemp(prefix="dedupe_multi_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_client()
    req = _make_req()
    coids: set[str] = set()
    with connect(db) as conn:
        for i in range(4):
            r = submit_order(client, conn, req, decision_price=50.0)
            check(f"attempt {i+1} submitted", r.submitted is True)
            coids.add(r.client_order_id)
            # Kill this attempt
            conn.execute(
                "UPDATE orders SET status = 'OrderStatus.CANCELED' WHERE client_order_id = ?",
                (r.client_order_id,),
            )
    check("4 retries produced 4 distinct coids", len(coids) == 4,
          f"got {len(coids)} unique")


def main() -> int:
    print("== Smoketest: dedupe handles dead orders correctly ==\n")
    print("--- 1. Active order keeps dedupe ---")
    t_active_order_still_deduped()
    print("\n--- 2. Canceled order allows re-submit ---")
    t_canceled_order_allows_resubmit()
    print("\n--- 3. Rejected order allows re-submit ---")
    t_rejected_order_allows_resubmit()
    print("\n--- 4. Filled order keeps dedupe (no double-fill) ---")
    t_filled_order_still_deduped()
    print("\n--- 5. Multiple consecutive cancels work ---")
    t_multiple_dead_retries()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
