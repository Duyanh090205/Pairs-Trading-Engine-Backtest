"""Smoketest: entry_z gets persisted from engine decision -> orders -> positions.

Audit finding 2026-05-28: trading_stream._maybe_open_position hardcoded
entry_z=0.0 because it had no way to know the Z that triggered the decision.
The engine had ctx.last_z but never persisted it.

Fix:
  - Add `entry_z` column to orders table (via migration)
  - submit_order accepts entry_z param and writes to orders.entry_z
  - _execute_action passes ctx.last_z to submit_order
  - _maybe_open_position reads entry_z from the leg rows when inserting position
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live.broker.trading_stream import TradingStreamHandler
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


def _raw_conn(db: Path):
    import sqlite3
    c = sqlite3.connect(db, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


def _mock_broker():
    client = MagicMock()
    def _submit(req):
        r = MagicMock()
        r.id = "brk_x"
        r.status = "OrderStatus.ACCEPTED"
        return r
    client.submit_order = _submit
    return client


def t_schema_has_entry_z_column():
    """Migration must add entry_z column to orders."""
    td = tempfile.mkdtemp(prefix="ez_schema_")
    db = Path(td) / "x.db"
    init_db(db)
    with connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(orders)")}
    check("orders.entry_z column exists after init_db",
          "entry_z" in cols, f"got cols: {sorted(cols)}")


def t_submit_writes_entry_z():
    """submit_order(entry_z=3.5) must persist that value in orders."""
    td = tempfile.mkdtemp(prefix="ez_submit_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()
    req = OrderRequest(pair_id="A_B", bar_ts="2026-01-01T20:00:00Z",
                       leg="A", ticker="A", side="buy", qty=10.0,
                       order_type="market")
    with connect(db) as conn:
        submit_order(client, conn, req, decision_price=100.0, entry_z=3.523)
        row = conn.execute(
            "SELECT entry_z FROM orders WHERE client_order_id = ?",
            (req.client_order_id(),)
        ).fetchone()
    check("submit_order persisted entry_z=3.523",
          abs(float(row["entry_z"]) - 3.523) < 1e-9,
          f"got {row['entry_z']}")


def t_submit_entry_z_none_stored_as_null():
    """When entry_z not passed (legacy callers), should be NULL."""
    td = tempfile.mkdtemp(prefix="ez_none_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()
    req = OrderRequest(pair_id="A_B", bar_ts="2026-01-01T20:00:00Z",
                       leg="A", ticker="A", side="buy", qty=10.0,
                       order_type="market")
    with connect(db) as conn:
        submit_order(client, conn, req, decision_price=100.0)  # no entry_z
        row = conn.execute(
            "SELECT entry_z FROM orders WHERE client_order_id = ?",
            (req.client_order_id(),)
        ).fetchone()
    check("submit_order without entry_z stores NULL",
          row["entry_z"] is None, f"got {row['entry_z']}")


def t_fill_inserts_position_with_entry_z():
    """When both legs fill, positions.entry_z must equal the order's entry_z."""
    td = tempfile.mkdtemp(prefix="ez_pos_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()
    req_a = OrderRequest(pair_id="X_Y", bar_ts="2026-01-01T20:00:00Z",
                         leg="A", ticker="X", side="buy", qty=10.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="X_Y", bar_ts="2026-01-01T20:00:00Z",
                         leg="B", ticker="Y", side="sell", qty=5.0,
                         order_type="market")
    expected_z = -3.872
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=100.0,
                           entry_z=expected_z)
        r_b = submit_order(client, conn, req_b, decision_price=200.0,
                           entry_z=expected_z)

    handler = TradingStreamHandler(
        cfg=SimpleNamespace(api_key="", secret_key="", paper=True),
        conn_factory=lambda: _raw_conn(db),
    )

    def _event(coid, ticker, qty, price):
        return SimpleNamespace(
            event="fill",
            order=SimpleNamespace(
                client_order_id=coid, status="OrderStatus.FILLED",
                id=f"brk_{ticker}", filled_qty=str(qty),
                filled_avg_price=str(price), symbol=ticker, side="buy",
            ),
        )
    handler.apply_event(_event(r_a.client_order_id, "X", 10.0, 100.0))
    handler.apply_event(_event(r_b.client_order_id, "Y", 5.0, 200.0))

    with connect(db) as conn:
        pos = conn.execute(
            "SELECT entry_z FROM positions WHERE pair_id = 'X_Y'"
        ).fetchone()
    check("position row inserted", pos is not None)
    if pos:
        check(f"positions.entry_z = expected {expected_z}",
              abs(float(pos["entry_z"]) - expected_z) < 1e-9,
              f"got {pos['entry_z']}")


def t_legacy_legs_without_entry_z_fall_back_to_zero():
    """If the legs were inserted before the migration (entry_z=NULL), the
    position insert should fall back to 0.0 — not crash."""
    td = tempfile.mkdtemp(prefix="ez_legacy_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()
    req_a = OrderRequest(pair_id="L_M", bar_ts="2026-01-01T20:00:00Z",
                         leg="A", ticker="L", side="buy", qty=5.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="L_M", bar_ts="2026-01-01T20:00:00Z",
                         leg="B", ticker="M", side="sell", qty=2.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=50.0)  # no entry_z
        r_b = submit_order(client, conn, req_b, decision_price=125.0)

    handler = TradingStreamHandler(
        cfg=SimpleNamespace(api_key="", secret_key="", paper=True),
        conn_factory=lambda: _raw_conn(db),
    )

    def _event(coid, ticker, qty, price):
        return SimpleNamespace(
            event="fill",
            order=SimpleNamespace(
                client_order_id=coid, status="OrderStatus.FILLED",
                id=f"brk_{ticker}", filled_qty=str(qty),
                filled_avg_price=str(price), symbol=ticker, side="buy",
            ),
        )
    handler.apply_event(_event(r_a.client_order_id, "L", 5.0, 50.0))
    handler.apply_event(_event(r_b.client_order_id, "M", 2.0, 125.0))

    with connect(db) as conn:
        pos = conn.execute(
            "SELECT entry_z FROM positions WHERE pair_id = 'L_M'"
        ).fetchone()
    check("legacy NULL entry_z falls back to 0.0",
          pos is not None and float(pos["entry_z"]) == 0.0,
          f"got {pos['entry_z'] if pos else None}")


def main() -> int:
    print("== Smoketest: entry_z persistence end-to-end ==\n")
    print("--- 1. Schema migration adds entry_z column ---")
    t_schema_has_entry_z_column()
    print("\n--- 2. submit_order persists entry_z ---")
    t_submit_writes_entry_z()
    print("\n--- 3. submit_order without entry_z stores NULL ---")
    t_submit_entry_z_none_stored_as_null()
    print("\n--- 4. Fill chain writes correct entry_z to positions ---")
    t_fill_inserts_position_with_entry_z()
    print("\n--- 5. Legacy legs without entry_z fall back to 0.0 ---")
    t_legacy_legs_without_entry_z_fall_back_to_zero()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
