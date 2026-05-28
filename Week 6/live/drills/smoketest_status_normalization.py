"""Smoketest: status normalization + fill -> position flow works end-to-end.

Bug discovered live 2026-05-28: Alpaca returns status as 'OrderStatus.FILLED'
but downstream code (_is_final_fill, _maybe_open_position, reconciliation)
compared against 'filled' lowercase -> positions table never populated ->
reconcile mismatch -> kill_switch halted at 9:37am ET.

Fix: normalize_status() helper called at every write site.
This test verifies the full chain works end-to-end with raw Alpaca strings.
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
from live.execution.order_manager import (
    OrderRequest, normalize_status, submit_order,
)
from live.execution.reconciliation import reconcile
from live.state.persist import connect, init_db


def _raw_conn(db: Path):
    """Open a raw (non-context-managed) connection like the engine does."""
    import sqlite3
    c = sqlite3.connect(db, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c

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


def _mock_broker(status_str: str):
    """Mock that returns the given status string (e.g. 'OrderStatus.ACCEPTED')."""
    client = MagicMock()
    def _submit(req):
        r = MagicMock()
        r.id = "broker_x"
        r.status = status_str
        return r
    client.submit_order = _submit
    return client


def t_normalize_status_unit():
    """Spot-check the helper."""
    check("OrderStatus.FILLED -> filled",
          normalize_status("OrderStatus.FILLED") == "filled")
    check("OrderStatus.ACCEPTED -> submitted",
          normalize_status("OrderStatus.ACCEPTED") == "submitted")
    check("OrderStatus.PARTIALLY_FILLED -> partial",
          normalize_status("OrderStatus.PARTIALLY_FILLED") == "partial")
    check("OrderStatus.CANCELED -> canceled",
          normalize_status("OrderStatus.CANCELED") == "canceled")
    check("lowercase 'filled' -> filled (idempotent)",
          normalize_status("filled") == "filled")
    check("empty -> empty", normalize_status("") == "")


def t_submit_order_writes_canonical_status():
    """submit_order should store 'submitted' not 'OrderStatus.ACCEPTED'."""
    td = tempfile.mkdtemp(prefix="ns_submit_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker("OrderStatus.ACCEPTED")
    req = OrderRequest(
        pair_id="X_Y", bar_ts="2026-01-01T20:00:00Z", leg="A",
        ticker="X", side="buy", qty=10.0, order_type="market",
    )
    with connect(db) as conn:
        submit_order(client, conn, req, decision_price=100.0)
        row = conn.execute(
            "SELECT status FROM orders WHERE client_order_id = ?",
            (req.client_order_id(),)
        ).fetchone()
    check("submit_order stored canonical status",
          row["status"] == "submitted", f"got '{row['status']}'")


def t_trade_update_fill_inserts_position():
    """Both-legs-filled WebSocket events -> positions table gets a row."""
    td = tempfile.mkdtemp(prefix="ns_fill_")
    db = Path(td) / "x.db"
    init_db(db)
    # First, insert 2 entry orders
    client = _mock_broker("OrderStatus.ACCEPTED")
    req_a = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-01T20:00:00Z",
                         leg="A", ticker="AAA", side="buy", qty=10.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-01T20:00:00Z",
                         leg="B", ticker="BBB", side="sell", qty=5.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=100.0)
        r_b = submit_order(client, conn, req_b, decision_price=200.0)

    # Now apply WebSocket fill events with raw Alpaca-style status strings
    handler = TradingStreamHandler(
        cfg=SimpleNamespace(api_key="", secret_key="", paper=True),
        conn_factory=lambda: _raw_conn(db),
    )
    def _make_event(coid, status_str, filled_qty, price):
        return SimpleNamespace(
            event="fill",
            order=SimpleNamespace(
                client_order_id=coid, status=status_str, id="brk_1",
                filled_qty=str(filled_qty), filled_avg_price=str(price),
                symbol="AAA", side="buy",
            ),
        )

    handler.apply_event(_make_event(r_a.client_order_id, "OrderStatus.FILLED",
                                     10.0, 101.0))
    handler.apply_event(_make_event(r_b.client_order_id, "OrderStatus.FILLED",
                                     5.0, 199.5))

    with connect(db) as conn:
        order_statuses = {
            r["ticker"]: r["status"] for r in conn.execute(
                "SELECT ticker, status FROM orders"
            )
        }
        pos_rows = conn.execute(
            "SELECT pair_id, side_a, side_b, direction, notional_a, notional_b, exit_ts "
            "FROM positions"
        ).fetchall()

    check("AAA order status canonical 'filled'",
          order_statuses.get("AAA") == "filled",
          f"got '{order_statuses.get('AAA')}'")
    check("BBB order status canonical 'filled'",
          order_statuses.get("BBB") == "filled",
          f"got '{order_statuses.get('BBB')}'")
    check("position row inserted after both fills",
          len(pos_rows) == 1, f"got {len(pos_rows)} positions")
    if pos_rows:
        p = pos_rows[0]
        check("position direction = +1 (leg A = buy)", p["direction"] == 1)
        check("position notional_a > 0",
              p["notional_a"] > 0, f"got {p['notional_a']}")
        check("position exit_ts is NULL (open)", p["exit_ts"] is None)


def t_reconcile_finds_zero_mismatches_after_proper_fill():
    """After fills create position row, reconcile vs broker should be clean."""
    td = tempfile.mkdtemp(prefix="ns_rec_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker("OrderStatus.ACCEPTED")
    req_a = OrderRequest(pair_id="C_D", bar_ts="2026-01-01T20:00:00Z",
                         leg="A", ticker="C", side="buy", qty=8.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="C_D", bar_ts="2026-01-01T20:00:00Z",
                         leg="B", ticker="D", side="sell", qty=4.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=100.0)
        r_b = submit_order(client, conn, req_b, decision_price=200.0)

    handler = TradingStreamHandler(
        cfg=SimpleNamespace(api_key="", secret_key="", paper=True),
        conn_factory=lambda: _raw_conn(db),
    )

    def _make_event(coid, ticker, side, qty, price):
        return SimpleNamespace(
            event="fill",
            order=SimpleNamespace(
                client_order_id=coid, status="OrderStatus.FILLED",
                id=f"brk_{ticker}", filled_qty=str(qty),
                filled_avg_price=str(price), symbol=ticker, side=side,
            ),
        )

    handler.apply_event(_make_event(r_a.client_order_id, "C", "buy", 8.0, 100.0))
    handler.apply_event(_make_event(r_b.client_order_id, "D", "sell", 4.0, 200.0))

    # Mock broker reporting matching positions
    broker_client = MagicMock()
    broker_client.get_all_positions.return_value = [
        SimpleNamespace(symbol="C", qty="8.0"),
        SimpleNamespace(symbol="D", qty="-4.0"),
    ]
    with connect(db) as conn:
        mismatches = reconcile(broker_client, conn, tolerance_shares=1.0)
    check("reconcile finds 0 mismatches after proper fill flow",
          len(mismatches) == 0, f"got {[m.__dict__ for m in mismatches]}")


def t_legacy_orderstatus_rows_dont_break_reconcile():
    """Existing rows with 'OrderStatus.FILLED' should still reconcile via the
    LOWER+REPLACE compatibility in the SQL queries."""
    td = tempfile.mkdtemp(prefix="ns_legacy_")
    db = Path(td) / "x.db"
    init_db(db)
    # Manually insert a legacy-style row + position
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, "
            "qty, order_type, status, submitted_ts, fill_qty, fill_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy_a", "L_M", "2026-01-01T20:00:00Z", "L", "buy", 5.0,
             "market", "OrderStatus.FILLED", "2026-01-01T20:00:00Z", 5.0, 50.0),
        )
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, "
            "qty, order_type, status, submitted_ts, fill_qty, fill_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy_b", "L_M", "2026-01-01T20:00:00Z", "M", "sell", 2.0,
             "market", "OrderStatus.FILLED", "2026-01-01T20:00:00Z", 2.0, 125.0),
        )
        conn.execute(
            "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
            "notional_a, notional_b, entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("L_M", "L", "M", 1.0, 1, 250.0, 250.0,
             "2026-01-01T20:00:00Z", 0.0),
        )

    broker_client = MagicMock()
    broker_client.get_all_positions.return_value = [
        SimpleNamespace(symbol="L", qty="5.0"),
        SimpleNamespace(symbol="M", qty="-2.0"),
    ]
    with connect(db) as conn:
        mismatches = reconcile(broker_client, conn, tolerance_shares=1.0)
    check("legacy OrderStatus.FILLED rows reconcile cleanly via LOWER+REPLACE",
          len(mismatches) == 0, f"got {[m.__dict__ for m in mismatches]}")


def main() -> int:
    print("== Smoketest: status normalization + fill->position flow ==\n")
    print("--- 1. Unit: normalize_status() ---")
    t_normalize_status_unit()
    print("\n--- 2. submit_order writes canonical 'submitted' ---")
    t_submit_order_writes_canonical_status()
    print("\n--- 3. Fill events insert position row ---")
    t_trade_update_fill_inserts_position()
    print("\n--- 4. Reconcile finds 0 mismatches after proper fill ---")
    t_reconcile_finds_zero_mismatches_after_proper_fill()
    print("\n--- 5. Legacy 'OrderStatus.FILLED' rows reconcile (back-compat) ---")
    t_legacy_orderstatus_rows_dont_break_reconcile()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
