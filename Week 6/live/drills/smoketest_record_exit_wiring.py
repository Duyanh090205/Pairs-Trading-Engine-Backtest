"""Smoketest: trade_update handler closes positions when exit legs fill.

Bug found Fri 2026-05-29: EOM flatten submitted close orders, broker
filled them, BUT no production code path called record_exit -> positions
table stayed 'open' -> reconcile mismatch -> kill_switch trip.

Fix: _maybe_close_position() runs alongside _maybe_open_position() on
every fill event. When both exit legs of an open pair are filled, it
computes realized P&L and updates positions.exit_ts.
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


def _event(coid, ticker, side, qty, price):
    """Build a fake Alpaca trade_update fill event."""
    return SimpleNamespace(
        event="fill",
        order=SimpleNamespace(
            client_order_id=coid, status="OrderStatus.FILLED",
            id=f"brk_{ticker}", filled_qty=str(qty),
            filled_avg_price=str(price), symbol=ticker, side=side,
        ),
    )


def _make_handler(db: Path) -> TradingStreamHandler:
    return TradingStreamHandler(
        cfg=SimpleNamespace(api_key="", secret_key="", paper=True),
        conn_factory=lambda: _raw_conn(db),
    )


def t_long_position_full_lifecycle():
    """Long A short B: enter at 100/200, exit at 110/190. Realized P&L should match math."""
    td = tempfile.mkdtemp(prefix="rec_long_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()

    # Entry orders (long A=AAA, short B=BBB)
    req_a = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-01T20:00:00Z",
                         leg="A", ticker="AAA", side="buy", qty=10.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-01T20:00:00Z",
                         leg="B", ticker="BBB", side="sell", qty=5.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=100.0, entry_z=3.5)
        r_b = submit_order(client, conn, req_b, decision_price=200.0, entry_z=3.5)

    h = _make_handler(db)
    # Entry fills (open position)
    h.apply_event(_event(r_a.client_order_id, "AAA", "buy", 10.0, 100.0))
    h.apply_event(_event(r_b.client_order_id, "BBB", "sell", 5.0, 200.0))

    # Verify position is OPEN
    with connect(db) as conn:
        pos = conn.execute(
            "SELECT exit_ts, realized_pnl FROM positions WHERE pair_id = 'AAA_BBB'"
        ).fetchone()
    check("position opened after entry fills", pos is not None)
    if pos:
        check("position is OPEN (exit_ts NULL) after entries",
              pos["exit_ts"] is None)

    # Exit orders (reverse side)
    req_a_exit = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-05T19:55:00Z",
                              leg="A_close", ticker="AAA", side="sell", qty=10.0,
                              order_type="market")
    req_b_exit = OrderRequest(pair_id="AAA_BBB", bar_ts="2026-01-05T19:55:00Z",
                              leg="B_close", ticker="BBB", side="buy", qty=5.0,
                              order_type="market")
    with connect(db) as conn:
        x_a = submit_order(client, conn, req_a_exit, decision_price=110.0)
        x_b = submit_order(client, conn, req_b_exit, decision_price=190.0)

    # Exit fills (close position)
    h.apply_event(_event(x_a.client_order_id, "AAA", "sell", 10.0, 110.0))
    h.apply_event(_event(x_b.client_order_id, "BBB", "buy", 5.0, 190.0))

    # Verify position is CLOSED with correct P&L
    # Long AAA: 10 * (110-100) = +100
    # Short BBB: 5 * (200-190) = +50
    # Total: +150
    with connect(db) as conn:
        pos = conn.execute(
            "SELECT exit_ts, realized_pnl, exit_reason FROM positions "
            "WHERE pair_id = 'AAA_BBB'"
        ).fetchone()
    check("position closed (exit_ts set)", pos["exit_ts"] is not None,
          f"got exit_ts={pos['exit_ts']}")
    check("exit_reason set", pos["exit_reason"] == "zero_cross",
          f"got '{pos['exit_reason']}'")
    check("realized P&L = +$150 (long +$100 + short +$50)",
          abs(float(pos["realized_pnl"]) - 150.0) < 0.01,
          f"got {pos['realized_pnl']}")


def t_short_position_full_lifecycle():
    """Short A long B: enter at 50/25, exit at 45/27."""
    td = tempfile.mkdtemp(prefix="rec_short_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()

    req_a = OrderRequest(pair_id="X_Y", bar_ts="2026-02-01T20:00:00Z",
                         leg="A", ticker="X", side="sell", qty=20.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="X_Y", bar_ts="2026-02-01T20:00:00Z",
                         leg="B", ticker="Y", side="buy", qty=40.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=50.0, entry_z=-3.2)
        r_b = submit_order(client, conn, req_b, decision_price=25.0, entry_z=-3.2)

    h = _make_handler(db)
    h.apply_event(_event(r_a.client_order_id, "X", "sell", 20.0, 50.0))
    h.apply_event(_event(r_b.client_order_id, "Y", "buy", 40.0, 25.0))

    # Exit
    req_a_x = OrderRequest(pair_id="X_Y", bar_ts="2026-02-05T19:55:00Z",
                           leg="A_close", ticker="X", side="buy", qty=20.0,
                           order_type="market")
    req_b_x = OrderRequest(pair_id="X_Y", bar_ts="2026-02-05T19:55:00Z",
                           leg="B_close", ticker="Y", side="sell", qty=40.0,
                           order_type="market")
    with connect(db) as conn:
        x_a = submit_order(client, conn, req_a_x, decision_price=45.0)
        x_b = submit_order(client, conn, req_b_x, decision_price=27.0)

    h.apply_event(_event(x_a.client_order_id, "X", "buy", 20.0, 45.0))
    h.apply_event(_event(x_b.client_order_id, "Y", "sell", 40.0, 27.0))

    # Short X: 20 * (50-45) = +100
    # Long Y: 40 * (27-25) = +80
    # Total: +180
    with connect(db) as conn:
        pos = conn.execute(
            "SELECT exit_ts, realized_pnl FROM positions WHERE pair_id = 'X_Y'"
        ).fetchone()
    check("short pair closed with correct P&L",
          pos["exit_ts"] is not None and
          abs(float(pos["realized_pnl"]) - 180.0) < 0.01,
          f"got pnl={pos['realized_pnl']}")


def t_entry_fill_does_not_close_position():
    """When only entry legs fill, position must stay OPEN (no false close)."""
    td = tempfile.mkdtemp(prefix="rec_open_only_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()

    req_a = OrderRequest(pair_id="P_Q", bar_ts="2026-03-01T20:00:00Z",
                         leg="A", ticker="P", side="buy", qty=5.0,
                         order_type="market")
    req_b = OrderRequest(pair_id="P_Q", bar_ts="2026-03-01T20:00:00Z",
                         leg="B", ticker="Q", side="sell", qty=10.0,
                         order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=50.0, entry_z=3.0)
        r_b = submit_order(client, conn, req_b, decision_price=25.0, entry_z=3.0)

    h = _make_handler(db)
    h.apply_event(_event(r_a.client_order_id, "P", "buy", 5.0, 50.0))
    h.apply_event(_event(r_b.client_order_id, "Q", "sell", 10.0, 25.0))

    with connect(db) as conn:
        pos = conn.execute(
            "SELECT exit_ts, realized_pnl FROM positions WHERE pair_id = 'P_Q'"
        ).fetchone()
    check("position stays OPEN after only entry fills (no false close)",
          pos["exit_ts"] is None and pos["realized_pnl"] is None,
          f"got exit_ts={pos['exit_ts']}, pnl={pos['realized_pnl']}")


def t_partial_exit_does_not_close():
    """When only ONE exit leg fills, position should stay open."""
    td = tempfile.mkdtemp(prefix="rec_partial_exit_")
    db = Path(td) / "x.db"
    init_db(db)
    client = _mock_broker()

    req_a = OrderRequest(pair_id="A_B", bar_ts="t1", leg="A", ticker="A",
                         side="buy", qty=10.0, order_type="market")
    req_b = OrderRequest(pair_id="A_B", bar_ts="t1", leg="B", ticker="B",
                         side="sell", qty=5.0, order_type="market")
    with connect(db) as conn:
        r_a = submit_order(client, conn, req_a, decision_price=100.0, entry_z=3.0)
        r_b = submit_order(client, conn, req_b, decision_price=200.0, entry_z=3.0)
    h = _make_handler(db)
    h.apply_event(_event(r_a.client_order_id, "A", "buy", 10.0, 100.0))
    h.apply_event(_event(r_b.client_order_id, "B", "sell", 5.0, 200.0))

    # Only exit A fills
    req_a_x = OrderRequest(pair_id="A_B", bar_ts="t2", leg="A_close", ticker="A",
                           side="sell", qty=10.0, order_type="market")
    with connect(db) as conn:
        x_a = submit_order(client, conn, req_a_x, decision_price=110.0)
    h.apply_event(_event(x_a.client_order_id, "A", "sell", 10.0, 110.0))

    with connect(db) as conn:
        pos = conn.execute(
            "SELECT exit_ts FROM positions WHERE pair_id = 'A_B'"
        ).fetchone()
    check("position stays OPEN with only one exit leg filled",
          pos["exit_ts"] is None, f"got exit_ts={pos['exit_ts']}")


def main() -> int:
    print("== Smoketest: trade_update handler closes positions ==\n")
    print("--- 1. Long lifecycle: long A short B ---")
    t_long_position_full_lifecycle()
    print("\n--- 2. Short lifecycle: short A long B ---")
    t_short_position_full_lifecycle()
    print("\n--- 3. Entry-only fills: position stays open ---")
    t_entry_fill_does_not_close_position()
    print("\n--- 4. Partial exit (1 of 2 legs): position stays open ---")
    t_partial_exit_does_not_close()
    print()
    if errors:
        print(f"{RED}FAIL: {len(errors)} - {errors}{RESET}")
        return 1
    print(f"{GREEN}PASS{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
