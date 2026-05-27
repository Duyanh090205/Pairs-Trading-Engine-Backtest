"""Phase 2 smoketest — engine + orders + reconcile + regime delegate.

Checks:
  1. decide() matches backtest _state_machine_daily on the same Z series (cross-path)
  2. client_order_id is unique for {market vs limit} + {different limit_price} (Day-2 Bug 2 fix)
  3. submit_order: hardstop tripped → refuses, audit-log entry created
  4. submit_order: kill_switch halted → refuses
  5. submit_order: duplicate client_order_id → returns existing (no double insert)
  6. submit_order: clean path → inserts row in orders table
  7. reconcile detects qty mismatch
  8. trip_halt_on_mismatch sets kill_switch
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
failures: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"{GREEN}PASS{RESET}  {name}")
    except Exception as e:
        print(f"{RED}FAIL{RESET}  {name}: {type(e).__name__}: {e}")
        failures.append(name)


def t_decide_matches_backtest_state_machine():
    """Cross-path: live decide() one-step matches backtest vectorized state machine."""
    from live.engine_live.live_pair import decide
    from engine_daily.engine_daily import _state_machine_daily

    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 200).astype(np.float64)
    z[10:15] = np.nan
    z[30] = -3.5  # entry
    z[50] = 0.1   # zero-cross exit
    z[80] = 5.5   # hard SL
    entry_z = 3.0
    hard_sl_z = 5.0

    # Backtest (vectorized numba)
    bt_pos, _ = _state_machine_daily(z, entry_z, hard_sl_z, initial_state=0)

    # Live: one-step per bar
    state = 0
    live_pos = np.zeros(len(z), dtype=np.int8)
    for i, zi in enumerate(z):
        d = decide(state, float(zi) if not np.isnan(zi) else float("nan"),
                   entry_z=entry_z, hard_sl_z=hard_sl_z)
        state = d.new_state
        live_pos[i] = state

    assert np.array_equal(live_pos, bt_pos), \
        f"divergence at bars: {np.where(live_pos != bt_pos)[0][:5]}"


def t_client_order_id_includes_order_type_and_limit():
    """Day-2 Bug 2 fix: market and limit orders with same params get DIFFERENT coids."""
    from live.execution.order_manager import OrderRequest
    market = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 100.0, "market")
    limit_a = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 100.0, "limit", 200.0)
    limit_b = OrderRequest("JPM_BAC", "2026-06-01T20:00:00Z", "A", "JPM", "buy", 100.0, "limit", 200.5)
    assert market.client_order_id() != limit_a.client_order_id()
    assert limit_a.client_order_id() != limit_b.client_order_id()


def _fresh_db():
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="phase2_")
    db = Path(td) / "state.db"
    init_db(db)
    return db, td


class _FakeBrokerOK:
    def submit_order(self, req):
        return type("R", (), {"id": "broker_id_42", "status": "accepted"})()

    def get_all_positions(self):
        return [type("P", (), {"symbol": "JPM", "qty": "100"})()]


def t_submit_order_refuses_on_hardstop():
    from live.execution.order_manager import OrderRequest, submit_order
    from live.state.persist import connect
    from live.safety import hardstop
    db, td = _fresh_db()
    flag = Path(td) / "HARDSTOP.flag"
    flag.write_text("test\nmanual\n")
    hardstop.HARDSTOP_FLAG_PATH = flag
    req = OrderRequest("X_Y", "2026-06-01T20:00:00Z", "A", "X", "buy", 10, "market")
    with connect(db) as conn:
        out = submit_order(_FakeBrokerOK(), conn, req, decision_price=100.0)
    assert out.status == "refused_hardstop", out
    assert out.refused_reason == "hardstop_tripped"
    assert out.submitted is False
    flag.unlink()


def t_submit_order_refuses_on_killswitch_halt():
    from live.execution.order_manager import OrderRequest, submit_order
    from live.state.persist import connect, set_halt
    from live.safety import hardstop
    db, td = _fresh_db()
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"   # ensure not tripped
    req = OrderRequest("X_Y", "2026-06-01T20:00:00Z", "A", "X", "buy", 10, "market")
    with connect(db) as conn:
        set_halt(conn, "test_halt_reason")
        out = submit_order(_FakeBrokerOK(), conn, req, decision_price=100.0)
    assert out.status == "refused_halt", out
    assert "test_halt_reason" in (out.refused_reason or "")


def t_submit_order_idempotency():
    from live.execution.order_manager import OrderRequest, submit_order
    from live.state.persist import connect
    from live.safety import hardstop
    db, td = _fresh_db()
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"
    req = OrderRequest("X_Y", "2026-06-01T20:00:00Z", "A", "X", "buy", 10, "market")
    with connect(db) as conn:
        out1 = submit_order(_FakeBrokerOK(), conn, req, decision_price=100.0)
        out2 = submit_order(_FakeBrokerOK(), conn, req, decision_price=100.0)
        n = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
    assert out1.submitted is True
    assert out2.submitted is False, "second call should hit dedupe path"
    assert out1.client_order_id == out2.client_order_id
    assert n == 1, f"expected 1 row in orders, got {n}"


def t_submit_order_clean_path_inserts():
    from live.execution.order_manager import OrderRequest, submit_order
    from live.state.persist import connect
    from live.safety import hardstop
    db, td = _fresh_db()
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"
    req = OrderRequest("X_Y", "2026-06-01T20:00:00Z", "A", "X", "buy", 10, "market")
    with connect(db) as conn:
        out = submit_order(_FakeBrokerOK(), conn, req, decision_price=100.0)
        row = conn.execute("SELECT ticker, qty, side, status FROM orders").fetchone()
    assert out.submitted is True
    assert out.broker_order_id == "broker_id_42"
    assert row["ticker"] == "X" and row["qty"] == 10 and row["side"] == "buy"


def t_reconcile_detects_mismatch():
    from live.execution.reconciliation import reconcile, trip_halt_on_mismatch
    from live.state.persist import connect, is_halted
    db, td = _fresh_db()
    # Broker says 100 shares JPM; locally we have 0 open positions → mismatch
    with connect(db) as conn:
        mm = reconcile(_FakeBrokerOK(), conn, tolerance_shares=1.0)
        assert len(mm) == 1, mm
        assert mm[0].ticker == "JPM" and mm[0].broker_qty == 100 and mm[0].local_qty == 0
        tripped = trip_halt_on_mismatch(conn, mm)
        assert tripped is True
        halted, _ = is_halted(conn)
        assert halted is True


def t_trading_stream_propagates_fills():
    """Fix Bug A: TradingStream apply_event updates orders.fill_qty + status to 'filled'."""
    from datetime import datetime, timezone
    from live.broker.trading_stream import TradingStreamHandler
    from live.execution.order_manager import OrderRequest, submit_order
    from live.safety import hardstop
    from live.state.persist import connect

    db, td = _fresh_db()
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"
    bar_ts = "2026-06-01T20:00:00Z"
    reqA = OrderRequest("JPM_BAC", bar_ts, "A", "JPM", "buy", 100.0, "market")
    reqB = OrderRequest("JPM_BAC", bar_ts, "B", "BAC", "sell", 120.0, "market")
    with connect(db) as conn:
        out_a = submit_order(_FakeBrokerOK(), conn, reqA, decision_price=190.0)
        out_b = submit_order(_FakeBrokerOK(), conn, reqB, decision_price=33.0)
    coid_a = out_a.client_order_id
    coid_b = out_b.client_order_id

    handler = TradingStreamHandler(cfg=None, conn_factory=lambda: sqlite3.connect(db))
    # Patch connection factory to set row_factory like persist.connect does
    def _factory():
        c = sqlite3.connect(db, isolation_level=None)
        c.execute("PRAGMA foreign_keys = ON")
        c.row_factory = sqlite3.Row
        return c
    handler.conn_factory = _factory

    fake_fill_a = {
        "event": "fill",
        "order": {
            "client_order_id": coid_a, "id": "broker_id_42", "symbol": "JPM",
            "side": "buy", "status": "filled", "filled_qty": 100.0,
            "filled_avg_price": 190.05,
        },
    }
    fake_fill_b = {
        "event": "fill",
        "order": {
            "client_order_id": coid_b, "id": "broker_id_43", "symbol": "BAC",
            "side": "sell", "status": "filled", "filled_qty": 120.0,
            "filled_avg_price": 33.02,
        },
    }
    fe_a = handler.apply_event(fake_fill_a)
    fe_b = handler.apply_event(fake_fill_b)
    assert fe_a is not None and fe_a.is_final_fill
    assert fe_b is not None and fe_b.is_final_fill

    with connect(db) as conn:
        rows = conn.execute(
            "SELECT ticker, status, fill_qty, fill_price FROM orders ORDER BY ticker"
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["status"].lower() == "filled", r["status"]
            assert r["fill_qty"] > 0
            assert r["fill_price"] > 0
        # Bug E fix: positions row was opened
        pos = conn.execute(
            "SELECT pair_id, direction, notional_a, notional_b FROM positions"
        ).fetchall()
        assert len(pos) == 1
        assert pos[0]["pair_id"] == "JPM_BAC"
        assert pos[0]["direction"] == 1, "buy A → direction +1"
        assert pos[0]["notional_a"] > 0
        assert pos[0]["notional_b"] > 0


def t_reconcile_clean_after_fills():
    """After Bug A + E fixes: reconcile is CLEAN when broker matches local positions."""
    from live.broker.trading_stream import TradingStreamHandler
    from live.execution.order_manager import OrderRequest, submit_order
    from live.execution.reconciliation import reconcile
    from live.safety import hardstop
    from live.state.persist import connect

    db, td = _fresh_db()
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"
    bar_ts = "2026-06-01T20:00:00Z"
    reqA = OrderRequest("JPM_BAC", bar_ts, "A", "JPM", "buy", 100.0, "market")
    reqB = OrderRequest("JPM_BAC", bar_ts, "B", "BAC", "sell", 120.0, "market")
    with connect(db) as conn:
        submit_order(_FakeBrokerOK(), conn, reqA, decision_price=190.0)
        submit_order(_FakeBrokerOK(), conn, reqB, decision_price=33.0)

    def _factory():
        c = sqlite3.connect(db, isolation_level=None)
        c.execute("PRAGMA foreign_keys = ON")
        c.row_factory = sqlite3.Row
        return c
    handler = TradingStreamHandler(cfg=None, conn_factory=_factory)
    handler.apply_event({
        "event": "fill",
        "order": {"client_order_id": reqA.client_order_id(), "id": "b1",
                  "symbol": "JPM", "side": "buy", "status": "filled",
                  "filled_qty": 100.0, "filled_avg_price": 190.05},
    })
    handler.apply_event({
        "event": "fill",
        "order": {"client_order_id": reqB.client_order_id(), "id": "b2",
                  "symbol": "BAC", "side": "sell", "status": "filled",
                  "filled_qty": 120.0, "filled_avg_price": 33.02},
    })

    class _BrokerMatchingLocal:
        def get_all_positions(self):
            return [
                type("P", (), {"symbol": "JPM", "qty": "100"})(),
                type("P", (), {"symbol": "BAC", "qty": "-120"})(),
            ]

    with connect(db) as conn:
        mm = reconcile(_BrokerMatchingLocal(), conn, tolerance_shares=1.0)
        assert mm == [], f"expected clean reconcile, got {mm}"


def main() -> int:
    print(f"{YELLOW}== Phase 2 Smoketest =={RESET}")
    check("decide_matches_backtest", t_decide_matches_backtest_state_machine)
    check("client_order_id_uniqueness", t_client_order_id_includes_order_type_and_limit)
    check("submit_refuses_on_hardstop", t_submit_order_refuses_on_hardstop)
    check("submit_refuses_on_killswitch", t_submit_order_refuses_on_killswitch_halt)
    check("submit_idempotency", t_submit_order_idempotency)
    check("submit_clean_inserts", t_submit_order_clean_path_inserts)
    check("reconcile_detects_mismatch", t_reconcile_detects_mismatch)
    check("trading_stream_propagates_fills (fix Bug A+E)", t_trading_stream_propagates_fills)
    check("reconcile_clean_after_fills (fix Bug A+E)", t_reconcile_clean_after_fills)
    if failures:
        print(f"{RED}{len(failures)} FAILED:{RESET} {failures}")
        return 1
    print(f"{GREEN}All Phase 2 smoketests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
