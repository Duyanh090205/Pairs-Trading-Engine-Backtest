"""End-to-end audit: full entry → fill → position → exit → close cycle through
ALL production code paths (decide → submit_order → TradingStream → reconcile →
trade_journal). Tests CROSS-MODULE wiring, not single-module unit logic.

If this passes, the modules are wired correctly. If it fails, the bug is in
integration — exactly the deep-audit-bug category.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.broker.trading_stream import TradingStreamHandler
from live.engine_live.live_pair import decide
from live.execution.order_manager import OrderRequest, submit_order
from live.execution.reconciliation import reconcile, trip_halt_on_mismatch
from live.monitor.trade_journal import list_recent, record_entry, record_exit
from live.safety import hardstop
from live.state.persist import connect, init_db


class _Broker:
    """Mimics Alpaca paper broker — accepts orders, lets us inject 'fills' later."""
    def __init__(self):
        self.submitted: list = []
        self._next_id = 1
    def submit_order(self, ord_req):
        oid = f"broker_{self._next_id:03d}"
        self._next_id += 1
        self.submitted.append((oid, ord_req))
        return type("R", (), {"id": oid, "status": "accepted"})()
    def get_all_positions(self):
        return getattr(self, "_positions", [])


def _factory(db):
    def f():
        c = sqlite3.connect(db, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c
    return f


def main() -> int:
    print("== E2E Full Pipeline Audit ==")
    print("  scenario: JPM_BAC pair -- enter long @ Z=-3.5 -> fill -> exit @ Z=0.1 -> close")

    td = tempfile.mkdtemp(prefix="e2e_")
    db = Path(td) / "state.db"
    init_db(db)
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"
    broker = _Broker()
    ts = TradingStreamHandler(cfg=None, conn_factory=_factory(db))

    errors: list[str] = []
    def check(name, cond, detail=""):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
        if not cond:
            errors.append(name)

    # ------------------- ENTRY -------------------
    state = 0
    z_entry = -3.5
    d = decide(state, z_entry, entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=0, z=-3.5) returns enter_long",
          d.action == "enter_long" and d.new_state == 1, f"got {d}")

    # Submit both legs at the SAME bar_ts (groups for position opening)
    bar_ts_entry = "2026-06-01T20:00:00Z"
    req_a = OrderRequest("JPM_BAC", bar_ts_entry, "A", "JPM", "buy", 50.0, "market")
    req_b = OrderRequest("JPM_BAC", bar_ts_entry, "B", "BAC", "sell", 110.0, "market")
    with connect(db) as conn:
        out_a = submit_order(broker, conn, req_a, decision_price=190.00)
        out_b = submit_order(broker, conn, req_b, decision_price=33.00)
    check("both entry orders submitted (broker IDs assigned)",
          out_a.broker_order_id and out_b.broker_order_id,
          f"A={out_a.broker_order_id} B={out_b.broker_order_id}")

    # ------------------- FILLS via TradingStream -------------------
    for coid, sym, side, qty, px in [
        (out_a.client_order_id, "JPM", "buy", 50.0, 190.05),
        (out_b.client_order_id, "BAC", "sell", 110.0, 33.01),
    ]:
        ts.apply_event({"event": "fill", "order": {
            "client_order_id": coid, "id": f"b_{sym}", "symbol": sym,
            "side": side, "status": "filled",
            "filled_qty": qty, "filled_avg_price": px,
        }})

    with connect(db) as conn:
        orders = conn.execute(
            "SELECT ticker, status, fill_qty, fill_price FROM orders ORDER BY ticker"
        ).fetchall()
        positions = conn.execute(
            "SELECT pair_id, direction, notional_a, notional_b, exit_ts "
            "FROM positions"
        ).fetchall()
    check("both orders show status='filled'",
          all(r["status"].lower() == "filled" for r in orders),
          f"statuses: {[r['status'] for r in orders]}")
    check("position opened on both-legs-filled",
          len(positions) == 1 and positions[0]["exit_ts"] is None,
          f"positions: {len(positions)}")
    check("position direction +1 (long-A short-B from 'buy' on leg A)",
          positions[0]["direction"] == 1 if positions else False)

    # ------------------- RECONCILE while open -------------------
    broker._positions = [
        type("P", (), {"symbol": "JPM", "qty": "50"})(),
        type("P", (), {"symbol": "BAC", "qty": "-110"})(),
    ]
    with connect(db) as conn:
        mm = reconcile(broker, conn, tolerance_shares=1.0)
    check("reconcile clean (broker matches local positions)",
          mm == [], f"mismatches: {mm}")

    # Record predicted cost as entry attribution
    with connect(db) as conn:
        record_entry(conn, "JPM_BAC", predicted_cost_bps=22.0)

    # ------------------- EXIT -------------------
    z_exit = 0.1  # zero-cross
    d = decide(state=1, z=z_exit, entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=+1, z=0.1) returns exit_zero",
          d.action == "exit_zero" and d.new_state == 0, f"got {d}")

    bar_ts_exit = "2026-06-08T20:00:00Z"
    # Closing trade: SELL the long leg, BUY-to-cover the short leg
    req_a_close = OrderRequest("JPM_BAC", bar_ts_exit, "A", "JPM", "sell", 50.0, "market")
    req_b_close = OrderRequest("JPM_BAC", bar_ts_exit, "B", "BAC", "buy", 110.0, "market")
    with connect(db) as conn:
        out_ac = submit_order(broker, conn, req_a_close, decision_price=192.00)
        out_bc = submit_order(broker, conn, req_b_close, decision_price=32.80)

    for coid, sym, side, qty, px in [
        (out_ac.client_order_id, "JPM", "sell", 50.0, 191.95),
        (out_bc.client_order_id, "BAC", "buy", 110.0, 32.82),
    ]:
        ts.apply_event({"event": "fill", "order": {
            "client_order_id": coid, "id": f"b_{sym}_close", "symbol": sym,
            "side": side, "status": "filled",
            "filled_qty": qty, "filled_avg_price": px,
        }})

    # Record exit attribution (engine would compute realized PnL from fills)
    # Long JPM at $190.05 → sold at $191.95 → +$95.00
    # Short BAC at $33.01 → covered at $32.82 → +$20.90
    realized_pnl = 50 * (191.95 - 190.05) + 110 * (33.01 - 32.82)
    with connect(db) as conn:
        record_exit(conn, "JPM_BAC", exit_ts=bar_ts_exit, exit_z=z_exit,
                    exit_reason="zero_cross",
                    realized_pnl=realized_pnl, realized_cost_bps=24.5)

    with connect(db) as conn:
        closed = conn.execute(
            "SELECT exit_ts, exit_z, exit_reason, realized_pnl, "
            "predicted_cost_bps, realized_cost_bps FROM positions"
        ).fetchone()
        attrs = list_recent(conn, limit=1)
    check("position has exit_ts populated",
          closed["exit_ts"] is not None)
    check("realized_pnl recorded with correct sign + magnitude",
          abs(float(closed["realized_pnl"]) - realized_pnl) < 1e-9,
          f"got {closed['realized_pnl']}, expected {realized_pnl}")
    check("trade_journal attribution: predicted=22, realized=24.5, drift=2.5",
          attrs[0].predicted_cost_bps == 22.0
          and attrs[0].realized_cost_bps == 24.5
          and abs(attrs[0].drift_bps - 2.5) < 1e-9)

    # ------------------- RECONCILE after close -------------------
    broker._positions = []  # broker flattened too
    with connect(db) as conn:
        mm_after = reconcile(broker, conn, tolerance_shares=1.0)
    check("reconcile after close: no mismatch", mm_after == [], f"mm: {mm_after}")

    # ------------------- SANITY: total orders + audit log -------------------
    with connect(db) as conn:
        n_orders = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
        n_audit = conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"]
    check("4 orders persisted (2 entry + 2 exit)", n_orders == 4, f"got {n_orders}")
    check("audit log populated", n_audit >= 8,
          f"got {n_audit} entries (expected >=8 for 4 submits + 4 fills)")

    print()
    if not errors:
        print("E2E PASS — full entry/fill/exit cycle works end-to-end.")
        return 0
    print(f"E2E FAIL — {len(errors)} step(s) failed: {errors}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
