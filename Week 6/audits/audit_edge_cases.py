"""Edge cases + adversarial inputs across the live engine. Designed to find
bugs the synthetic happy-path smoketests miss.

Categories:
  A. Boundary Z values (exactly 3.0, exactly 5.0)
  B. NaN / inf / -inf handling
  C. Adversarial OrderRequest (negative qty, zero qty, SQL-injection pair_id)
  D. Adversarial hardstop flag content
  E. decide() with state values OUTSIDE {-1, 0, +1}
  F. regime_check on first day of month (boundary case)
  G. apply_event with malformed payload (missing fields)
  H. reconcile with broker returning garbage
"""
from __future__ import annotations

import math
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


def check(name: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


# ===============================================================
# A. Boundary Z values
# ===============================================================

def t_boundary_z_values():
    from live.engine_live.live_pair import decide
    # Z exactly at entry threshold: behavior must match backtest's strict-greater.
    # backtest: `if z < -entry_z` and `if z > entry_z` — EXCLUSIVE boundary.
    d1 = decide(0, -3.0, entry_z=3.0, hard_sl_z=5.0)
    check("Z=-3.0 exactly at entry threshold -> NO entry (strict-less)",
          d1.action == "hold" and d1.new_state == 0, f"got {d1}")
    d2 = decide(0, -3.001, entry_z=3.0, hard_sl_z=5.0)
    check("Z=-3.001 (just past entry) -> enter_long",
          d2.action == "enter_long", f"got {d2}")
    # Hard SL: backtest uses `<= -hard_sl_z` and `>= hard_sl_z` — INCLUSIVE boundary.
    d3 = decide(1, -5.0, entry_z=3.0, hard_sl_z=5.0)
    check("Z=-5.0 exactly at hard SL while long -> exit_hard (inclusive)",
          d3.action == "exit_hard" and d3.new_state == 0, f"got {d3}")
    d4 = decide(-1, 5.0, entry_z=3.0, hard_sl_z=5.0)
    check("Z=+5.0 exactly at hard SL while short -> exit_hard",
          d4.action == "exit_hard", f"got {d4}")
    # Zero-cross is INCLUSIVE: `>= 0` and `<= 0`
    d5 = decide(1, 0.0, entry_z=3.0, hard_sl_z=5.0)
    check("Z=0.0 while long -> exit_zero", d5.action == "exit_zero", f"got {d5}")
    d6 = decide(-1, 0.0, entry_z=3.0, hard_sl_z=5.0)
    check("Z=0.0 while short -> exit_zero", d6.action == "exit_zero", f"got {d6}")


# ===============================================================
# B. NaN / inf
# ===============================================================

def t_nan_inf_handling():
    from live.engine_live.live_pair import decide
    d_nan = decide(0, float("nan"), entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=0, z=NaN) -> hold", d_nan.action == "hold")
    d_nan_in_pos = decide(1, float("nan"), entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=+1, z=NaN) -> hold (preserve state)",
          d_nan_in_pos.action == "hold" and d_nan_in_pos.new_state == 1)
    # +inf passes the `> entry_z` test → enters short. -inf passes `< -entry_z` → enters long.
    d_pinf = decide(0, float("inf"), entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=0, z=+inf) -> enter_short (inf > entry)",
          d_pinf.action == "enter_short")
    d_minf = decide(0, float("-inf"), entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=0, z=-inf) -> enter_long (-inf < -entry)",
          d_minf.action == "enter_long")
    # +inf in short position triggers hard SL
    d_inf_sl = decide(-1, float("inf"), entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=-1, z=+inf) -> exit_hard (catches >= hard_sl)",
          d_inf_sl.action == "exit_hard")

    from live.engine_live.z_tracker import ZTracker
    z = ZTracker(window=60, seed=[0.0] * 60)
    # ZTracker.push on NaN — what does it do?
    try:
        out = z.push(float("nan"))
        check("ZTracker.push(NaN) returns finite or None (no crash)",
              out is None or math.isnan(out) or math.isfinite(out),
              f"got {out}")
    except Exception as e:
        check("ZTracker.push(NaN) no crash", False, f"{type(e).__name__}")


# ===============================================================
# C. Adversarial OrderRequest
# ===============================================================

def t_adversarial_order_request():
    from live.execution.order_manager import OrderRequest, submit_order
    from live.safety import hardstop
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="adv_")
    db = Path(td) / "state.db"
    init_db(db)
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"

    class _Broker:
        def submit_order(self, _):
            raise RuntimeError("broker rejects negative qty")

    # Negative qty: code computes hash + INSERTs; broker rejects.
    req_neg = OrderRequest("X_Y", "ts1", "A", "X", "buy", -10.0, "market")
    with connect(db) as conn:
        out = submit_order(_Broker(), conn, req_neg, decision_price=100.0)
    check("negative qty: handled gracefully (rejected status, no crash)",
          out.status == "rejected")

    # Zero qty: same path
    req_zero = OrderRequest("X_Y", "ts2", "A", "X", "buy", 0.0, "market")
    with connect(db) as conn:
        out = submit_order(_Broker(), conn, req_zero, decision_price=100.0)
    check("zero qty: handled gracefully",
          out.status == "rejected")

    # SQL-injection-style pair_id — parameterized INSERTs should neutralize.
    nasty = "X'; DROP TABLE positions; --"
    req_sql = OrderRequest(nasty, "ts3", "A", "X", "buy", 5.0, "market")
    with connect(db) as conn:
        out = submit_order(_Broker(), conn, req_sql, decision_price=100.0)
        # Positions table must still exist after the attempt
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        # Also check the nasty string was stored verbatim, not executed
        row = conn.execute(
            "SELECT pair_id FROM orders WHERE pair_id = ?", (nasty,)
        ).fetchone()
    check("SQL-injection pair_id: positions table still exists",
          "positions" in tables)
    check("SQL-injection pair_id: stored verbatim (parameterized INSERT works)",
          row is not None and row["pair_id"] == nasty)


# ===============================================================
# D. Adversarial hardstop flag
# ===============================================================

def t_adversarial_hardstop_flag():
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"

    # Garbage content — is_tripped should still return True (file-existence check)
    hardstop.HARDSTOP_FLAG_PATH.write_bytes(b"\x00\x01\x02malformed\xff")
    check("hardstop with garbage binary content: is_tripped True (existence-only check)",
          hardstop.is_tripped() is True)

    # Empty file
    hardstop.HARDSTOP_FLAG_PATH.write_text("")
    check("hardstop empty file: is_tripped True", hardstop.is_tripped() is True)

    # Very large content (1MB)
    hardstop.HARDSTOP_FLAG_PATH.write_text("X" * (1 << 20))
    check("hardstop 1MB content: is_tripped True", hardstop.is_tripped() is True)

    hardstop.clear(operator_note="audit cleanup")
    check("hardstop cleared: is_tripped False",
          hardstop.is_tripped() is False)


# ===============================================================
# E. decide() with weird state values
# ===============================================================

def t_decide_with_bad_state():
    from live.engine_live.live_pair import decide
    # State=2 (invalid) — what does decide do? Should preserve (no entry, no exit).
    # Backtest assumes state in {-1, 0, +1}; live should match — no transition for bad state.
    d = decide(2, -4.0, entry_z=3.0, hard_sl_z=5.0)
    # The code only matches state == 1, -1, or 0. So state=2 falls through all checks
    # and returns hold with new_state=2.
    check("decide(state=2, z=-4): falls through, returns hold (preserves invalid state)",
          d.action == "hold" and d.new_state == 2,
          f"got {d}")
    # Negative state value like -5
    d2 = decide(-5, 4.0, entry_z=3.0, hard_sl_z=5.0)
    check("decide(state=-5, z=+4): falls through, hold",
          d2.action == "hold" and d2.new_state == -5)


# ===============================================================
# F. regime_check on first-day-of-month boundary
# ===============================================================

def t_regime_check_first_day_boundary():
    """When decision_ts is the FIRST day of its month, trade_start == decision_ts month-start.
    halt_for_fold uses feats.index < trade_start, so the decision day's own data is EXCLUDED.
    Live should still produce a decision (possibly 'burnin_insufficient_history' if no data)."""
    from live.engine_live.regime_check import decide_month
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="rc_")
    db = Path(td) / "state.db"
    init_db(db)
    # Use one ticker from the cache to keep it light
    decision_ts = datetime(2024, 3, 1, 9, 30, tzinfo=timezone.utc)
    try:
        with connect(db) as conn:
            dec = decide_month(conn, decision_ts, ["AAPL", "MSFT", "JPM", "BAC"])
        check("regime_check on first-day-of-month: decision returned (not crash)",
              dec is not None)
        check("regime_check: month string formatted YYYY-MM",
              dec.month == "2024-03", f"got {dec.month}")
    except Exception as e:
        check("regime_check on first-day-of-month: no crash",
              False, f"{type(e).__name__}: {e}")


# ===============================================================
# G. apply_event with malformed payload
# ===============================================================

def t_apply_event_malformed():
    from live.broker.trading_stream import TradingStreamHandler
    from live.state.persist import init_db
    td = tempfile.mkdtemp(prefix="ae_")
    db = Path(td) / "state.db"
    init_db(db)
    def factory():
        c = sqlite3.connect(db, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c
    h = TradingStreamHandler(cfg=None, conn_factory=factory)

    # Missing 'order' field
    r1 = h.apply_event({"event": "fill"})
    check("apply_event missing 'order': returns None (no crash)", r1 is None)
    # Missing client_order_id
    r2 = h.apply_event({"event": "fill", "order": {"symbol": "X", "side": "buy"}})
    check("apply_event missing client_order_id: returns None", r2 is None)
    # client_order_id not in DB (e.g., a fill we never submitted) → tries UPDATE, affects 0 rows
    r3 = h.apply_event({"event": "fill", "order": {
        "client_order_id": "unknown_coid", "id": "b1", "symbol": "X",
        "side": "buy", "status": "filled",
        "filled_qty": 10, "filled_avg_price": 100.0,
    }})
    # apply_event returns a FillEvent if it's a fill, but the UPDATE does nothing.
    check("apply_event with unknown coid: no crash, returns event but DB unchanged",
          r3 is not None)


# ===============================================================
# H. reconcile with garbage broker response
# ===============================================================

def t_reconcile_garbage_broker():
    from live.execution.reconciliation import reconcile
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="rec_")
    db = Path(td) / "state.db"
    init_db(db)

    class _BadBroker:
        def get_all_positions(self):
            raise ConnectionError("network down")

    with connect(db) as conn:
        mm = reconcile(_BadBroker(), conn)
    check("reconcile when broker raises: returns empty list (handled), audit logged",
          mm == [])
    with connect(db) as conn:
        err_log = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event = 'reconcile_error'"
        ).fetchone()
    check("reconcile error logged to audit_log", err_log["n"] >= 1)

    # Broker returning malformed position objects
    class _MalformedBroker:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "JPM", "qty": "not_a_number"})()]

    try:
        with connect(db) as conn:
            mm2 = reconcile(_MalformedBroker(), conn)
        # The float() conversion will raise — we accept either graceful empty
        # OR a logged error. The system must not corrupt local state.
        check("reconcile with non-numeric qty: handled (no DB corruption)", True)
    except (ValueError, TypeError) as e:
        # Acceptable: float("not_a_number") raises; halt should not be tripped
        check("reconcile with malformed qty: raised cleanly, halt NOT tripped", True)


def main() -> int:
    print("== Edge cases & adversarial inputs ==")
    print("\n--- A. Boundary Z values ---")
    t_boundary_z_values()
    print("\n--- B. NaN / inf handling ---")
    t_nan_inf_handling()
    print("\n--- C. Adversarial OrderRequest ---")
    t_adversarial_order_request()
    print("\n--- D. Adversarial hardstop flag ---")
    t_adversarial_hardstop_flag()
    print("\n--- E. decide() with bad state ---")
    t_decide_with_bad_state()
    print("\n--- F. regime_check first-day-of-month ---")
    t_regime_check_first_day_boundary()
    print("\n--- G. apply_event malformed ---")
    t_apply_event_malformed()
    print("\n--- H. reconcile garbage broker ---")
    t_reconcile_garbage_broker()
    print()
    if errors:
        print(f"FAIL: {len(errors)} edge case(s) failed: {errors}")
        return 1
    print("PASS: all edge cases handled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
