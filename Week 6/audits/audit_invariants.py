"""Property-based invariant tests. For random inputs, certain properties MUST
hold regardless of the specific data. If any property is violated, it's a bug.

Properties checked:
  A. BarBuilder OHLC invariants over 1000 random tick streams:
       low <= open <= high, low <= close <= high, volume == sum(tick.size),
       n_ticks == count of ticks in minute, all values finite & positive
  B. ZTracker monotonicity property: at large positive spread, Z should be
       positive (relative to recent mean); at large negative, negative.
  C. Order state machine invariants: status transitions are append-only valid
       (submitted/accepted -> filled/partial/rejected/canceled; never backwards)
  D. Reconcile transitivity: if local == broker, mismatch=[]; if local diverges
       by exactly the tolerance, no halt; just past tolerance, halt.
  E. Hardstop monotonicity: once tripped, repeated check() doesn't un-trip.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


# ===============================================================
# A. BarBuilder OHLC invariants (1000 random streams)
# ===============================================================

def t_bar_builder_invariants():
    from live.engine_live.bar_builder import BarBuilder

    n_streams = 1000
    violations = {
        "low_above_open": 0, "low_above_close": 0,
        "high_below_open": 0, "high_below_close": 0,
        "volume_mismatch": 0, "n_ticks_mismatch": 0,
        "non_finite": 0, "non_positive_price": 0,
    }

    base = datetime(2026, 5, 25, 14, 30, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    for stream_i in range(n_streams):
        n_ticks = int(rng.integers(2, 50))
        prices = (100 + rng.normal(0, 0.5, n_ticks)).clip(0.01, 1000)
        sizes = rng.integers(1, 1000, n_ticks).astype(float)
        # Random tick times within a 30-second window (single minute)
        offsets_s = rng.uniform(0, 30, n_ticks)
        offsets_s.sort()

        bb = BarBuilder()
        for i in range(n_ticks):
            ts = base + timedelta(seconds=float(offsets_s[i]))
            bb.on_tick("X", ts, float(prices[i]), float(sizes[i]))
        bar = bb.in_progress("X")
        if bar is None:
            continue

        # Invariants
        if bar.low > bar.open + 1e-9:
            violations["low_above_open"] += 1
        if bar.low > bar.close + 1e-9:
            violations["low_above_close"] += 1
        if bar.high < bar.open - 1e-9:
            violations["high_below_open"] += 1
        if bar.high < bar.close - 1e-9:
            violations["high_below_close"] += 1
        if abs(bar.volume - sizes.sum()) > 1e-6:
            violations["volume_mismatch"] += 1
        if bar.n_ticks != n_ticks:
            violations["n_ticks_mismatch"] += 1
        for v in (bar.open, bar.high, bar.low, bar.close, bar.volume):
            if not math.isfinite(v):
                violations["non_finite"] += 1
                break
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            violations["non_positive_price"] += 1

    total_viols = sum(violations.values())
    print(f"  ran {n_streams} random tick streams")
    for k, v in violations.items():
        if v > 0:
            print(f"    VIOLATION {k}: {v}")
    check(f"BarBuilder OHLC invariants over {n_streams} random streams",
          total_viols == 0, f"total violations: {total_viols}")


# ===============================================================
# B. ZTracker monotonicity
# ===============================================================

def t_ztracker_monotonicity():
    from live.engine_live.z_tracker import ZTracker
    seed = [0.0] * 60
    z = ZTracker(window=60, seed=seed)
    out_high = z.push(10.0)   # 10 std-devs above mean (mean=0, std=tiny)
    z2 = ZTracker(window=60, seed=seed)
    out_low = z2.push(-10.0)
    check("ZTracker.push(+10) returns large positive Z",
          out_high is not None and out_high > 0,
          f"got {out_high}")
    check("ZTracker.push(-10) returns large negative Z",
          out_low is not None and out_low < 0,
          f"got {out_low}")
    check("ZTracker symmetry: Z(+10) == -Z(-10) for symmetric seed",
          out_high is not None and out_low is not None
          and abs(out_high + out_low) < 1e-9)


# ===============================================================
# C. Order state monotonicity (status transitions)
# ===============================================================

def t_order_state_monotonicity():
    """Once an order is 'filled', subsequent applies should not un-fill it."""
    from live.broker.trading_stream import TradingStreamHandler
    from live.execution.order_manager import OrderRequest, submit_order
    from live.safety import hardstop
    from live.state.persist import connect, init_db
    import sqlite3

    td = tempfile.mkdtemp(prefix="osm_")
    db = Path(td) / "state.db"
    init_db(db)
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "no_flag.flag"

    class _B:
        def submit_order(self, _):
            return type("R", (), {"id": "b1", "status": "accepted"})()

    req = OrderRequest("X_Y", "ts1", "A", "X", "buy", 10.0, "market")
    with connect(db) as conn:
        out = submit_order(_B(), conn, req, decision_price=100.0)
    coid = out.client_order_id

    def factory():
        c = sqlite3.connect(db, isolation_level=None); c.row_factory = sqlite3.Row; return c
    h = TradingStreamHandler(cfg=None, conn_factory=factory)
    # Apply fill
    h.apply_event({"event": "fill", "order": {
        "client_order_id": coid, "id": "b1", "symbol": "X",
        "side": "buy", "status": "filled", "filled_qty": 10.0,
        "filled_avg_price": 100.05,
    }})
    # Then apply a "new" event (older state) for same order
    h.apply_event({"event": "new", "order": {
        "client_order_id": coid, "id": "b1", "symbol": "X",
        "side": "buy", "status": "new", "filled_qty": 0, "filled_avg_price": 0,
    }})
    # The current implementation UPDATEs status to 'new' here (no protection
    # against going backwards). This is a real bug — flag it.
    with connect(db) as conn:
        row = conn.execute(
            "SELECT status, fill_qty FROM orders WHERE client_order_id = ?", (coid,),
        ).fetchone()
    check("FLAG: order status protected against backwards transitions",
          row["status"].lower() == "filled" and float(row["fill_qty"]) == 10.0,
          f"status={row['status']} fill_qty={row['fill_qty']}")


# ===============================================================
# D. Reconcile tolerance boundary
# ===============================================================

def t_reconcile_tolerance_boundary():
    from live.execution.reconciliation import reconcile
    from live.state.persist import connect, init_db
    from datetime import datetime, timezone
    td = tempfile.mkdtemp(prefix="rtb_")
    db = Path(td) / "state.db"
    init_db(db)

    # Insert position + filled order so local_qty = 100 JPM
    ts = datetime.now(timezone.utc).isoformat()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
            "notional_a, notional_b, entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("JPM_BAC", "JPM", "BAC", 1.0, 1, 19000, 3300, ts, 3.1),
        )
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, submitted_ts, fill_qty, fill_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("c1", "JPM_BAC", ts, "JPM", "buy", 100, "market", "filled", ts, 100, 190.0),
        )
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, submitted_ts, fill_qty, fill_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("c2", "JPM_BAC", ts, "BAC", "sell", 100, "market", "filled", ts, 100, 33.0),
        )

    # Case 1: broker matches exactly -> no mismatch
    class _BrokerExact:
        def get_all_positions(self):
            return [
                type("P", (), {"symbol": "JPM", "qty": "100"})(),
                type("P", (), {"symbol": "BAC", "qty": "-100"})(),
            ]
    with connect(db) as conn:
        mm = reconcile(_BrokerExact(), conn, tolerance_shares=1.0)
    check("reconcile exact match: empty", mm == [], f"{mm}")

    # Case 2: broker off by exactly tolerance (1 share) -> still no mismatch (>tolerance, not >=)
    class _BrokerAtTol:
        def get_all_positions(self):
            return [
                type("P", (), {"symbol": "JPM", "qty": "101"})(),
                type("P", (), {"symbol": "BAC", "qty": "-100"})(),
            ]
    with connect(db) as conn:
        mm2 = reconcile(_BrokerAtTol(), conn, tolerance_shares=1.0)
    check("reconcile at-tolerance (diff=1, tol=1): NO mismatch (strict >)",
          mm2 == [], f"{mm2}")

    # Case 3: broker just past tolerance (1.5 shares) -> mismatch detected
    class _BrokerPastTol:
        def get_all_positions(self):
            return [
                type("P", (), {"symbol": "JPM", "qty": "102"})(),
                type("P", (), {"symbol": "BAC", "qty": "-100"})(),
            ]
    with connect(db) as conn:
        mm3 = reconcile(_BrokerPastTol(), conn, tolerance_shares=1.0)
    check("reconcile past-tolerance (diff=2, tol=1): mismatch detected",
          len(mm3) == 1, f"{mm3}")


# ===============================================================
# E. Hardstop monotonicity
# ===============================================================

def t_hardstop_monotonicity():
    from live.safety import hardstop
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="hm_")
    db = Path(td) / "state.db"
    init_db(db)
    flag = Path(td) / "HARDSTOP.flag"
    hardstop.HARDSTOP_FLAG_PATH = flag

    with connect(db) as conn:
        # Trip via large equity drop
        st1 = hardstop.check(conn, session_start_equity=100_000.0,
                             current_equity=90_000.0)
        assert st1.tripped
        # Now even with healthy equity, is_tripped should still be True
        check("hardstop persists: file-based check ignores equity recovery",
              hardstop.is_tripped() is True)
        # check() again on healthy equity -- should NOT un-trip (only manual clear)
        st2 = hardstop.check(conn, session_start_equity=100_000.0,
                             current_equity=100_000.0)
        check("hardstop check() with healthy equity: still tripped (flag file exists)",
              st2.tripped is True)
        # Manual clear
        hardstop.clear("audit_invariants")
        check("hardstop cleared: is_tripped False",
              hardstop.is_tripped() is False)


def main() -> int:
    print("== Invariants / property tests ==")
    print("\n--- A. BarBuilder OHLC invariants ---")
    t_bar_builder_invariants()
    print("\n--- B. ZTracker monotonicity ---")
    t_ztracker_monotonicity()
    print("\n--- C. Order state monotonicity ---")
    t_order_state_monotonicity()
    print("\n--- D. Reconcile tolerance boundary ---")
    t_reconcile_tolerance_boundary()
    print("\n--- E. Hardstop monotonicity ---")
    t_hardstop_monotonicity()
    print()
    if errors:
        print(f"FAIL: {len(errors)} invariant(s) violated: {errors}")
        return 1
    print("PASS: all invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
