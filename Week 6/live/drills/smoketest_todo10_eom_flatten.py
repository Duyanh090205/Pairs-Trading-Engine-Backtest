"""TODO 10 smoketest: EOM flatten end-of-run cleanup."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


class _Broker:
    def __init__(self):
        self.submitted = []
        self.i = 0
    def submit_order(self, req):
        self.i += 1
        self.submitted.append(req)
        return type("R", (), {"id": f"b_{self.i}", "status": "accepted"})()


def _seed_open_position(db: Path, pair_id="JPM_BAC", direction=1):
    """Insert position + entry fills."""
    from live.state.persist import connect, init_db
    init_db(db)
    ts = datetime.now(timezone.utc).isoformat()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO positions (pair_id, side_a, side_b, beta, direction, "
            "notional_a, notional_b, entry_ts, entry_z) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, "JPM", "BAC", 1.2, direction, 9500.0, 11400.0, ts, 3.15),
        )
        # Entry fills
        for coid, ticker, side, qty, price in [
            ("e_a", "JPM", "buy" if direction == 1 else "sell", 50.0, 190.0),
            ("e_b", "BAC", "sell" if direction == 1 else "buy", 100.0, 114.0),
        ]:
            conn.execute(
                "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
                "order_type, status, fill_qty, fill_price, decision_price, submitted_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (coid, pair_id, "2026-06-01T20:00:00Z", ticker, side, qty,
                 "market", "filled", qty, price, price, ts),
            )


def t_flatten_no_open_positions_noop():
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.state.persist import connect, init_db
    td = tempfile.mkdtemp(prefix="eom_t10a_")
    db = Path(td) / "state.db"
    init_db(db)
    with connect(db) as conn:
        out = flatten_all_open_positions(_Broker(), conn, "2026-06-30T20:00:00Z")
    check("empty positions -> no flatten actions", out == [], f"got {out}")


def t_flatten_long_pair_submits_correct_legs():
    """direction=+1 (long A, short B) -> close = SELL A, BUY B."""
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="eom_t10b_")
    db = Path(td) / "state.db"
    _seed_open_position(db, direction=1)
    broker = _Broker()
    with connect(db) as conn:
        out = flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z")
    check("1 flatten action returned", len(out) == 1, f"got {len(out)}")
    check("2 broker orders submitted (both legs)",
          len(broker.submitted) == 2, f"got {len(broker.submitted)}")
    sides = {r.symbol: r.side.name for r in broker.submitted
             if hasattr(r, "symbol") and hasattr(r, "side")}
    check("close JPM is SELL (was long)", sides.get("JPM") == "SELL", str(sides))
    check("close BAC is BUY (was short)", sides.get("BAC") == "BUY", str(sides))


def t_flatten_short_pair_submits_correct_legs():
    """direction=-1 (short A, long B) -> close = BUY A, SELL B."""
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="eom_t10c_")
    db = Path(td) / "state.db"
    _seed_open_position(db, direction=-1)
    broker = _Broker()
    with connect(db) as conn:
        out = flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z")
    sides = {r.symbol: r.side.name for r in broker.submitted}
    check("close JPM is BUY (was short)", sides.get("JPM") == "BUY", str(sides))
    check("close BAC is SELL (was long)", sides.get("BAC") == "SELL", str(sides))


def t_flatten_bypasses_hardstop():
    """Hardstop tripped should NOT block flatten — closing positions during emergency."""
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.safety import hardstop
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="eom_t10d_")
    db = Path(td) / "state.db"
    _seed_open_position(db, direction=1)
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    hardstop.HARDSTOP_FLAG_PATH.write_text("manual halt\n")
    assert hardstop.is_tripped()
    broker = _Broker()
    with connect(db) as conn:
        out = flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z",
                                         bypass_hardstop=True)
    check("flatten proceeds despite hardstop trip (bypass=True)",
          len(broker.submitted) == 2, f"got {len(broker.submitted)} orders")
    # Cleanup
    hardstop.clear("t10 cleanup")


def t_flatten_idempotent_on_double_call():
    """Calling flatten twice should NOT submit duplicate orders (coid dedupe)."""
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="eom_t10e_")
    db = Path(td) / "state.db"
    _seed_open_position(db, direction=1)
    broker = _Broker()
    with connect(db) as conn:
        out1 = flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z")
        out2 = flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z")
    # Second call should dedupe (returns OrderResult with submitted=False)
    n_actually_submitted = sum(1 for a in out2
                                if a.leg_a_order.submitted or a.leg_b_order.submitted)
    check("2nd flatten call dedupes (no new submissions)",
          n_actually_submitted == 0, f"got {n_actually_submitted}")
    check("broker received only 2 orders (from 1st call)",
          len(broker.submitted) == 2, f"got {len(broker.submitted)}")


def t_flatten_reconstructs_qty_from_fills():
    """Closing qty must match the entry qty (reconstructed from fill history)."""
    from live.engine_live.eom_flatten import flatten_all_open_positions
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="eom_t10f_")
    db = Path(td) / "state.db"
    _seed_open_position(db, direction=1)   # entry qty: JPM=50, BAC=100
    broker = _Broker()
    with connect(db) as conn:
        flatten_all_open_positions(broker, conn, "2026-06-30T20:00:00Z")
    qty_by_ticker = {r.symbol: float(r.qty) for r in broker.submitted}
    check("close JPM qty == entry qty 50", qty_by_ticker.get("JPM") == 50.0,
          str(qty_by_ticker))
    check("close BAC qty == entry qty 100", qty_by_ticker.get("BAC") == 100.0,
          str(qty_by_ticker))


def t_hardstop_still_works():
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t10_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo10")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 10 Smoketest: EOM flatten ==\n")
    print("--- No-op when no positions ---")
    t_flatten_no_open_positions_noop()
    print("\n--- Long pair flatten ---")
    t_flatten_long_pair_submits_correct_legs()
    print("\n--- Short pair flatten ---")
    t_flatten_short_pair_submits_correct_legs()
    print("\n--- Bypass hardstop ---")
    t_flatten_bypasses_hardstop()
    print("\n--- Idempotency on double call ---")
    t_flatten_idempotent_on_double_call()
    print("\n--- Qty reconstruction from fill history ---")
    t_flatten_reconstructs_qty_from_fills()
    print("\n--- Hardstop ---")
    t_hardstop_still_works()
    print()
    if errors:
        print(f"FAIL: {len(errors)} - {errors}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
