"""TODO 9 smoketest: dual cost track (realized broker + predicted overlay)."""
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


def _seed_filled_pair(db: Path, pair_id: str,
                      ta: str, tb: str,
                      bar_entry: str, bar_exit: str,
                      dec_a_entry: float, fill_a_entry: float,
                      dec_a_exit: float, fill_a_exit: float,
                      dec_b_entry: float, fill_b_entry: float,
                      dec_b_exit: float, fill_b_exit: float,
                      qty_a: float, qty_b: float):
    """Insert 4 filled orders (entry+exit, both legs) for a single pair trade.

    Each order records the DECISION PRICE at the time it was submitted, which
    differs between entry and exit bars in real trading.
    """
    from live.state.persist import connect, init_db
    init_db(db)
    ts = datetime.now(timezone.utc).isoformat()
    with connect(db) as conn:
        # Entry leg A (buy)
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, fill_qty, fill_price, decision_price, submitted_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e_a", pair_id, bar_entry, ta, "buy", qty_a, "market", "filled",
             qty_a, fill_a_entry, dec_a_entry, ts),
        )
        # Entry leg B (sell)
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, fill_qty, fill_price, decision_price, submitted_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e_b", pair_id, bar_entry, tb, "sell", qty_b, "market", "filled",
             qty_b, fill_b_entry, dec_b_entry, ts),
        )
        # Exit leg A (sell to close)
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, fill_qty, fill_price, decision_price, submitted_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("x_a", pair_id, bar_exit, ta, "sell", qty_a, "market", "filled",
             qty_a, fill_a_exit, dec_a_exit, ts),
        )
        # Exit leg B (buy to cover)
        conn.execute(
            "INSERT INTO orders (client_order_id, pair_id, bar_ts, ticker, side, qty, "
            "order_type, status, fill_qty, fill_price, decision_price, submitted_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("x_b", pair_id, bar_exit, tb, "buy", qty_b, "market", "filled",
             qty_b, fill_b_exit, dec_b_exit, ts),
        )


def t_realized_slippage_basic():
    from live.monitor.cost_overlay import compute_realized_cost_usd
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="cost_t9a_")
    db = Path(td) / "state.db"
    # AAPL buy at decision $200, filled $200.10 → +$0.10 adverse slippage per share, 50 shares = $5
    # MSFT sell at decision $400, filled $399.90 → +$0.10 adverse slippage per share, 25 shares = $2.50
    # Exit AAPL sell at $202 decision, filled $201.90 → +$0.10 adverse * 50 = $5
    # Exit MSFT buy at $401 decision, filled $401.05 → +$0.05 adverse * 25 = $1.25
    # Total slippage = 5 + 2.50 + 5 + 1.25 = $13.75
    # All 4 fills are ADVERSE (paying spread, slippage):
    #   entry A buy: dec $200, fill $200.10 -> +$0.10 * 50 = +$5
    #   entry B sell: dec $400, fill $399.90 -> +$0.10 * 25 = +$2.50
    #   exit A sell: dec $202, fill $201.90 -> +$0.10 * 50 = +$5
    #   exit B buy: dec $401, fill $401.05 -> +$0.05 * 25 = +$1.25
    # Total adverse slippage = $13.75
    _seed_filled_pair(
        db, "AAPL_MSFT", "AAPL", "MSFT",
        bar_entry="2026-06-01T20:00:00Z", bar_exit="2026-06-08T20:00:00Z",
        dec_a_entry=200.0, fill_a_entry=200.10,
        dec_a_exit=202.0, fill_a_exit=201.90,
        dec_b_entry=400.0, fill_b_entry=399.90,
        dec_b_exit=401.0, fill_b_exit=401.05,
        qty_a=50.0, qty_b=25.0,
    )
    with connect(db) as conn:
        slip, traded = compute_realized_cost_usd(
            conn, "AAPL_MSFT",
            "2026-06-01T20:00:00Z", "2026-06-08T20:00:00Z",
        )
    check(f"realized slippage = $13.75 (sum of adverse fills)",
          abs(slip - 13.75) < 1e-9, f"got {slip}")
    # traded_usd = sum(qty * decision_price) across all 4 fills
    # = 50*200 + 25*400 + 50*200 + 25*400 = 30000
    # traded_usd = sum(qty * decision_price) across all 4 fills
    # = 50*200 + 25*400 + 50*202 + 25*401 = 10000 + 10000 + 10100 + 10025 = 40125
    check("traded notional = $40,125 (sum of qty*decision across 4 fills)",
          abs(traded - 40125) < 1e-9, f"got {traded}")


def t_predicted_cost_with_fallback():
    """When CostData not available, fallback path computes flat 30 bps + borrow."""
    from live.monitor.cost_overlay import compute_predicted_cost_usd
    # 50 shares * $200 = $10k notional per leg, 7-day hold (1 week)
    cost = compute_predicted_cost_usd(
        cost_data=None,   # force fallback
        ticker_a="AAPL", ticker_b="MSFT",
        entry_ts="2026-06-01T20:00:00Z",
        exit_ts="2026-06-08T20:00:00Z",
        notional_per_leg=10_000.0, side_a=1,
    )
    # Fallback: spread 30 bps + commission 2 bps per side, 2 legs per side
    # = $10k * 2 legs * (30 + 30 + 4) bps / 10000 = $10k * 2 * 0.0064 = $128
    # + borrow: 50 bps/yr * 7/365 * $10k = $9.59
    # ~ $137.59
    expected = 10_000 * 2 * (30 + 30 + 4) / 10_000 + 50/10_000/365 * 10_000 * 7
    check(f"fallback predicted cost ~ ${expected:.2f}",
          abs(cost - expected) < 1.0, f"got ${cost:.2f}, expected ${expected:.2f}")


def t_dual_cost_drift_signed():
    """If realized > predicted → drift positive (paper actually worse than model says, rare)."""
    from live.monitor.cost_overlay import compute_dual_cost
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="cost_t9b_")
    db = Path(td) / "state.db"
    # All 4 fills are ADVERSE (paying spread, slippage):
    #   entry A buy: dec $200, fill $200.10 -> +$0.10 * 50 = +$5
    #   entry B sell: dec $400, fill $399.90 -> +$0.10 * 25 = +$2.50
    #   exit A sell: dec $202, fill $201.90 -> +$0.10 * 50 = +$5
    #   exit B buy: dec $401, fill $401.05 -> +$0.05 * 25 = +$1.25
    # Total adverse slippage = $13.75
    _seed_filled_pair(
        db, "AAPL_MSFT", "AAPL", "MSFT",
        bar_entry="2026-06-01T20:00:00Z", bar_exit="2026-06-08T20:00:00Z",
        dec_a_entry=200.0, fill_a_entry=200.10,
        dec_a_exit=202.0, fill_a_exit=201.90,
        dec_b_entry=400.0, fill_b_entry=399.90,
        dec_b_exit=401.0, fill_b_exit=401.05,
        qty_a=50.0, qty_b=25.0,
    )
    with connect(db) as conn:
        breakdown = compute_dual_cost(
            conn, cost_data=None,
            pair_id="AAPL_MSFT", ticker_a="AAPL", ticker_b="MSFT",
            entry_ts="2026-06-01T20:00:00Z", exit_ts="2026-06-08T20:00:00Z",
            notional_per_leg=10_000.0, side_a=1,
            bar_ts_entry="2026-06-01T20:00:00Z", bar_ts_exit="2026-06-08T20:00:00Z",
        )
    check("realized < predicted (paper has $0 commission, model includes 2bps)",
          breakdown.realized_cost_bps < breakdown.predicted_cost_bps,
          f"realized={breakdown.realized_cost_bps:.2f}, predicted={breakdown.predicted_cost_bps:.2f}")
    check("drift_bps is signed (realized - predicted)",
          abs(breakdown.drift_bps - (breakdown.realized_cost_bps - breakdown.predicted_cost_bps)) < 1e-9)


def t_zero_slippage_clean_paper():
    """Paper fills at decision price → realized cost = 0. Predicted still nonzero from model."""
    from live.monitor.cost_overlay import compute_dual_cost
    from live.state.persist import connect
    td = tempfile.mkdtemp(prefix="cost_t9c_")
    db = Path(td) / "state.db"
    # All 4 fills EXACTLY at decision price -> slippage = 0
    _seed_filled_pair(
        db, "X_Y", "X", "Y",
        bar_entry="2026-06-01T20:00:00Z", bar_exit="2026-06-02T20:00:00Z",
        dec_a_entry=100.0, fill_a_entry=100.0,
        dec_a_exit=101.0, fill_a_exit=101.0,
        dec_b_entry=50.0, fill_b_entry=50.0,
        dec_b_exit=49.5, fill_b_exit=49.5,
        qty_a=100.0, qty_b=200.0,
    )
    with connect(db) as conn:
        # All fills at decision price → realized cost = 0
        breakdown = compute_dual_cost(
            conn, cost_data=None,
            pair_id="X_Y", ticker_a="X", ticker_b="Y",
            entry_ts="2026-06-01T20:00:00Z", exit_ts="2026-06-02T20:00:00Z",
            notional_per_leg=10_000.0, side_a=1,
            bar_ts_entry="2026-06-01T20:00:00Z", bar_ts_exit="2026-06-02T20:00:00Z",
        )
    check("realized cost = 0 when fills exactly at decision",
          abs(breakdown.realized_cost_bps) < 1e-9,
          f"got {breakdown.realized_cost_bps}")
    check("predicted cost > 0 even with perfect fills (model assumes spread+borrow)",
          breakdown.predicted_cost_bps > 0,
          f"got {breakdown.predicted_cost_bps}")


def t_hardstop_still_works():
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t9_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips", hardstop.is_tripped())
    hardstop.clear("todo9")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 9 Smoketest: dual cost track ==\n")
    print("--- Realized slippage from broker fills ---")
    t_realized_slippage_basic()
    print("\n--- Predicted cost (fallback path) ---")
    t_predicted_cost_with_fallback()
    print("\n--- Dual cost breakdown signed drift ---")
    t_dual_cost_drift_signed()
    print("\n--- Zero slippage (perfect paper fills) ---")
    t_zero_slippage_clean_paper()
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
