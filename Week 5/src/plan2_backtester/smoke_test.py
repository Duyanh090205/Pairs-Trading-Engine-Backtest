"""
Plan 2 smoke test.

Strategy: load REAL Week 4 trade_log and rebalance_log, but generate SYNTHETIC
microstructure data covering only the tickers and timestamps present in those
trades. This proves the orchestrator works end-to-end with real Week 4 schemas
without requiring the 4.1 GB orderbook ingest.
"""
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

WEEK5_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WEEK5_ROOT))

from src.plan2_backtester import (
    FOLD_SCHEDULE,
    calibrate_fold,
    run_plan2,
    load_trade_log,
    load_rebalance_log,
)
from src.plan1_cost_model.interface_contract import validate_cost_log


WEEK4_INPUTS = WEEK5_ROOT / "data" / "week4_inputs"


# ----------------------------------------------------------------- #
# Synthetic microstructure generator covering real trade timestamps
# ----------------------------------------------------------------- #
def _build_synthetic_microstructure(trade_log: pd.DataFrame, rebalance_log: pd.DataFrame, out_dir: Path):
    """
    Generates spreads_1min.parquet and spread_rolling.parquet covering every
    (ticker, day) needed by the trade and rebalance logs, with full session
    coverage (390 bars/day) so the at-or-before lookup always succeeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = set()
    for r in trade_log.itertuples():
        for ticker, ts in [(r.ticker_A, r.entry_ts), (r.ticker_B, r.entry_ts),
                           (r.ticker_A, r.exit_ts),  (r.ticker_B, r.exit_ts)]:
            needed.add((ticker, ts.date()))
            # Also include all dates in between for borrow days
            d = ts.date()
            for delta in range(-1, 2):
                needed.add((ticker, d + pd.Timedelta(days=delta)))
    for r in rebalance_log.itertuples():
        needed.add((r.ticker, r.rebalance_ts.date()))

    rng = np.random.default_rng(42)
    rows = []
    for ticker, day in sorted(needed):
        # Fixed half-spread per ticker (deterministic from hash for reproducibility)
        seed_hs = abs(hash(ticker)) % 30 + 1   # 1–30 bps half-spread
        # Build full session bars for that day
        start = pd.Timestamp(day).tz_localize("US/Eastern") + pd.Timedelta(hours=9, minutes=30)
        idx = pd.date_range(start, periods=390, freq="1min")
        full_spread = seed_hs * 2 + rng.normal(0, 0.5, len(idx))
        full_spread = np.maximum(full_spread, 0.1)
        for ts, fs in zip(idx, full_spread):
            rows.append({
                "timestamp_et": ts,
                "ticker": ticker,
                "mid_px": 100.0,
                "full_spread_l1_bps": fs,
                "half_spread_l1_bps": fs / 2.0,
                "is_valid": True,
            })

    spreads = pd.DataFrame(rows)
    spreads.to_parquet(out_dir / "spreads_1min.parquet")

    # Rolling: simple constant sigma per ticker
    rolling_rows = []
    for ticker in spreads["ticker"].unique():
        seed_sigma = (abs(hash(ticker)) % 5) + 1.0   # 1–6 bps sigma
        sub = spreads[spreads["ticker"] == ticker]
        for ts in sub["timestamp_et"]:
            rolling_rows.append({
                "timestamp_et": ts,
                "ticker": ticker,
                "spread_std_1d": seed_sigma,
                "raw_spread_mean_1d": seed_sigma * 4,
            })
    pd.DataFrame(rolling_rows).to_parquet(out_dir / "spread_rolling.parquet")


# ----------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------- #
def test_fold_schedule():
    print("\n--- Fold Schedule ---")
    assert len(FOLD_SCHEDULE) == 45, f"Expected 45 folds, got {len(FOLD_SCHEDULE)}"
    f1 = FOLD_SCHEDULE[0]
    assert f1.fold_id == 1
    assert f1.formation_start.strftime("%Y-%m-%d") == "2022-01-03"
    assert f1.trading_start.strftime("%Y-%m") == "2022-07"
    f45 = FOLD_SCHEDULE[44]
    assert f45.fold_id == 45
    assert f45.trading_start.strftime("%Y-%m") == "2026-03"
    print(f"[PASS] FOLD_SCHEDULE: 45 folds, fold 1 trading {f1.trading_label}, fold 45 trading {f45.trading_label}")


def test_lookahead_guard():
    print("\n--- Lookahead Guard ---")
    f = FOLD_SCHEDULE[0]
    try:
        calibrate_fold(
            fold_id=1,
            formation_start=f.trading_start + pd.Timedelta(days=10),
            formation_end=f.trading_start + pd.Timedelta(days=20),
            trading_start=f.trading_start,
            tickers=["AAPL"],
            spreads_df=pd.DataFrame(columns=["timestamp_et", "ticker", "full_spread_l1_bps", "is_valid"]),
        )
    except ValueError as e:
        assert "Lookahead" in str(e)
        print("[PASS] calibrate_fold raises ValueError when formation_end >= trading_start")
        return
    raise AssertionError("calibrate_fold did not raise on lookahead")


def test_real_week4_schemas():
    print("\n--- Real Week 4 Schema Validation ---")
    tl = load_trade_log()
    rb = load_rebalance_log()
    print(f"      trade_log:     {len(tl)} trades, {tl['fold_id'].nunique()} folds, "
          f"{pd.concat([tl.ticker_A, tl.ticker_B]).nunique()} unique tickers")
    print(f"      rebalance_log: {len(rb)} events, {rb['fold_id'].nunique()} folds")
    assert tl["entry_ts"].dt.tz is not None, "entry_ts is not tz-aware"
    assert str(tl["entry_ts"].dt.tz) == "US/Eastern", f"entry_ts tz is {tl['entry_ts'].dt.tz}"
    assert rb["rebalance_ts"].dt.tz is not None, "rebalance_ts not tz-aware"
    print("[PASS] Real Week 4 trade_log + rebalance_log loaded; tz=US/Eastern; schemas valid")


def test_end_to_end_orchestrator():
    print("\n--- End-to-End Orchestrator ---")
    tl = load_trade_log()
    rb = load_rebalance_log()

    tmp = Path(tempfile.mkdtemp(prefix="plan2_smoke_"))
    micro_dir = tmp / "microstructure"
    out_dir = tmp / "out"

    print(f"      Building synthetic microstructure under {micro_dir} ...")
    _build_synthetic_microstructure(tl, rb, micro_dir)

    cost_log, kappa_audit = run_plan2(
        trade_log_path=WEEK4_INPUTS / "trade_log.csv",
        rebalance_log_path=WEEK4_INPUTS / "rebalance_log.csv",
        micro_dir=micro_dir,
        out_dir=out_dir,
    )

    # 1. cost_log row count matches trade count
    assert len(cost_log) == len(tl), f"cost_log has {len(cost_log)} rows; trade_log has {len(tl)}"
    print(f"[PASS] cost_log row count matches trade_log: {len(cost_log)}")

    # 2. Schema validation
    validate_cost_log(cost_log)
    for col in ["total_cost_static", "total_cost_gross", "gross_pnl_dollars", "net_pnl_static"]:
        assert col in cost_log.columns, f"missing comparison column {col}"
    print("[PASS] cost_log schema validated; three regimes present")

    # 3. total_cost_dollars >= 0 for 100% of trades
    assert (cost_log["total_cost_dollars"] >= 0).all(), \
        f"{(cost_log['total_cost_dollars'] < 0).sum()} trades have negative total cost"
    print(f"[PASS] total_cost_dollars >= 0 for all {len(cost_log)} trades")

    # 4. net_pnl_dollars <= gross_pnl_dollars for 100% of trades
    violations = (cost_log["net_pnl_dollars"] > cost_log["gross_pnl_dollars"]).sum()
    assert violations == 0, f"{violations} trades have net > gross"
    print("[PASS] net_pnl_dollars <= gross_pnl_dollars for all trades (dynamic regime)")
    violations_static = (cost_log["net_pnl_static"] > cost_log["gross_pnl_dollars"]).sum()
    assert violations_static == 0, f"{violations_static} trades have net_static > gross"
    print("[PASS] net_pnl_static <= gross_pnl_dollars for all trades (static regime)")

    # 5. Three regimes are distinct
    assert (cost_log["total_cost_gross"] == 0).all(), "Gross regime must be zero cost"
    assert (cost_log["total_cost_static"] > 0).all(), "Static regime must be positive"
    assert (cost_log["total_cost_dollars"] > 0).any(), "Dynamic should be positive on average"
    diff = (cost_log["total_cost_dollars"] - cost_log["total_cost_static"]).std()
    assert diff > 0, "Dynamic and static should differ across trades (variable spreads)"
    print(f"[PASS] Three regimes distinct: Gross=0, Static>0, Dynamic varies (std diff={diff:.2f})")

    # 6. Kappa audit non-empty, valid kappa values
    assert len(kappa_audit) > 0, "kappa_audit empty"
    valid_k = {0.3, 0.5, 0.8}
    bad = set(kappa_audit["kappa"].unique()) - valid_k
    assert not bad, f"Invalid kappa values: {bad}"
    print(f"[PASS] kappa_audit: {len(kappa_audit)} (fold,ticker) pairs; all kappa in {valid_k}")

    # 7. Rebalance cost line items match rebalance_log rows
    n_rebals_used = (cost_log["rebalance_cost_dollars"] > 0).sum()
    n_trades_with_rebals = rb["trade_id"].nunique()
    assert n_rebals_used == n_trades_with_rebals, \
        f"trades-with-rebals from cost_log: {n_rebals_used}; from rebalance_log: {n_trades_with_rebals}"
    print(f"[PASS] Rebalance accounting: {n_rebals_used} trades carry rebalance cost (matches rebalance_log)")

    # 8. Cost components sum to total
    component_sum = (
        cost_log["spread_cost_dollars"]
        + cost_log["impact_cost_dollars"]
        + cost_log["borrow_cost_dollars"]
        + cost_log["rebalance_cost_dollars"]
    )
    assert np.allclose(component_sum, cost_log["total_cost_dollars"], rtol=1e-9), \
        "Components do not sum to total"
    print("[PASS] Cost components sum to total_cost_dollars")

    # 9. Borrow cost: zero for intraday trades only
    intraday = tl[tl["entry_ts"].dt.date == tl["exit_ts"].dt.date]
    if len(intraday) > 0:
        intraday_ids = set(intraday["trade_id"])
        intraday_borrow = cost_log[cost_log["trade_id"].isin(intraday_ids)]["borrow_cost_dollars"]
        assert (intraday_borrow == 0).all(), "Borrow charged on intraday trades"
        print(f"[PASS] Intraday trades (n={len(intraday)}) carry zero borrow cost")
    else:
        print("[INFO] No intraday trades in this trade_log")

    # Print summary statistics — join on trade_id to get allocated_capital
    merged = cost_log.merge(tl[["trade_id", "allocated_capital"]], on="trade_id")
    avg_cost_bps = (merged["total_cost_dollars"] / merged["allocated_capital"]).mean() * 10000
    print(f"\n      Mean dynamic RT cost: ${cost_log['total_cost_dollars'].mean():.2f} "
          f"(~{avg_cost_bps:.1f} bps of allocated capital)")
    print(f"      Mean static RT cost:  ${cost_log['total_cost_static'].mean():.2f}")
    print(f"      Mean rebalance cost:  ${cost_log['rebalance_cost_dollars'].mean():.2f}")

    # Cleanup
    shutil.rmtree(tmp)


def main():
    print("--- Running Plan 2 Smoke Test ---")
    test_fold_schedule()
    test_lookahead_guard()
    test_real_week4_schemas()
    test_end_to_end_orchestrator()
    print("\nPlan 2 Smoke Test Passed Successfully!")


if __name__ == "__main__":
    main()
