"""
End-to-end pipeline: Plan 0 → Plan 2 → Plan 3 → final report.

Usage:
    python run_pipeline.py           # full run; skips Plan 0 if already done
    python run_pipeline.py --force0  # re-run Plan 0 even if outputs exist

Plan 0 outputs are considered done when all 4 microstructure parquets exist in
data/microstructure/.  Set --force0 to regenerate them from the raw orderbook.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

WEEK5_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK5_ROOT))

MICRO_DIR   = WEEK5_ROOT / "data" / "microstructure"
MICRO_FILES = [
    MICRO_DIR / "spreads_1min.parquet",
    MICRO_DIR / "spread_rolling.parquet",
    MICRO_DIR / "spread_seasonality.parquet",
    MICRO_DIR / "spread_summary.parquet",
]


def _section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ── Plan 0 ──────────────────────────────────────────────────────────────────

def run_plan0(force: bool = False) -> None:
    missing = [f for f in MICRO_FILES if not f.exists()]
    if not force and not missing:
        _section("Plan 0 — SKIPPED (microstructure parquets already exist)")
        for f in MICRO_FILES:
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.0f} MB")
        return

    if force:
        _section("Plan 0 — Re-running (--force0 flag set)")
    else:
        _section(f"Plan 0 — Running (missing: {[f.name for f in missing]})")

    from src.plan0_gateway.run_real_data import run_plan0_real
    t0 = time.time()
    result = run_plan0_real()
    elapsed = time.time() - t0
    print(f"\n  Plan 0 done: {result['ok_count']} tickers, "
          f"{result['total_rows']:,} rows, {elapsed/60:.1f} min")
    if result["failed_count"]:
        print(f"  [WARN] {result['failed_count']} tickers failed: "
              f"{result['failed_tickers'][:5]}")


# ── Plan 2 ──────────────────────────────────────────────────────────────────

def run_plan2() -> tuple:
    _section("Plan 2 — Cost application (real microstructure)")
    from src.plan2_backtester.orchestrator import run_plan2 as _run2
    import pandas as pd

    t0 = time.time()
    cost_log, kappa = _run2()
    elapsed = time.time() - t0

    print(f"\n  Trades priced: {len(cost_log)}")
    print(f"  Kappa (fold,ticker) pairs: {len(kappa)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()

    # Quick sanity checks
    neg_cost = (cost_log["total_cost_dollars"] < 0).sum()
    net_gt_gross = (cost_log["net_pnl_dollars"] > cost_log["gross_pnl_dollars"]).sum()
    mean_dyn_bps = (cost_log["total_cost_dollars"] / 20_000).mean() * 10_000
    mean_static_bps = (cost_log["total_cost_static"] / 20_000).mean() * 10_000

    print(f"  Mean dynamic RT cost: {mean_dyn_bps:.1f} bps  "
          f"(was {mean_static_bps:.1f} bps static)")
    print(f"  Math violations — negative cost: {neg_cost}, net > gross: {net_gt_gross}")

    kappa_dist = kappa["kappa"].value_counts().sort_index()
    print(f"  Kappa distribution:")
    for k, n in kappa_dist.items():
        tier = {0.3: "Tier1 Tight", 0.5: "Tier2 Medium", 0.8: "Tier3 Wide"}.get(k, "?")
        print(f"    kappa={k} ({tier}): {n} ({n/len(kappa)*100:.0f}%)")

    return cost_log, kappa


# ── Plan 3 ──────────────────────────────────────────────────────────────────

def run_plan3() -> dict:
    _section("Plan 3 — Validation & report")
    from src.plan3_validation.report_builder import assemble_report
    import pandas as pd

    t0 = time.time()
    result = assemble_report()
    elapsed = time.time() - t0

    out_path = Path(result["output_path"])
    print(f"\n  Report written: {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


# ── Final consolidated summary ───────────────────────────────────────────────

def _print_final_summary(plan3_result: dict) -> None:
    import pandas as pd
    import numpy as np

    _section("FINAL RESULTS — Real Microstructure")

    ba  = plan3_result["before_after"]
    wf  = plan3_result["waterfall"]
    rc  = plan3_result["regime_costs"]
    ov  = plan3_result["overfitting"]
    rf  = plan3_result["red_flags"]
    kz  = plan3_result["kill_zone_part_b"]
    sens = plan3_result["sensitivity"]

    # ── Before / After ──────────────────────────────────────────────────
    print("\n── Before / After (Gross / Static 60bps / Dynamic) ──")
    print(ba.to_string())

    # ── Cost waterfall ──────────────────────────────────────────────────
    print("\n── Cost Waterfall (bps of allocated capital, per trade) ──")
    print(wf.to_string())

    # ── Regime costs ────────────────────────────────────────────────────
    print("\n── Regime-Conditional Costs ──")
    print(rc.to_string())

    # ── Overfitting ──────────────────────────────────────────────────────
    print("\n── Overfitting Diagnostics (net returns) ──")
    print(ov.to_string())

    # ── Kill zones ───────────────────────────────────────────────────────
    kz_bad = kz[kz["kill_zone"] == True] if "kill_zone" in kz.columns else kz[kz["net_bps"] < 0]
    print(f"\n── Kill Zones ({len(kz_bad)} buckets where net alpha < 0) ──")
    print(kz_bad.to_string())

    # ── OAT sensitivity range ────────────────────────────────────────────
    valid_sens = sens[sens["net_sharpe"].notna()]
    if not valid_sens.empty:
        print(f"\n── OAT Sensitivity ──")
        print(f"  Net Sharpe range: {valid_sens['net_sharpe'].min():.4f} — "
              f"{valid_sens['net_sharpe'].max():.4f}  "
              f"(spread {valid_sens['net_sharpe'].max() - valid_sens['net_sharpe'].min():.4f})")

    # ── Red flags ────────────────────────────────────────────────────────
    print("\n── Red Flags ──")
    for flag, val in rf.items():
        status = "FIRED" if val else "ok"
        print(f"  {flag}: {status}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print(f"\n── VERDICT ──")
    print(f"  {plan3_result['verdict']}")
    print()


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 end-to-end pipeline")
    parser.add_argument("--force0", action="store_true",
                        help="Re-run Plan 0 even if microstructure parquets exist")
    args = parser.parse_args()

    wall_start = time.time()

    run_plan0(force=args.force0)
    cost_log, _ = run_plan2()
    plan3_result = run_plan3()
    _print_final_summary(plan3_result)

    total = time.time() - wall_start
    _section(f"Pipeline complete — {total/60:.1f} min total")
    print(f"  Report: {plan3_result['output_path']}")
    print(f"  methodology_results.md: update manually after reviewing numbers above")


if __name__ == "__main__":
    main()
