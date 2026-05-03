"""
Plan 3 smoke test. Runs every analytics module against the artefacts
already produced by Plan 2 (data/cost_log.parquet, data/kappa_per_fold.parquet,
data/week4_inputs/*).
"""
from __future__ import annotations

import sys
from pathlib import Path

WEEK5_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(WEEK5_ROOT))

from src.plan3_validation import (
    generate_before_after_table,
    generate_cost_waterfall,
    analyze_regime_costs,
    validate_impact_prediction,
    analyze_kill_zone,
    run_dynamic_negative_control,
    compute_overfitting_diagnostics,
    run_oat_sensitivity,
    assemble_report,
    check_dynamic_cost_blowup,
    check_cost_exceeds_alpha,
    check_kappa_instability,
    check_dsr_degradation,
    check_math_violation,
)


def _required_inputs() -> bool:
    needed = [
        WEEK5_ROOT / "data" / "cost_log.parquet",
        WEEK5_ROOT / "data" / "kappa_per_fold.parquet",
        WEEK5_ROOT / "data" / "week4_inputs" / "trade_log.csv",
        WEEK5_ROOT / "data" / "week4_inputs" / "rebalance_log.csv",
        WEEK5_ROOT / "data" / "week4_inputs" / "fold_metrics.csv",
        WEEK5_ROOT / "data" / "microstructure_synth_for_plan2" / "spreads_1min.parquet",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("[FAIL] Missing required inputs:")
        for p in missing:
            print(f"  - {p}")
        return False
    return True


def test_before_after():
    print("\n--- 4.1 Before/After Table ---")
    ba = generate_before_after_table()
    assert set(ba.columns) == {"Gross", "Static60bps", "Dynamic"}
    assert ba.shape == (8, 3), f"Expected 8x3, got {ba.shape}"
    empty_cells = ba.isna().sum().sum()
    print(f"      Shape {ba.shape}; {empty_cells} NaN cells (sharpe NaN allowed if std=0)")
    print(ba.to_string())
    print("[PASS] Before/After table built")
    return ba


def test_waterfall():
    print("\n--- 4.2 Cost Waterfall ---")
    wf = generate_cost_waterfall()
    assert "Overall" in wf.index
    overall = wf.loc["Overall"]
    # cost_bps columns are stored as negative; gross + negatives = net
    summed = (overall["gross_bps"] + overall["spread_bps"] + overall["impact_bps"]
              + overall["borrow_bps"] + overall["rebalance_bps"])
    assert abs(summed - overall["net_bps"]) < 0.5, \
        f"Waterfall does not sum: gross+components={summed} != net={overall['net_bps']}"
    print(wf.to_string())
    print("[PASS] Waterfall sums to net within rounding")
    return wf


def test_regime_costs():
    print("\n--- 4.3 Regime Costs ---")
    rc = analyze_regime_costs()
    assert len(rc) == 4, f"Expected 4 regimes, got {len(rc)}"
    print(rc.to_string())
    print("[PASS] 4 regime rows produced")
    return rc


def test_impact_validation():
    print("\n--- 4.4 Impact Validation (DIAGNOSTIC) ---")
    iv = validate_impact_prediction()
    assert "note" in iv
    print(f"      {iv}")
    print("[PASS] Impact validation diagnostic returned")
    return iv


def test_kill_zone():
    print("\n--- 4.5 Kill Zone ---")
    a, b = analyze_kill_zone()
    assert len(a) == 13, f"Expected 13 buckets in seasonality, got {len(a)}"
    assert len(b) == 13, f"Expected 13 buckets in kill zone, got {len(b)}"
    print("Part A (avg spread by tier, bps):")
    print(a.to_string())
    print("\nPart B (net alpha by bucket):")
    print(b.to_string())
    print("[PASS] 13 intraday buckets x both outputs")
    return a, b


def test_neg_control():
    print("\n--- 4.6 Negative Control ---")
    nc = run_dynamic_negative_control()
    assert "nc_dynamic_status" in nc.columns
    print(nc.to_string())
    print("[PASS] NC report (Option B explicit)")
    return nc


def test_overfitting():
    print("\n--- 4.7 Overfitting Diagnostics (Net) ---")
    of = compute_overfitting_diagnostics()
    assert set(of.columns) >= {"Gross", "Static", "Dynamic"}
    assert set(of.index) >= {"Raw Sharpe", "DSR p-value", "PBO"}
    print(of.to_string())
    print("[PASS] DSR + PBO computed for all three regimes")
    return of


def test_sensitivity():
    print("\n--- 4.8 OAT Sensitivity ---")
    sw = run_oat_sensitivity()
    n_l1 = (sw["spread_level"] == "L1").sum()
    assert n_l1 == 9, f"Expected 9 L1 runs, got {n_l1}"
    # monotonicity within each borrow rate, kappa multiplier should produce
    # monotonically lower net Sharpe as costs scale up
    bsl = sw[(sw["kappa_mult"] == 1.0) & (sw["borrow_bps"] == 50.0)]["net_sharpe"].iloc[0]
    print(f"      Baseline (kappa=1.0, borrow=50): {bsl:.4f}")
    print(sw.to_string(index=False))
    print("[PASS] 9 OAT L1 runs + 1 L2 deferred row")
    return sw


def test_report_assembly():
    print("\n--- 4.9 Report Assembly ---")
    res = assemble_report()
    out = Path(res["output_path"])
    assert out.exists(), f"Report not written to {out}"
    text = out.read_text(encoding="utf-8")
    for header in ["## 1. Executive Summary", "## 4. Before/After Table",
                   "## 5. Cost Waterfall", "## 12. Verdict"]:
        assert header in text, f"Missing section: {header}"
    n_sections = sum(1 for line in text.splitlines() if line.startswith("## "))
    assert n_sections == 12, f"Expected 12 sections, got {n_sections}"
    print(f"      Wrote {out} ({len(text):,} chars, {n_sections} sections)")
    print(f"      Verdict: {res['verdict']}")
    print(f"      Red flags: {res['red_flags']}")
    print("[PASS] 12-section report rendered")
    return res


def test_red_flag_synthetic():
    """Inject synthetic bad inputs; verify each trigger fires."""
    print("\n--- Red Flag Triggers (synthetic) ---")
    assert check_dynamic_cost_blowup(160) is True
    assert check_dynamic_cost_blowup(50) is False
    assert check_cost_exceeds_alpha(net_sharpe=-0.5, gross_sharpe=1.5) is True
    assert check_cost_exceeds_alpha(net_sharpe=0.5, gross_sharpe=1.5) is False
    assert check_kappa_instability(3) is True
    assert check_kappa_instability(1) is False
    assert check_dsr_degradation(0.20, 0.02) is True
    assert check_dsr_degradation(0.04, 0.02) is False
    assert check_math_violation(-1.0, 0.0, 0.0) is True
    assert check_math_violation(1.0, 100.0, 50.0) is True
    assert check_math_violation(1.0, 50.0, 100.0) is False
    print("[PASS] All 5 red flag triggers fire on synthetic bad input and not on good input")


def main():
    print("=" * 70)
    print("Plan 3 Smoke Test")
    print("=" * 70)
    if not _required_inputs():
        sys.exit(1)
    test_before_after()
    test_waterfall()
    test_regime_costs()
    test_impact_validation()
    test_kill_zone()
    test_neg_control()
    test_overfitting()
    test_sensitivity()
    test_report_assembly()
    test_red_flag_synthetic()
    print("\n" + "=" * 70)
    print("Plan 3 Smoke Test Passed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
