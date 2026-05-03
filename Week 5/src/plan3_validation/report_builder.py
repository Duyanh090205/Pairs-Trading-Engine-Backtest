"""
4.9 — Assemble the 12-section Net-of-Fees Performance Report.

Writes reports/net_of_fees_report.md.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.plan3_validation.sharpe_net import generate_before_after_table
from src.plan3_validation.cost_waterfall import generate_cost_waterfall
from src.plan3_validation.regime_costs import analyze_regime_costs
from src.plan3_validation.impact_validation import validate_impact_prediction
from src.plan3_validation.kill_zone import analyze_kill_zone
from src.plan3_validation.negative_control import run_dynamic_negative_control
from src.plan3_validation.overfitting_net import compute_overfitting_diagnostics
from src.plan3_validation.sensitivity_oat import run_oat_sensitivity
from src.plan3_validation.red_flags import (
    check_dynamic_cost_blowup,
    check_cost_exceeds_alpha,
    check_kappa_instability,
    check_dsr_degradation,
)


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = WEEK5_ROOT / "reports" / "net_of_fees_report.md"


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown()


def assemble_report(output_path: Path | str = DEFAULT_OUTPUT) -> dict:
    """
    Builds the 12-section report. Returns a dict of the section outputs
    so the caller can inspect numerically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cost_log = pd.read_parquet(WEEK5_ROOT / "data" / "cost_log.parquet")
    kappa = pd.read_parquet(WEEK5_ROOT / "data" / "kappa_per_fold.parquet")

    sec4 = generate_before_after_table()
    sec5 = generate_cost_waterfall()
    sec6 = analyze_regime_costs()
    sec7 = validate_impact_prediction()
    sec8a, sec8b = analyze_kill_zone()
    sec9 = run_dynamic_negative_control()
    sec10 = compute_overfitting_diagnostics()
    sec11 = run_oat_sensitivity()

    # ----- Red flag evaluation -----
    mean_rt_bps = (cost_log["total_cost_dollars"] / 20_000).mean() * 10_000
    flag_blowup = check_dynamic_cost_blowup(mean_rt_bps)
    # Per-row check: any negative cost OR any net > gross
    neg_costs = (cost_log["total_cost_dollars"] < 0).any()
    net_gt_gross = (cost_log["net_pnl_dollars"] > cost_log["gross_pnl_dollars"]).any()
    flag_math = bool(neg_costs or net_gt_gross)
    ticker_kappa_changes = (kappa.groupby("ticker")["kappa"].nunique() > 1).sum()
    flag_kappa = check_kappa_instability(int(ticker_kappa_changes))

    sharpe_gross = float(sec4.loc["Sharpe (annual)", "Gross"])
    sharpe_dyn = float(sec4.loc["Sharpe (annual)", "Dynamic"])
    flag_cost_exceeds = check_cost_exceeds_alpha(sharpe_dyn, sharpe_gross)

    dsr_gross = float(sec10.loc["DSR p-value", "Gross"]) if "Gross" in sec10.columns else float("nan")
    dsr_dyn = float(sec10.loc["DSR p-value", "Dynamic"]) if "Dynamic" in sec10.columns else float("nan")
    flag_dsr = check_dsr_degradation(1 - dsr_dyn if not pd.isna(dsr_dyn) else 1.0,
                                      1 - dsr_gross if not pd.isna(dsr_gross) else 1.0)

    # ----- Verdict -----
    if flag_cost_exceeds:
        verdict = "Strategy alpha is real but untradeable: net Sharpe < 0 under dynamic costs."
    elif sharpe_dyn > 0:
        verdict = f"Strategy survives friction. Dynamic net Sharpe = {sharpe_dyn:.3f}."
    elif sharpe_dyn <= 0 and sharpe_gross <= 0:
        verdict = "Strategy underperforms even before costs; friction is not the limiting factor."
    else:
        verdict = f"Strategy degraded by friction. Dynamic Sharpe = {sharpe_dyn:.3f}."

    # ----- Compose markdown -----
    md_lines: list[str] = []
    md_lines.append("# Net-of-Fees Performance Report — Week 5")
    md_lines.append("")
    md_lines.append("## 1. Executive Summary")
    md_lines.append("")
    md_lines.append(f"**Verdict:** {verdict}")
    md_lines.append("")
    md_lines.append(f"- Trades evaluated: {len(cost_log)}")
    md_lines.append(f"- Mean dynamic RT cost: ${cost_log['total_cost_dollars'].mean():.2f} "
                    f"(~{mean_rt_bps:.1f} bps of allocated capital)")
    md_lines.append(f"- Sharpe (Gross): {sharpe_gross:.3f}")
    md_lines.append(f"- Sharpe (Static60bps): {float(sec4.loc['Sharpe (annual)', 'Static60bps']):.3f}")
    md_lines.append(f"- Sharpe (Dynamic): {sharpe_dyn:.3f}")
    md_lines.append("")

    md_lines.append("## 2. Empirical Spread Profile")
    md_lines.append("")
    md_lines.append("Per-tier intraday U-shape (from kill-zone Part A):")
    md_lines.append("")
    md_lines.append(_md_table(sec8a))
    md_lines.append("")

    md_lines.append("## 3. Slippage Model Spec")
    md_lines.append("")
    md_lines.append("Three-component dynamic cost: `C_total = C_spread + C_impact + C_borrow`.")
    md_lines.append("")
    md_lines.append(f"- Kappa tier distribution across {len(kappa)} (fold,ticker) pairs:")
    md_lines.append("")
    kt = kappa["kappa"].value_counts().sort_index().rename("count").to_frame()
    md_lines.append(_md_table(kt))
    md_lines.append("")
    md_lines.append("Impact-prediction OLS (diagnostic, no pass/fail):")
    md_lines.append("")
    md_lines.append("```")
    for k, v in sec7.items():
        md_lines.append(f"  {k}: {v}")
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## 4. Before/After Table")
    md_lines.append("")
    md_lines.append(_md_table(sec4))
    md_lines.append("")

    md_lines.append("## 5. Cost Waterfall")
    md_lines.append("")
    md_lines.append("All values in bps of allocated capital, per trade. Cost columns are negative.")
    md_lines.append("")
    md_lines.append(_md_table(sec5))
    md_lines.append("")

    md_lines.append("## 6. Regime-Conditional Costs")
    md_lines.append("")
    md_lines.append(_md_table(sec6))
    md_lines.append("")

    md_lines.append("## 7. Spread-Vol Correlation (Diagnostic)")
    md_lines.append("")
    md_lines.append("```")
    for k, v in sec7.items():
        md_lines.append(f"  {k}: {v}")
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## 8. Kill Zone + Seasonality")
    md_lines.append("")
    md_lines.append("**Part A — intraday U-shape by tier (avg L1 spread, bps):**")
    md_lines.append("")
    md_lines.append(_md_table(sec8a))
    md_lines.append("")
    md_lines.append("**Part B — net alpha heatmap (kill_zone=True if avg net_bps < 0):**")
    md_lines.append("")
    md_lines.append(_md_table(sec8b))
    md_lines.append("")

    md_lines.append("## 9. Negative Control (Dynamic)")
    md_lines.append("")
    md_lines.append(_md_table(sec9))
    md_lines.append("")

    md_lines.append("## 10. Overfitting Diagnostics (Net)")
    md_lines.append("")
    md_lines.append(_md_table(sec10))
    md_lines.append("")

    md_lines.append("## 11. Sensitivity Analysis")
    md_lines.append("")
    md_lines.append(_md_table(sec11))
    md_lines.append("")

    md_lines.append("## 12. Verdict")
    md_lines.append("")
    md_lines.append(f"**{verdict}**")
    md_lines.append("")
    md_lines.append("Red flag triggers:")
    md_lines.append("")
    md_lines.append(f"- `dynamic_cost_blowup`: {flag_blowup}")
    md_lines.append(f"- `cost_exceeds_alpha`:  {flag_cost_exceeds}")
    md_lines.append(f"- `kappa_instability`:   {flag_kappa} (drift_count={ticker_kappa_changes})")
    md_lines.append(f"- `dsr_degradation`:     {flag_dsr}")
    md_lines.append(f"- `math_violation`:      {flag_math}")
    md_lines.append("")
    md_lines.append("All three regimes (Gross / Static / Dynamic) shown side-by-side. "
                    "No cherry-picking.")

    output_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "before_after": sec4,
        "waterfall": sec5,
        "regime_costs": sec6,
        "impact_validation": sec7,
        "kill_zone_part_a": sec8a,
        "kill_zone_part_b": sec8b,
        "neg_control": sec9,
        "overfitting": sec10,
        "sensitivity": sec11,
        "verdict": verdict,
        "red_flags": {
            "dynamic_cost_blowup": flag_blowup,
            "cost_exceeds_alpha": flag_cost_exceeds,
            "kappa_instability": flag_kappa,
            "dsr_degradation": flag_dsr,
            "math_violation": flag_math,
        },
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    res = assemble_report()
    print(f"Report written to {res['output_path']}")
