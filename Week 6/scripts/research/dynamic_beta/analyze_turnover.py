"""
Step 4 — Turnover-adjusted analysis.

Decomposes B1's net Sharpe lift into:
  - Gross edge:    Sharpe lift on pre-cost P&L (does Kalman β actually pick better trades?)
  - Cost savings:  Sharpe lift from lower turnover (B1 trades less → pays less)

If gross_lift ≈ net_lift → real β edge
If gross_lift << net_lift → lift mostly cost savings
If gross_lift < 0 < net_lift → Kalman entries are WORSE; lower turnover masks it

Usage:
    python -m scripts.research.dynamic_beta.analyze_turnover            # latest run
    python -m scripts.research.dynamic_beta.analyze_turnover 20260525_154443
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd

WEEK6_ROOT = Path(__file__).resolve().parents[3]
TOTAL_CAPITAL = 1_000_000.0


def _annual_sharpe_from_returns(returns_pct: np.ndarray, std_proxy: float) -> float:
    """Approximate annualized Sharpe given total return and a daily-std proxy."""
    if len(returns_pct) == 0 or std_proxy < 1e-12:
        return 0.0
    return 0.0   # unused; we go fold-by-fold via daily series for net Sharpe


def _load_run(timestamp: str | None) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = WEEK6_ROOT / "results" / "dynamic_beta_smoke"
    if timestamp is None:
        pointer = base / "latest.txt"
        timestamp = pointer.read_text(encoding="utf-8").strip()
    run_dir = base / timestamp
    per_pair = pd.read_parquet(run_dir / "per_pair.parquet")
    per_fold = pd.read_parquet(run_dir / "per_fold.parquet")
    daily_pnl = pd.read_parquet(run_dir / "daily_pnl.parquet")
    return run_dir, per_pair, per_fold, daily_pnl


def per_fold_decomposition(per_pair: pd.DataFrame, daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    For each (fold, arm), aggregate:
      - n_trades
      - total_pnl_gross (sum across pairs)
      - total_pnl_net (sum across pairs)
      - total_cost = gross - net
      - cost_per_trade = total_cost / n_trades
      - gross_return = total_pnl_gross / capital
      - net_return   = total_pnl_net / capital
      - net_sharpe_annual (from daily_pnl_net series)
      - gross_sharpe_proxy (annual): same daily std, gross daily mean ≈ net_daily_mean + cost/n_days
                              i.e. gross_sharpe ≈ (gross_return × ann_factor / n_days) / sigma_daily_net
        This is an APPROXIMATION because std of gross might differ from std of net.
        Costs are concentrated on entry/exit bars, so they add discrete spikes to gross,
        but for small cost magnitudes the std change is second-order.
    """
    rows = []
    per_pair_traded = per_pair[per_pair["n_bars"] > 0].copy()
    grouped = per_pair_traded.groupby(["fold", "trading_month", "arm"])
    for (fold, tm, arm), grp in grouped:
        n_trades = int(grp["n_trades"].sum())
        gross = float(grp["total_pnl_gross"].sum())
        net = float(grp["total_pnl_net"].sum())
        total_cost = gross - net
        cpt = total_cost / n_trades if n_trades > 0 else 0.0

        # daily net P&L series
        dseries = daily_pnl[(daily_pnl["fold"] == fold) & (daily_pnl["arm"] == arm)].sort_values("date")
        if len(dseries) < 2:
            net_sharpe = 0.0
            sigma_daily = 0.0
            n_days = max(len(dseries), 1)
        else:
            pnl_arr = dseries["daily_pnl_net"].values
            n_days = len(pnl_arr)
            mu = float(np.mean(pnl_arr))
            sigma_daily = float(np.std(pnl_arr, ddof=1))
            net_sharpe = (mu / sigma_daily) * math.sqrt(252.0) if sigma_daily > 1e-12 else 0.0

        # Gross Sharpe proxy: bump mean by avg-daily-cost; keep same sigma
        if n_days > 1 and sigma_daily > 1e-12:
            avg_cost_per_day = total_cost / n_days
            mu_gross = float(np.mean(pnl_arr)) + avg_cost_per_day
            gross_sharpe = (mu_gross / sigma_daily) * math.sqrt(252.0)
        else:
            gross_sharpe = 0.0

        rows.append({
            "fold": fold, "trading_month": tm, "arm": arm,
            "n_trades": n_trades,
            "total_pnl_gross": gross,
            "total_pnl_net": net,
            "total_cost": total_cost,
            "cost_per_trade": cpt,
            "gross_return_pct": 100 * gross / TOTAL_CAPITAL,
            "net_return_pct": 100 * net / TOTAL_CAPITAL,
            "net_sharpe": net_sharpe,
            "gross_sharpe_proxy": gross_sharpe,
            "n_days": n_days,
        })
    return pd.DataFrame(rows).sort_values(["fold", "arm"]).reset_index(drop=True)


def turnover_normalized_lift(per_pair: pd.DataFrame, daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Counterfactual: what if B1 had been forced to pay A0's per-trade cost?

    For each fold/arm, compute:
      - normalized_net_pnl = total_pnl_gross - (n_trades_arm × cost_per_trade_a0)
      - normalized_net_return = above / capital
      - normalized_net_sharpe (approx, same daily std as actual)

    If B1's normalized return > A0's return → B1's gross edge survives the
    cost normalization → real Kalman edge.
    """
    decomp = per_fold_decomposition(per_pair, daily_pnl)
    a0 = decomp[decomp["arm"] == "A0_static_v4"][["fold", "cost_per_trade"]].rename(
        columns={"cost_per_trade": "cost_per_trade_A0"}
    )
    merged = decomp.merge(a0, on="fold", how="left")
    merged["normalized_total_cost"] = merged["n_trades"] * merged["cost_per_trade_A0"]
    merged["normalized_total_pnl_net"] = merged["total_pnl_gross"] - merged["normalized_total_cost"]
    merged["normalized_net_return_pct"] = 100 * merged["normalized_total_pnl_net"] / TOTAL_CAPITAL

    # Normalized Sharpe proxy: same daily std as net path
    # We don't have daily-gross for exact recomputation; approximate by shifting mean.
    pnl_d = daily_pnl.copy()
    pnl_d = pnl_d.sort_values(["fold", "arm", "date"])
    per_pair_grouped = per_pair[per_pair["n_bars"] > 0].groupby(
        ["fold", "arm"]
    )["n_trades"].sum().reset_index()
    a0_cpt = decomp[decomp["arm"] == "A0_static_v4"][["fold", "cost_per_trade"]].set_index("fold")
    norm_sharpes = []
    for (fold, arm), grp_pnl in pnl_d.groupby(["fold", "arm"]):
        pnl_arr = grp_pnl["daily_pnl_net"].values
        n_days = len(pnl_arr)
        if n_days < 2:
            norm_sharpes.append({"fold": fold, "arm": arm, "normalized_net_sharpe": 0.0})
            continue
        sigma = float(np.std(pnl_arr, ddof=1))
        if sigma < 1e-12:
            norm_sharpes.append({"fold": fold, "arm": arm, "normalized_net_sharpe": 0.0})
            continue
        # original cost shift back to gross, then resubtract normalized cost
        arm_row = decomp[(decomp["fold"] == fold) & (decomp["arm"] == arm)]
        if arm_row.empty:
            continue
        arm_cost = float(arm_row["total_cost"].iloc[0])
        norm_cost = float(arm_row["n_trades"].iloc[0]) * float(
            a0_cpt.loc[fold, "cost_per_trade"]
        ) if fold in a0_cpt.index else arm_cost
        mu_net = float(np.mean(pnl_arr))
        # mu_gross = mu_net + arm_cost/n_days; mu_normalized = mu_gross - norm_cost/n_days
        # → mu_normalized = mu_net + (arm_cost - norm_cost)/n_days
        mu_norm = mu_net + (arm_cost - norm_cost) / n_days
        sharpe = (mu_norm / sigma) * math.sqrt(252.0)
        norm_sharpes.append({"fold": fold, "arm": arm, "normalized_net_sharpe": sharpe})
    norm_df = pd.DataFrame(norm_sharpes)
    merged = merged.merge(norm_df, on=["fold", "arm"], how="left")
    return merged


def main(timestamp: str | None = None) -> int:
    run_dir, per_pair, per_fold, daily_pnl = _load_run(timestamp)
    print(f"Loaded run: {run_dir.name}")

    decomp = per_fold_decomposition(per_pair, daily_pnl)
    norm = turnover_normalized_lift(per_pair, daily_pnl)

    # ---- Summary table: arm-level aggregates over all folds ----
    print("\n=== Arm-level totals across all folds ===")
    arm_totals = decomp.groupby("arm").agg(
        n_folds=("fold", "nunique"),
        n_trades_sum=("n_trades", "sum"),
        cost_total=("total_cost", "sum"),
        gross_return_mean_pct=("gross_return_pct", "mean"),
        net_return_mean_pct=("net_return_pct", "mean"),
        gross_sharpe_mean=("gross_sharpe_proxy", "mean"),
        net_sharpe_mean=("net_sharpe", "mean"),
    ).round(3)
    arm_totals["cost_per_trade_avg"] = (decomp.groupby("arm")["cost_per_trade"]
                                        .mean()
                                        .round(2))
    print(arm_totals.to_string())

    # ---- Per-arm vs A0: paired median diffs (net, gross, normalized) ----
    print("\n=== Lift decomposition: net = gross_edge + cost_savings ===")
    pivot_net = decomp.pivot(index="fold", columns="arm", values="net_sharpe")
    pivot_gross = decomp.pivot(index="fold", columns="arm", values="gross_sharpe_proxy")
    pivot_norm = norm.pivot(index="fold", columns="arm", values="normalized_net_sharpe")
    a0_net = pivot_net["A0_static_v4"]
    a0_gross = pivot_gross["A0_static_v4"]

    rows = []
    for arm in sorted([c for c in pivot_net.columns if c != "A0_static_v4"]):
        net_lift = (pivot_net[arm] - a0_net).median()
        gross_lift = (pivot_gross[arm] - a0_gross).median()
        norm_lift = (pivot_norm[arm] - pivot_norm["A0_static_v4"]).median()
        # Cost savings portion is the part of net lift NOT explained by gross edge
        cost_savings = net_lift - gross_lift
        rows.append({
            "arm": arm,
            "median_net_lift": round(net_lift, 3),
            "median_gross_lift_proxy": round(gross_lift, 3),
            "median_cost_savings_lift": round(cost_savings, 3),
            "median_norm_net_lift": round(norm_lift, 3),
            "pct_lift_from_cost_savings": (
                round(100 * cost_savings / net_lift, 1) if abs(net_lift) > 1e-6 else float("nan")
            ),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # ---- Interpretation guide ----
    print("\n=== Interpretation ===")
    for _, r in summary.iterrows():
        arm = r["arm"]
        nl = r["median_net_lift"]
        gl = r["median_gross_lift_proxy"]
        nrm = r["median_norm_net_lift"]
        pct_cs = r["pct_lift_from_cost_savings"]
        print(f"\n  {arm}:")
        print(f"    Net lift (raw):        {nl:+.3f}")
        print(f"    Gross lift (proxy):    {gl:+.3f}  ← real β edge if positive")
        print(f"    Normalized net lift:   {nrm:+.3f}  ← lift if forced to A0's cost rate")
        if not pd.isna(pct_cs):
            print(f"    % of net lift from cost savings: {pct_cs:.0f}%")
        if gl < 0 and nl > 0:
            print(f"    VERDICT: ⚠ Gross Sharpe LOWER than A0 — Kalman entries actually WORSE; "
                  f"lower turnover masks it on net.")
        elif gl < 0.1 * abs(nl) and nl > 0:
            print(f"    VERDICT: Lift mostly cost savings (gross edge <10% of net lift). "
                  f"Marginal real edge.")
        elif gl > 0.5 * nl and nl > 0:
            print(f"    VERDICT: ✓ Real β edge — gross lift captures majority of net lift.")
        elif nl <= 0:
            print(f"    VERDICT: No net lift; nothing to decompose.")
        else:
            print(f"    VERDICT: Mixed — gross lift positive but smaller than net lift, "
                  f"some real edge + some cost savings.")

    # ---- Save detailed tables ----
    out_csv = run_dir / "turnover_decomposition.csv"
    decomp.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    norm_csv = run_dir / "turnover_normalized.csv"
    norm.to_csv(norm_csv, index=False)
    print(f"Wrote {norm_csv}")
    return 0


if __name__ == "__main__":
    ts = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(ts))
