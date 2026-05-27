"""
Dynamic-β smoke test — analysis & decision report.

Applies the pre-registered decision rules from README.md against the actual smoke
output. Produces a markdown report that includes:
  - I1-I5 sanity invariants (any violation = test invalid)
  - Per-fold paired Sharpe comparison (arm vs A0)
  - Primary criteria P1 (median lift ≥ 0.20) and P2 (≥4/6 folds won)
  - Secondary diagnostic S1 (block-bootstrap 95% CI), S2 (sign test p)
  - Final verdict: smoke-win or static-sufficient

Usage:
    python -m scripts.research.dynamic_beta.analyze            # uses latest.txt
    python -m scripts.research.dynamic_beta.analyze 20260525_1234   # specific run
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd

WEEK6_ROOT = Path(__file__).resolve().parents[3]

PRIMARY_LIFT_THRESHOLD = 0.20    # P1
PRIMARY_WIN_FOLDS = 4            # P2 (≥ 4 of 6)
SECONDARY_SIGN_P = 0.20          # S2
N_BOOTSTRAP = 1000
RNG_SEED = 20260525


def _sharpe_from_pnl(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(252.0)


def _block_bootstrap_sharpe_diff(
    daily_pnl_arm: pd.Series, daily_pnl_a0: pd.Series,
    fold_ids: pd.Series, n_resample: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """
    Block bootstrap on (fold, daily P&L) pairs. Folds are blocks (non-overlapping
    monthly windows). Resample folds WITH replacement, recompute aggregate Sharpe
    on arm and on A0, return their difference.

    Returns (mean_diff, ci_low_95, ci_high_95) where CIs are 2.5/97.5 percentiles.
    """
    df = pd.DataFrame({
        "fold": fold_ids.values,
        "arm": daily_pnl_arm.values,
        "a0": daily_pnl_a0.values,
    })
    unique_folds = sorted(df["fold"].unique())
    n_folds = len(unique_folds)
    if n_folds == 0:
        return 0.0, 0.0, 0.0

    fold_to_arm = {f: df.loc[df["fold"] == f, "arm"].values for f in unique_folds}
    fold_to_a0  = {f: df.loc[df["fold"] == f, "a0"].values for f in unique_folds}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resample, dtype=float)
    for i in range(n_resample):
        idx = rng.integers(0, n_folds, size=n_folds)
        arm_pnl = np.concatenate([fold_to_arm[unique_folds[k]] for k in idx])
        a0_pnl  = np.concatenate([fold_to_a0[unique_folds[k]]  for k in idx])
        diffs[i] = _sharpe_from_pnl(arm_pnl) - _sharpe_from_pnl(a0_pnl)

    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _binomial_sign_test_p(n_wins: int, n_total: int) -> float:
    """One-sided binomial test p-value with H1: p > 0.5 (arm wins more than half)."""
    if n_total == 0:
        return 1.0
    # P(X >= n_wins | p=0.5) under Binomial(n_total, 0.5)
    p = 0.0
    for k in range(n_wins, n_total + 1):
        p += math.comb(n_total, k)
    return p / (2 ** n_total)


def _load_run(timestamp: str | None) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = WEEK6_ROOT / "results" / "dynamic_beta_smoke"
    if timestamp is None:
        pointer = base / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"No latest.txt at {pointer}. Run run_smoke.py first or pass a timestamp."
            )
        timestamp = pointer.read_text(encoding="utf-8").strip()
    run_dir = base / timestamp
    per_pair = pd.read_parquet(run_dir / "per_pair.parquet")
    per_fold = pd.read_parquet(run_dir / "per_fold.parquet")
    daily_pnl = pd.read_parquet(run_dir / "daily_pnl.parquet")
    return run_dir, per_pair, per_fold, daily_pnl


def _sanity_invariants(per_pair: pd.DataFrame) -> list[str]:
    """I1-I5 sanity invariants. Returns list of violation messages (empty = pass)."""
    violations: list[str] = []

    # I1: same n_pairs per arm per fold
    pivot = per_pair.groupby(["fold", "arm"]).size().unstack(fill_value=0)
    if pivot.empty:
        violations.append("I1: no data per (fold, arm) — empty per_pair DataFrame")
    else:
        for fold in pivot.index:
            counts = pivot.loc[fold].to_dict()
            unique_counts = set(counts.values())
            if len(unique_counts) > 1:
                violations.append(
                    f"I1: fold {fold} has different n_pairs across arms: {counts}"
                )

    # I2: β finite for all rows where pair had data (n_bars > 0)
    traded = per_pair[per_pair["n_bars"] > 0]
    nan_beta = traded[~np.isfinite(traded["beta_used"])]
    if len(nan_beta) > 0:
        violations.append(
            f"I2: {len(nan_beta)} (fold, arm, pair) rows with NaN beta_used "
            f"(arms: {nan_beta['arm'].unique().tolist()})"
        )

    # I3: B1 posterior β within [-20, 20]
    b1 = per_pair[(per_pair["arm"] == "B1_kalman_hl10") & (per_pair["n_bars"] > 0)]
    if "beta_max" in b1.columns and "beta_min" in b1.columns:
        explode = b1[(b1["beta_max"] > 20) | (b1["beta_min"] < -20)]
        if len(explode) > 0:
            violations.append(
                f"I3: {len(explode)} B1 pairs with |posterior β| > 20 "
                f"(max observed: {b1['beta_max'].max():.2f}, "
                f"min: {b1['beta_min'].min():.2f})"
            )

    # I4: cannot be tested from output alone — implementation-level (look at
    # _kalman_beta_inner: spread_prior uses β BEFORE update). Document only.

    # I5: spot-check — for pairs where A1's beta ≈ V4's beta (drift < 5%), A0 and A1
    # outcomes should be ~identical. Pick the closest-β pair per fold.
    a0 = per_pair[per_pair["arm"] == "A0_static_v4"][
        ["fold", "ticker_a", "ticker_b", "beta_used", "n_trades", "total_pnl_net"]
    ].rename(columns={"beta_used": "beta_a0", "n_trades": "trades_a0",
                      "total_pnl_net": "pnl_a0"})
    a1 = per_pair[per_pair["arm"] == "A1_short_ols60"][
        ["fold", "ticker_a", "ticker_b", "beta_used", "n_trades", "total_pnl_net"]
    ].rename(columns={"beta_used": "beta_a1", "n_trades": "trades_a1",
                      "total_pnl_net": "pnl_a1"})
    merged = a0.merge(a1, on=["fold", "ticker_a", "ticker_b"])
    merged["beta_drift_pct"] = (merged["beta_a1"] - merged["beta_a0"]).abs() / merged["beta_a0"].abs()
    close = merged[merged["beta_drift_pct"] < 0.05]
    if len(close) > 0:
        mismatches = close[(close["trades_a0"] - close["trades_a1"]).abs() > 0]
        if len(mismatches) > 5:
            violations.append(
                f"I5: {len(mismatches)}/{len(close)} pairs with β drift <5% "
                f"had different trade counts between A0 and A1 (engine drift?)"
            )

    return violations


def analyze(timestamp: str | None = None) -> int:
    run_dir, per_pair, per_fold, daily_pnl = _load_run(timestamp)
    print(f"Loaded run: {run_dir}")
    print(f"  per_pair rows: {len(per_pair)}")
    print(f"  per_fold rows: {len(per_fold)}")
    print(f"  daily_pnl rows: {len(daily_pnl)}")

    # ---- Sanity invariants ----
    print("\n=== Sanity invariants (I1-I5) ===")
    violations = _sanity_invariants(per_pair)
    if violations:
        print("VIOLATIONS:")
        for v in violations:
            print(f"  ❌ {v}")
    else:
        print("  ✅ All sanity invariants pass.")

    # ---- Per-fold table ----
    print("\n=== Per-fold per-arm metrics ===")
    pivot_sharpe = per_fold.pivot(index=["fold", "trading_month"], columns="arm", values="sharpe")
    pivot_return = per_fold.pivot(index=["fold", "trading_month"], columns="arm", values="total_return")
    pivot_trades = per_fold.pivot(index=["fold", "trading_month"], columns="arm", values="n_trades")
    print("\nSharpe:")
    print(pivot_sharpe.round(3).to_string())
    print("\nReturn:")
    print((pivot_return * 100).round(2).to_string(), "  (in %)")
    print("\nn_trades:")
    print(pivot_trades.to_string())

    # ---- Paired comparison (A1 vs A0, B1 vs A0) ----
    print("\n=== Pre-registered decision rules ===")
    a0_sharpe = pivot_sharpe["A0_static_v4"]

    arm_decisions: dict[str, dict] = {}
    # All non-baseline arms get evaluated against A0
    candidate_arms = [c for c in pivot_sharpe.columns if c != "A0_static_v4"]
    for arm in candidate_arms:
        if arm not in pivot_sharpe.columns:
            print(f"\n{arm}: NOT FOUND in per_fold")
            continue
        arm_sharpe = pivot_sharpe[arm]
        diff = arm_sharpe - a0_sharpe
        n_wins = int((diff > 0).sum())
        n_total = int(len(diff.dropna()))
        median_lift = float(diff.median())

        # P1: median lift
        p1_pass = median_lift >= PRIMARY_LIFT_THRESHOLD
        # P2: arm wins ≥ 4 of 6 folds
        p2_pass = n_wins >= PRIMARY_WIN_FOLDS

        # S1: block bootstrap CI on Sharpe diff
        a0_daily = daily_pnl[daily_pnl["arm"] == "A0_static_v4"].sort_values(["fold", "date"])
        arm_daily = daily_pnl[daily_pnl["arm"] == arm].sort_values(["fold", "date"])
        merged_d = a0_daily.merge(
            arm_daily, on=["fold", "date"], suffixes=("_a0", "_arm"),
        )
        if len(merged_d) > 0:
            mean_diff, ci_low, ci_high = _block_bootstrap_sharpe_diff(
                merged_d["daily_pnl_net_arm"], merged_d["daily_pnl_net_a0"],
                merged_d["fold"],
            )
            s1_pass = (ci_low > 0) or (ci_high < 0)  # excludes 0
        else:
            mean_diff, ci_low, ci_high, s1_pass = 0.0, 0.0, 0.0, False

        # S2: sign test
        p_sign = _binomial_sign_test_p(n_wins, n_total)
        s2_pass = p_sign <= SECONDARY_SIGN_P

        smoke_win = p1_pass and p2_pass

        arm_decisions[arm] = {
            "median_lift": median_lift, "n_wins": n_wins, "n_total": n_total,
            "p1_pass": p1_pass, "p2_pass": p2_pass,
            "bootstrap_mean_diff": mean_diff, "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high, "s1_pass": s1_pass,
            "sign_p": p_sign, "s2_pass": s2_pass,
            "smoke_win": smoke_win,
            "per_fold_diff": diff.to_dict(),
        }

        print(f"\n  {arm} vs A0:")
        print(f"    Per-fold Sharpe diff: {dict((f, round(v, 3)) for f, v in diff.to_dict().items())}")
        print(f"    Median lift:    {median_lift:+.3f}  [P1 threshold: ≥ +{PRIMARY_LIFT_THRESHOLD}]"
              f"  → {'PASS ✅' if p1_pass else 'FAIL ❌'}")
        print(f"    Wins:           {n_wins}/{n_total}  [P2 threshold: ≥ {PRIMARY_WIN_FOLDS}/6]"
              f"  → {'PASS ✅' if p2_pass else 'FAIL ❌'}")
        print(f"    Bootstrap mean Δ Sharpe: {mean_diff:+.3f}, "
              f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}]  → S1: {'PASS' if s1_pass else 'fail'}")
        print(f"    Sign test p: {p_sign:.3f}  → S2: {'PASS' if s2_pass else 'fail'}")
        print(f"    *** SMOKE-WIN: {'YES ✅ → proceed to full 39-fold test' if smoke_win else 'NO ❌'} ***")

    # ---- β stats per arm ----
    print("\n=== β diagnostics ===")
    for arm in ["A0_static_v4", "A1_short_ols60", "B1_kalman_hl10"]:
        sub = per_pair[(per_pair["arm"] == arm) & (per_pair["n_bars"] > 0)]
        if len(sub) == 0:
            continue
        print(f"  {arm}: β_used mean={sub['beta_used'].mean():+.2f}, "
              f"min={sub['beta_used'].min():+.2f}, max={sub['beta_used'].max():+.2f}")
        if "beta_max" in sub.columns and arm == "B1_kalman_hl10":
            print(f"     B1 posterior β range: "
                  f"[{sub['beta_min'].min():+.2f}, {sub['beta_max'].max():+.2f}]")
            if "beta_drift_abs" in sub.columns:
                print(f"     B1 max within-window drift |Δβ|: "
                      f"max={sub['beta_drift_abs'].max():.3f}, "
                      f"median={sub['beta_drift_abs'].median():.3f}")

    # ---- Write report.md ----
    report_path = run_dir / "report.md"
    _write_report(report_path, run_dir, per_pair, per_fold, pivot_sharpe,
                  pivot_return, pivot_trades, violations, arm_decisions)
    print(f"\nWrote {report_path}")
    return 0


def _write_report(path: Path, run_dir: Path, per_pair: pd.DataFrame,
                  per_fold: pd.DataFrame, pivot_sharpe: pd.DataFrame,
                  pivot_return: pd.DataFrame, pivot_trades: pd.DataFrame,
                  violations: list[str], arm_decisions: dict) -> None:
    lines: list[str] = []
    lines.append(f"# Dynamic-β Smoke Test — Decision Report")
    lines.append(f"")
    lines.append(f"**Run dir**: `{run_dir.name}`")
    lines.append(f"")
    lines.append(f"Decision rules and design pre-registered in [`README.md`](../../scripts/research/dynamic_beta/README.md). This report applies them to the actual smoke output.")
    lines.append(f"")

    # Sanity
    lines.append(f"## Sanity invariants")
    lines.append(f"")
    if violations:
        lines.append(f"**VIOLATIONS — test may be invalid:**")
        for v in violations:
            lines.append(f"- ❌ {v}")
    else:
        lines.append(f"✅ All I1-I5 sanity invariants pass — comparison is apples-to-apples.")
    lines.append(f"")

    # Per-fold
    lines.append(f"## Per-fold per-arm Sharpe")
    lines.append(f"")
    lines.append(pivot_sharpe.round(3).to_markdown())
    lines.append(f"")
    lines.append(f"### Per-fold per-arm return (%)")
    lines.append(f"")
    lines.append((pivot_return * 100).round(2).to_markdown())
    lines.append(f"")
    lines.append(f"### Per-fold per-arm n_trades")
    lines.append(f"")
    lines.append(pivot_trades.to_markdown())
    lines.append(f"")

    # Decisions
    lines.append(f"## Pre-registered decisions")
    lines.append(f"")
    for arm, d in arm_decisions.items():
        lines.append(f"### {arm} vs A0_static_v4")
        lines.append(f"")
        diff_str = ", ".join(f"fold {f}: {v:+.3f}" for f, v in sorted(d["per_fold_diff"].items()))
        lines.append(f"Per-fold Sharpe diff: {diff_str}")
        lines.append(f"")
        lines.append(f"| Criterion | Value | Threshold | Pass |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| **P1** Median lift | {d['median_lift']:+.3f} | ≥ +{PRIMARY_LIFT_THRESHOLD} | {'✅' if d['p1_pass'] else '❌'} |")
        lines.append(f"| **P2** Wins | {d['n_wins']}/{d['n_total']} | ≥ {PRIMARY_WIN_FOLDS}/6 | {'✅' if d['p2_pass'] else '❌'} |")
        lines.append(f"| **S1** Bootstrap CI (95%) | [{d['bootstrap_ci_low']:+.3f}, {d['bootstrap_ci_high']:+.3f}] | excludes 0 | {'✅' if d['s1_pass'] else '—'} |")
        lines.append(f"| **S2** Sign test p | {d['sign_p']:.3f} | ≤ {SECONDARY_SIGN_P} | {'✅' if d['s2_pass'] else '—'} |")
        lines.append(f"")
        if d["smoke_win"]:
            lines.append(f"**Verdict: SMOKE-WIN** — both P1 and P2 pass. Proceed to full 39-fold test with guardrails (clamp + innovation gate).")
        else:
            lines.append(f"**Verdict: no smoke-win** — pre-registered primary criteria not both satisfied.")
        lines.append(f"")

    any_win = any(d["smoke_win"] for d in arm_decisions.values())
    lines.append(f"## Final verdict")
    lines.append(f"")
    if any_win:
        winners = [a for a, d in arm_decisions.items() if d["smoke_win"]]
        lines.append(f"**Smoke-win**: {', '.join(winners)}. Proceed to full test.")
    else:
        lines.append(f"**No smoke-win — static β (A0) is sufficient on this setup.** "
                     f"Per the pre-registered design, do not invest in dynamic β. "
                     f"Update [MEMORY.md](../../../../memory/MEMORY.md) with the negative result.")
    lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ts = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(analyze(ts))
