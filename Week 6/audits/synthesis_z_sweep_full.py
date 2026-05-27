"""
synthesis_z_sweep_full.py — Final comparison across Z entry settings
=====================================================================
Reads all 4 full-39-fold runs (baseline Z=2.0, Z=2.5, Z=3.0, Z=3.5)
and applies B3 (skip-high-vol) filter post-hoc for each.

⚠️⚠️⚠️  DEPRECATED 2026-05-25 — DO NOT USE FOR DECISIONS  ⚠️⚠️⚠️

This script contains TWO LOOK-AHEAD BIASES identified by /deep-audit-bug:

  Bug 1 (feature): compute_regime_per_fold() uses TRADING-month vol →
                   feature value computed from data unavailable at the
                   decision moment (end of formation = day before trade).

  Bug 2 (threshold): pd.qcut(avg_vol_ann, q=3) over all 39 folds →
                     tertile boundary uses future folds (knows the
                     dataset-wide distribution).

Result: prior "Z=3.0 + B3 = +1.37 Sharpe" claim from this script is
INFLATED by ~+0.7 Sharpe vs honest forward-looking equivalent (+0.42).

For HONEST regime-filter comparison, use:
    audits/test_composite_zscore.py

which implements the production-grade composite z-score filter with
strict forward-looking semantics (trailing 252d window throughout).

This script is kept ONLY as historical artifact + the no-filter mean
Sharpe numbers are still valid (no filter = no look-ahead).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from scripts.run_v4_pipeline import FOLD_SCHEDULE, DATA_DAILY, _load_all_daily, _slice_daily


RUNS = [
    # (label, csv_path)
    ("V4 baseline Z=2.0", "beta_cap5/fold_metrics.csv"),
    ("Z=2.5",             "z25_dyncost/fold_metrics.csv"),
    ("Z=3.0",             "z30_dyncost/fold_metrics.csv"),
    ("Z=3.5",             "z35_dyncost/fold_metrics.csv"),
]


def monthly_sharpe_ann(returns: pd.Series) -> float:
    """Annualized monthly Sharpe = mean / std * sqrt(12)."""
    if returns.std(ddof=1) > 1e-12:
        return float(returns.mean() / returns.std(ddof=1) * np.sqrt(12))
    return 0.0


def compute_regime_per_fold() -> pd.DataFrame:
    """Compute avg_vol_ann per fold's trading month (same as B3 analysis)."""
    print("Computing per-fold regime indicators (avg vol annualized)...")
    cache = _load_all_daily(DATA_DAILY)
    rows = []
    for fold_n, fs, fe, tm in FOLD_SCHEDULE:
        trade_start = tm + "-01"
        trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        trading_data = _slice_daily(cache, trade_start, trade_end)
        if not trading_data:
            continue
        returns = []
        for tk, df in trading_data.items():
            rets = df["log_close"].diff().dropna()
            if len(rets) >= 5:
                returns.append(rets)
        if not returns:
            continue
        R = pd.concat(returns, axis=1).dropna(how="all")
        if R.empty:
            continue
        avg_vol_daily = R.std(axis=0).mean() * np.sqrt(252)
        rows.append({"fold": fold_n, "avg_vol_ann": avg_vol_daily})
    return pd.DataFrame(rows)


def main() -> int:
    print("=== Synthesis: Z entry sweep full 39 folds ===\n")

    # ---- Load all runs ----
    dfs: dict[str, pd.DataFrame] = {}
    for label, path in RUNS:
        full = WEEK6 / "results" / "v4" / path
        if not full.exists():
            print(f"  [WARN] {full} not found -- skip")
            continue
        df = pd.read_csv(full)
        dfs[label] = df
        print(f"  Loaded {label}: {len(df)} folds from {path}")

    if not dfs:
        print("No runs found.")
        return 1

    # ---- Regime tagging ----
    reg = compute_regime_per_fold()
    reg["regime"] = pd.qcut(reg["avg_vol_ann"], q=3, labels=["Low_vol", "Mid_vol", "High_vol"])
    print(f"  Regime split: Low={(reg['regime']=='Low_vol').sum()}, "
          f"Mid={(reg['regime']=='Mid_vol').sum()}, "
          f"High={(reg['regime']=='High_vol').sum()}")
    print()

    # ---- Synthesis table ----
    rows = []
    for label, df in dfs.items():
        merged = df.merge(reg[["fold", "regime"]], on="fold")

        # All folds
        traded = merged[merged["n_trades"] > 0]
        mean_s = traded["sharpe"].mean()
        med_s = traded["sharpe"].median()
        sum_r = traded["total_return"].sum()
        ms_ann = monthly_sharpe_ann(traded["total_return"])
        wf = (traded["sharpe"] > 0).sum()
        n_total = len(traded)
        n_trades_total = int(traded["n_trades"].sum())

        rows.append({
            "config": label, "filter": "all 39",
            "n_folds": n_total, "n_trades": n_trades_total,
            "mean_sharpe": round(mean_s, 3), "median_sharpe": round(med_s, 3),
            "monthly_sharpe_ann": round(ms_ann, 3),
            "sum_return": round(sum_r, 4), "winners": f"{wf}/{n_total}",
        })

        # Filter: Low+Mid vol only
        filt = merged[merged["regime"].isin(["Low_vol", "Mid_vol"]) & (merged["n_trades"] > 0)]
        if len(filt) >= 2:
            mean_s_f = filt["sharpe"].mean()
            med_s_f = filt["sharpe"].median()
            sum_r_f = filt["total_return"].sum()
            ms_ann_f = monthly_sharpe_ann(filt["total_return"])
            wf_f = (filt["sharpe"] > 0).sum()
            n_total_f = len(filt)
            rows.append({
                "config": label, "filter": "skip high-vol (B3)",
                "n_folds": n_total_f, "n_trades": int(filt["n_trades"].sum()),
                "mean_sharpe": round(mean_s_f, 3), "median_sharpe": round(med_s_f, 3),
                "monthly_sharpe_ann": round(ms_ann_f, 3),
                "sum_return": round(sum_r_f, 4), "winners": f"{wf_f}/{n_total_f}",
            })

    synth = pd.DataFrame(rows)
    out_csv = WEEK6 / "results" / "v4" / "synthesis_z_sweep.csv"
    synth.to_csv(out_csv, index=False)

    # ---- Display ----
    print("=" * 110)
    print("FINAL SYNTHESIS")
    print("=" * 110)
    print(synth.to_string(index=False))
    print()
    print(f"Wrote {out_csv}")

    # ---- Highlight sweet spot ----
    all_runs = synth[synth["filter"] == "all 39"]
    best_all = all_runs.loc[all_runs["mean_sharpe"].idxmax()]
    print()
    print("=== SWEET SPOT IDENTIFICATION ===")
    print(f"  Best mean Sharpe (all 39 folds): {best_all['config']} "
          f"with Sharpe {best_all['mean_sharpe']:+.3f}, "
          f"monthly ann. {best_all['monthly_sharpe_ann']:+.3f}")

    b3_runs = synth[synth["filter"] == "skip high-vol (B3)"]
    if not b3_runs.empty:
        best_b3 = b3_runs.loc[b3_runs["mean_sharpe"].idxmax()]
        print(f"  Best mean Sharpe (with B3 filter): {best_b3['config']} "
              f"with Sharpe {best_b3['mean_sharpe']:+.3f}, "
              f"monthly ann. {best_b3['monthly_sharpe_ann']:+.3f}")

    # ---- Per-year breakdown for each config ----
    print()
    print("=== Per-year mean Sharpe (all 39 folds, no filter) ===")
    yearly_rows = []
    for label, df in dfs.items():
        df["year"] = pd.to_datetime(df["trading_month"]).dt.year
        traded = df[df["n_trades"] > 0]
        by_year = traded.groupby("year")["sharpe"].mean()
        for y, s in by_year.items():
            yearly_rows.append({"config": label, "year": int(y),
                                "mean_sharpe": round(float(s), 3)})
    py = pd.DataFrame(yearly_rows).pivot(index="year", columns="config", values="mean_sharpe")
    print(py.to_string(float_format=lambda x: f"{x:+.3f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
