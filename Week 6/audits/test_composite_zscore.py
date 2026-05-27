"""
test_composite_zscore.py
========================
Post-hoc test of composite stress z-score filter on Z=3.0 results.

Per HMM research recommendation (documents/HMM Regime Detection.md):

    stress_z(t) = z(vol_60d) - z(corr_60d) + z(dispersion_20d)
                   trailing 252d normalization

    halt fold N iff:
        stress_z(t*) > q_67(trailing 252d) AND vol(t*) > median(trailing 252d)
        where t* = last trading day of month N-1 (decision moment)

Forward-looking: only uses data available at end of fold N-1's formation.
No engine re-run needed — just filter existing Z=3.0 fold_metrics post-hoc.

Compare:
    1. Z=3.0 no filter (baseline)
    2. Z=3.0 + B3 (current production: top-tertile vol of whole dataset)
    3. Z=3.0 + composite (proposed: forward-looking 3-feature rule)

Pass criteria (locked before run):
    - Mean Sharpe ≥ B3's +1.37
    - Monthly Sharpe annualized ≥ B3's +1.00
    - Catch at least 2 of 3 2026 folds
    - n_folds_kept reasonable (15-30, not too aggressive)
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

from scripts.run_v4_pipeline import (
    DATA_DAILY, FOLD_SCHEDULE, _load_all_daily,
)


# Config — locked per research recommendation
VOL_WINDOW = 60          # realized vol window
CORR_WINDOW = 60         # pairwise correlation window
DISPERSION_WINDOW = 20   # cross-sectional dispersion window
ZSCORE_WINDOW = 252      # trailing window for z-score normalization
THRESHOLD_QUANTILE = 0.67  # top tertile of trailing stress_z
MIN_BURNIN = 126         # use expanding window until this many days, then rolling 252


def compute_daily_features(cache_daily: dict) -> pd.DataFrame:
    """Build a (T, 3) DataFrame of daily regime features.

    Features:
        vol_60d              : rolling 60d annualized vol of EW index
        avg_pairwise_corr_60d: avg pairwise Pearson corr, rolling 60d
        dispersion_20d       : 20d-avg of cross-sectional std of daily returns
    """
    # Build big returns DataFrame: index = dates, columns = tickers
    returns_dict = {}
    for tk, df in cache_daily.items():
        rets = df["log_close"].diff()
        returns_dict[tk] = rets
    R = pd.DataFrame(returns_dict).sort_index()
    print(f"  Returns matrix: {R.shape[0]} days x {R.shape[1]} tickers", flush=True)

    # EW index returns
    eq_returns = R.mean(axis=1)

    # vol_60d: rolling 60d annualized std of EW index
    vol_60d = eq_returns.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)

    # dispersion_20d: cross-sectional std per day, then rolling 20d mean
    cs_std = R.std(axis=1)
    dispersion_20d = cs_std.rolling(DISPERSION_WINDOW, min_periods=5).mean()

    # avg_pairwise_corr_60d: compute on rolling 60d windows
    # This is expensive — use a tractable proxy: corr of TOP 50 tickers by liquidity
    # Or sample 100 tickers. Or compute full corr on rolling basis.
    # For tractability: sample ~80 tickers (alphabetical first 80 with full data)
    R_clean = R.dropna(axis=1, thresh=int(0.95 * len(R)))  # tickers with 95% completeness
    sample_tickers = sorted(R_clean.columns.tolist())[:80]
    R_sample = R_clean[sample_tickers]
    print(f"  Sampling {len(sample_tickers)} tickers for pairwise corr (alphabetical, top complete)",
          flush=True)

    corr_60d_values = []
    dates = R_sample.index
    for i in range(len(dates)):
        if i < CORR_WINDOW:
            corr_60d_values.append(np.nan)
            continue
        window = R_sample.iloc[i - CORR_WINDOW:i].dropna(axis=1, thresh=50)
        if window.shape[1] < 10:
            corr_60d_values.append(np.nan)
            continue
        C = window.corr().values
        n = C.shape[0]
        off_diag = C[np.triu_indices(n, k=1)]
        corr_60d_values.append(np.nanmean(off_diag))
        if i % 200 == 0:
            print(f"    corr progress: {i}/{len(dates)}", flush=True)
    avg_pairwise_corr_60d = pd.Series(corr_60d_values, index=dates)

    feats = pd.DataFrame({
        "vol_60d": vol_60d,
        "corr_60d": avg_pairwise_corr_60d,
        "dispersion_20d": dispersion_20d,
    })
    return feats


def rolling_zscore_expanding(series: pd.Series, window: int, min_burnin: int) -> pd.Series:
    """Z-score with expanding window until min_burnin, then rolling `window`."""
    z = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < min_burnin:
            # Use expanding window
            past = series.iloc[:i].dropna()
        else:
            past = series.iloc[max(0, i - window):i].dropna()
        if len(past) < 20:
            z.iloc[i] = np.nan
            continue
        mu, sigma = past.mean(), past.std(ddof=1)
        if sigma <= 1e-12:
            z.iloc[i] = 0.0
            continue
        z.iloc[i] = (series.iloc[i] - mu) / sigma
    return z


def compute_composite_score(feats: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored columns + composite stress_z column to feats."""
    feats = feats.copy()
    feats["z_vol"] = rolling_zscore_expanding(feats["vol_60d"], ZSCORE_WINDOW, MIN_BURNIN)
    feats["z_corr"] = rolling_zscore_expanding(feats["corr_60d"], ZSCORE_WINDOW, MIN_BURNIN)
    feats["z_disp"] = rolling_zscore_expanding(feats["dispersion_20d"], ZSCORE_WINDOW, MIN_BURNIN)
    # Composite: high vol + low corr + high dispersion = stress
    feats["stress_z"] = feats["z_vol"] - feats["z_corr"] + feats["z_disp"]
    return feats


def apply_halt_filter(feats: pd.DataFrame, z25_results: pd.DataFrame,
                      mode: str = "composite") -> pd.DataFrame:
    """For each fold, decide halt based on signal at last trading day of formation.

    mode='composite': stress_z > q_67(trailing) AND vol > median(trailing)
    mode='composite_simple': stress_z > q_67(trailing) only
    mode='b3': top-tertile vol_60d on whole dataset (replicates current B3)
    """
    out = []
    for _, fold_row in z25_results.iterrows():
        fold_n = int(fold_row["fold"])
        tm = fold_row["trading_month"]

        # Find last trading day of formation = day before trading month starts
        trade_start = pd.Timestamp(tm + "-01")
        decision_dates = feats.index[feats.index < trade_start]
        if len(decision_dates) == 0:
            out.append({**fold_row.to_dict(), "halt": False, "halt_reason": "no_data"})
            continue
        t_star = decision_dates[-1]

        # Trailing 252d window ending at t_star
        past_end = feats.index.get_loc(t_star)
        past_start = max(0, past_end - ZSCORE_WINDOW)
        past_window = feats.iloc[past_start:past_end]

        if mode == "composite":
            sz = feats.loc[t_star, "stress_z"]
            vl = feats.loc[t_star, "vol_60d"]
            past_sz = past_window["stress_z"].dropna()
            past_vol = past_window["vol_60d"].dropna()
            if len(past_sz) < 30 or len(past_vol) < 30 or np.isnan(sz) or np.isnan(vl):
                halt, reason = False, "burnin"
            else:
                q67_sz = past_sz.quantile(THRESHOLD_QUANTILE)
                med_vol = past_vol.median()
                halt = (sz > q67_sz) and (vl > med_vol)
                reason = f"sz={sz:+.2f}(q67={q67_sz:+.2f}), vol={vl:.1%}(med={med_vol:.1%})"
        elif mode == "composite_simple":
            sz = feats.loc[t_star, "stress_z"]
            past_sz = past_window["stress_z"].dropna()
            if len(past_sz) < 30 or np.isnan(sz):
                halt, reason = False, "burnin"
            else:
                q67_sz = past_sz.quantile(THRESHOLD_QUANTILE)
                halt = sz > q67_sz
                reason = f"sz={sz:+.2f}(q67={q67_sz:+.2f})"
        elif mode == "b3":
            # BUG FIX (2026-05-25, /deep-audit-bug): previous version used
            # feats["vol_60d"].dropna() (WHOLE dataset) for threshold —
            # look-ahead bias. Now uses trailing 252d window (past_window) only.
            vl = feats.loc[t_star, "vol_60d"]
            past_vol = past_window["vol_60d"].dropna()
            if len(past_vol) < 30 or np.isnan(vl):
                halt, reason = False, "burnin"
            else:
                q67_trailing = past_vol.quantile(THRESHOLD_QUANTILE)
                halt = vl > q67_trailing
                reason = f"vol={vl:.1%}(q67_trailing={q67_trailing:.1%})"
        else:
            halt, reason = False, "unknown_mode"

        out.append({**fold_row.to_dict(), "halt": bool(halt), "halt_reason": reason,
                    "decision_day": str(t_star.date())})
    return pd.DataFrame(out)


def summarize(name: str, df: pd.DataFrame) -> dict:
    kept = df[(~df["halt"]) & (df["n_trades"] > 0)]
    traded = df[df["n_trades"] > 0]
    n_halts = int(df["halt"].sum())
    if len(kept) < 2:
        return {"name": name, "n_kept": len(kept), "n_halted": n_halts}
    ms_ann = float(kept["total_return"].mean() / kept["total_return"].std(ddof=1) * np.sqrt(12)) \
        if kept["total_return"].std(ddof=1) > 1e-12 else 0.0
    # Halt accuracy: of halted folds, what % were actually negative?
    halted_traded = df[df["halt"] & (df["n_trades"] > 0)]
    halt_accuracy = (halted_traded["sharpe"] < 0).mean() if len(halted_traded) else np.nan
    return {
        "name": name,
        "n_kept": len(kept),
        "n_halted": n_halts,
        "mean_sharpe": round(kept["sharpe"].mean(), 3),
        "median_sharpe": round(kept["sharpe"].median(), 3),
        "monthly_sharpe_ann": round(ms_ann, 3),
        "sum_return": round(kept["total_return"].sum(), 4),
        "winners": f"{(kept['sharpe']>0).sum()}/{len(kept)}",
        "halt_accuracy": round(halt_accuracy, 2) if not np.isnan(halt_accuracy) else None,
    }


def main() -> int:
    print("=== Composite z-score filter test ===\n", flush=True)

    print("Loading cache...", flush=True)
    cache = _load_all_daily(DATA_DAILY)
    print(f"  {len(cache)} tickers\n", flush=True)

    print("Computing daily regime features...", flush=True)
    feats = compute_daily_features(cache)
    feats = compute_composite_score(feats)
    print(f"  Features computed: {len(feats)} days, "
          f"valid stress_z from {feats['stress_z'].dropna().index[0].date()}\n",
          flush=True)

    z30 = pd.read_csv(WEEK6 / "results" / "v4" / "z30_dyncost" / "fold_metrics.csv")
    print(f"Loaded Z=3.0 results: {len(z30)} folds\n", flush=True)

    # Apply each mode
    no_filter = z30.copy()
    no_filter["halt"] = False
    no_filter["halt_reason"] = "no_filter"

    b3 = apply_halt_filter(feats, z30, mode="b3")
    comp_simple = apply_halt_filter(feats, z30, mode="composite_simple")
    composite = apply_halt_filter(feats, z30, mode="composite")

    # Show 2026 folds explicitly
    print("=== 2026 fold decisions ===")
    for label, df in [("B3 (vol-only)", b3), ("Composite simple", comp_simple),
                       ("Composite double-threshold", composite)]:
        sub = df[df["trading_month"].str.startswith("2026")]
        print(f"\n  {label}:")
        for _, r in sub.iterrows():
            print(f"    Fold {r['fold']} ({r['trading_month']}): "
                  f"halt={r['halt']}, sharpe={r['sharpe']:+.3f}, reason={r['halt_reason']}")

    # Summary table
    print("\n=== Summary by filter ===")
    results = [
        summarize("No filter (Z=3.0)", no_filter),
        summarize("B3 (vol-only, forward-looking trailing 252d)", b3),
        summarize("Composite SIMPLE (stress_z > q_67 trailing 252d)", comp_simple),
        summarize("Composite DOUBLE (stress_z > q_67 AND vol > median)", composite),
    ]
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Save
    out_csv = WEEK6 / "results" / "v4" / "composite_zscore_test.csv"
    composite.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    feats_csv = WEEK6 / "results" / "v4" / "composite_zscore_features.csv"
    feats.to_csv(feats_csv)
    print(f"Wrote {feats_csv}")

    # Pre-committed verdict
    print()
    print("=== PRE-COMMITTED VERDICT ===")
    comp_summary = results[3]
    b3_summary = results[1]
    print(f"  Composite double Sharpe : {comp_summary['mean_sharpe']:+.3f}, monthly {comp_summary['monthly_sharpe_ann']:+.3f}")
    print(f"  B3 (current) Sharpe     : {b3_summary['mean_sharpe']:+.3f}, monthly {b3_summary['monthly_sharpe_ann']:+.3f}")
    print()
    lift = comp_summary['mean_sharpe'] - b3_summary['mean_sharpe']
    monthly_lift = comp_summary['monthly_sharpe_ann'] - b3_summary['monthly_sharpe_ann']
    print(f"  Lift composite vs B3    : Sharpe {lift:+.3f}, monthly {monthly_lift:+.3f}")
    print()
    if comp_summary['mean_sharpe'] >= b3_summary['mean_sharpe']:
        print(f"  -> Composite >= B3: SHIP composite z-score (cleaner mechanism + better/equal performance)")
    else:
        print(f"  -> Composite < B3: STICK with B3 for now (composite did NOT improve)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
