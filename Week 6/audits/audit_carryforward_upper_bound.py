"""
audit_carryforward_upper_bound.py
==================================
Upper-bound test for the carry-forward gate hypothesis.

Question:
    Of the pairs selected in fold N, what fraction would PASS a single-pair
    Johansen test (p < 0.05) when re-evaluated on fold (N+1)'s formation
    window — WITHOUT BH-FDR competition?

Why "upper bound":
    The prior measurement (consecutive-fold overlap from
    audit_posthoc_johansen.csv) gives 19.1% mean carryover. But that figure is
    constrained by BH-FDR being competitive — a pair may pass pure Johansen
    p<0.05 yet still get rejected because more attractive pairs exist in fold
    N+1. The carry-forward gate would only require pure Johansen p<0.05 on
    the specific (a,b) pair. So the gate-eligible rate >= 19.1%.

Procedure (per fold N in [1, 38]):
    1. Load fold N+1's formation data (12 months).
    2. apply_hard_screens -> universe survivors.
    3. fit_factor_model -> residual_log_prices for fold N+1.
    4. For each pair (a,b) in fold N's selected pairs:
         - eligible iff both a, b in residual matrix
         - if eligible, run _johansen_pvalue on stacked (resid_a, resid_b)
         - record p-value
    5. Aggregate.

Runtime: ~38 PCA fits + ~5,850 single-pair Johansen tests ≈ 8-10 min.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from scripts.run_v4_pipeline import (
    DATA_DAILY,
    FOLD_SCHEDULE,
    _load_all_daily,
    _slice_daily,
)
from engine.phase1_cointegration.discovery import (
    _apply_hard_screens,
    _johansen_pvalue,
)
from engine.phase1_cointegration.factor_residual import fit_factor_model


POSTHOC_CSV = WEEK6 / "results" / "v4" / "audit_posthoc_johansen.csv"
OUT_CSV = WEEK6 / "results" / "v4" / "audit_carryforward_upper_bound.csv"
N_FACTOR_COMPONENTS = 5     # matches engine_daily.discovery_daily._N_FACTOR_COMPONENTS
MIN_OBS_DAILY = 100         # matches engine_daily.discovery_daily._MIN_OBS_DAILY


def compute_fold_residuals(
    cache_daily: dict, formation_start: str, formation_end: str,
) -> tuple[pd.DataFrame, set[str]]:
    """Replicate engine_daily.discovery_daily Phase 1 prep through residual_log_prices.

    Returns (residual_log_prices_df, survivor_tickers_set).
    """
    formation_data = _slice_daily(cache_daily, formation_start, formation_end)
    if not formation_data:
        return pd.DataFrame(), set()

    survivors, _, _ = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        return pd.DataFrame(), set()

    formation_data = {t: formation_data[t] for t in survivors}

    try:
        _, factor_tickers, residual_log_prices, _ = fit_factor_model(
            formation_data, n_components=N_FACTOR_COMPONENTS,
        )
    except ValueError:
        return pd.DataFrame(), set()

    # Apply min-obs filter (same as discovery_daily)
    kept = {
        tk: residual_log_prices[tk] for tk in factor_tickers
        if residual_log_prices[tk].dropna().size >= MIN_OBS_DAILY
    }
    if not kept:
        return pd.DataFrame(), set()
    resid_df = pd.DataFrame(kept)
    return resid_df, set(kept.keys())


def test_pair_johansen(
    resid_df: pd.DataFrame, ticker_a: str, ticker_b: str,
) -> float:
    """Run Johansen on (resid_a, resid_b) from fold N+1's residual matrix."""
    s_a = resid_df[ticker_a].dropna()
    s_b = resid_df[ticker_b].dropna()
    aligned = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
    if len(aligned) < 50:    # need enough overlap for Johansen
        return np.nan
    X = aligned.values
    return _johansen_pvalue(X)


def main() -> int:
    print("=== Carry-forward upper-bound audit ===\n", flush=True)

    posthoc = pd.read_csv(POSTHOC_CSV)
    pairs_by_fold: dict[int, list[tuple[str, str]]] = {}
    for fold, group in posthoc.groupby("fold"):
        pairs_by_fold[int(fold)] = [
            tuple(p.split("/")) for p in group["pair"].tolist()
        ]

    folds_in_csv = sorted(pairs_by_fold.keys())
    print(f"posthoc CSV: {len(posthoc)} pair-fold records, folds {min(folds_in_csv)}..{max(folds_in_csv)}\n",
          flush=True)

    print("Loading daily cache...", flush=True)
    t_load = time.time()
    cache_daily = _load_all_daily(DATA_DAILY)
    print(f"  {len(cache_daily)} tickers loaded in {time.time()-t_load:.1f}s\n",
          flush=True)

    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}

    rows: list[dict] = []
    t0 = time.time()

    for fold_n in folds_in_csv:
        next_n = fold_n + 1
        if next_n not in sched_by_fold:
            print(f"  fold {fold_n}: no fold {next_n} — skip", flush=True)
            continue
        if fold_n not in pairs_by_fold:
            continue
        pairs_n = pairs_by_fold[fold_n]
        if not pairs_n:
            continue

        _, fs_next, fe_next, _ = sched_by_fold[next_n]

        t_fold = time.time()
        resid_df, survivors = compute_fold_residuals(cache_daily, fs_next, fe_next)
        if resid_df.empty:
            print(f"  fold {fold_n}->{next_n}: empty residuals (skip)", flush=True)
            continue

        n_total = len(pairs_n)
        n_eligible = 0
        n_pass_05 = 0
        n_pass_01 = 0
        n_pass_001 = 0
        for a, b in pairs_n:
            if a not in survivors or b not in survivors:
                rows.append({
                    "fold_n": fold_n, "ticker_a": a, "ticker_b": b,
                    "eligible": 0, "pval_nplus1": np.nan,
                })
                continue
            n_eligible += 1
            pval = test_pair_johansen(resid_df, a, b)
            rows.append({
                "fold_n": fold_n, "ticker_a": a, "ticker_b": b,
                "eligible": 1, "pval_nplus1": pval,
            })
            if not np.isnan(pval):
                if pval < 0.05:
                    n_pass_05 += 1
                if pval < 0.01:
                    n_pass_01 += 1
                if pval < 0.001:
                    n_pass_001 += 1

        elapsed = time.time() - t_fold
        print(
            f"  fold {fold_n:>2}->{next_n:>2}: total={n_total:>4} | "
            f"eligible={n_eligible:>4} ({100*n_eligible/n_total:.0f}%) | "
            f"p<0.05={n_pass_05:>4} ({100*n_pass_05/max(n_eligible,1):.0f}%) | "
            f"p<0.01={n_pass_01:>4} | p<0.001={n_pass_001:>4} | {elapsed:.1f}s",
            flush=True,
        )

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)\n")

    # --- Aggregate report ---
    n_total = len(df)
    n_elig = int(df["eligible"].sum())
    elig = df[df["eligible"] == 1].copy()
    elig_valid = elig.dropna(subset=["pval_nplus1"])

    print("=== AGGREGATE RESULTS ===")
    print(f"Total pair-fold records              : {n_total}")
    print(f"Eligible (both tickers in N+1 univ)  : {n_elig} ({100*n_elig/n_total:.1f}%)")
    print(f"Eligible AND Johansen ran            : {len(elig_valid)} "
          f"({100*len(elig_valid)/n_total:.1f}% of total)")
    print()
    if len(elig_valid):
        for thresh in (0.05, 0.01, 0.001):
            n_pass = int((elig_valid["pval_nplus1"] < thresh).sum())
            print(f"  p < {thresh:<5}   : {n_pass} of {len(elig_valid)} eligible "
                  f"({100*n_pass/len(elig_valid):.1f}%) | "
                  f"{100*n_pass/n_total:.1f}% of all selected")
        print()
        print(f"  median pval_n+1 : {elig_valid['pval_nplus1'].median():.4f}")
        print(f"  mean   pval_n+1 : {elig_valid['pval_nplus1'].mean():.4f}")

    print()
    print("=== Per-fold pass rate (p<0.05 of eligible) ===")
    per = elig_valid.groupby("fold_n").apply(
        lambda g: (g["pval_nplus1"] < 0.05).mean() * 100, include_groups=False
    ).round(1)
    print(per.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
