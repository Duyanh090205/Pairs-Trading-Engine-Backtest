"""
audit_carryforward_control.py
==============================
Control test for audit_carryforward_upper_bound.py.

Question:
    The selected-pair test reported 78% pass rate (p<0.05) on fold N+1's
    formation. But fold N and fold N+1 formations overlap 11/12 months
    (92% data overlap) — so any pair cointegrated in fold N is mechanically
    likely to test cointegrated in fold N+1 simply because the windows
    share most data.

    Is the gate ACTUALLY discriminating, or is it rubber-stamping?

Methodology:
    Same procedure as audit_carryforward_upper_bound, but with RANDOM pairs
    drawn from fold N+1's universe that were NOT selected by fold N's
    discovery.

    - Per fold N: sample K random non-selected pairs (K = number of fold N
      selected pairs, capped at 200 for runtime), test on fold N+1 formation.
    - Compare random-pair pass rate vs selected-pair 78%.

Interpretation:
    - If random pass rate << 78%: gate IS informative; 78% is real signal.
    - If random pass rate ≈ 78%: gate rubber-stamps; cointegration is so
      common across overlapping windows that selection doesn't matter.
    - Middle: gate has some value; magnitude TBD.

Determinism: numpy RNG seed=42.
Runtime: ~5 min (1 PCA per fold + ~150 Johansen tests per fold).
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

from scripts.run_v4_pipeline import DATA_DAILY, FOLD_SCHEDULE, _load_all_daily
from audits.audit_carryforward_upper_bound import (
    compute_fold_residuals,
    test_pair_johansen,
)


POSTHOC_CSV = WEEK6 / "results" / "v4" / "audit_posthoc_johansen.csv"
OUT_CSV = WEEK6 / "results" / "v4" / "audit_carryforward_control.csv"
SAMPLE_CAP = 200      # max random pairs per fold (runtime bound)
SEED = 42


def main() -> int:
    print("=== Carry-forward CONTROL: random non-selected pairs ===\n", flush=True)

    posthoc = pd.read_csv(POSTHOC_CSV)
    selected_by_fold: dict[int, set[tuple[str, str]]] = {}
    for fold, group in posthoc.groupby("fold"):
        sel = set()
        for p in group["pair"]:
            a, b = p.split("/")
            # Store both orderings so membership test is symmetric
            sel.add((a, b))
            sel.add((b, a))
        selected_by_fold[int(fold)] = sel

    folds_in_csv = sorted(selected_by_fold.keys())
    n_selected_per_fold = {
        f: len(s) // 2 for f, s in selected_by_fold.items()
    }
    print(f"posthoc CSV: folds {min(folds_in_csv)}..{max(folds_in_csv)}\n",
          flush=True)

    print("Loading daily cache...", flush=True)
    t_load = time.time()
    cache_daily = _load_all_daily(DATA_DAILY)
    print(f"  {len(cache_daily)} tickers loaded in {time.time()-t_load:.1f}s\n",
          flush=True)

    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    t0 = time.time()

    for fold_n in folds_in_csv:
        next_n = fold_n + 1
        if next_n not in sched_by_fold:
            continue
        n_selected = n_selected_per_fold.get(fold_n, 0)
        if n_selected == 0:
            continue
        k_sample = min(n_selected, SAMPLE_CAP)

        _, fs_next, fe_next, _ = sched_by_fold[next_n]

        t_fold = time.time()
        resid_df, survivors = compute_fold_residuals(cache_daily, fs_next, fe_next)
        if resid_df.empty or len(survivors) < 5:
            print(f"  fold {fold_n}->{next_n}: empty residuals (skip)", flush=True)
            continue

        survivors_list = sorted(survivors)
        sel = selected_by_fold[fold_n]

        # Random sampling without replacement of NON-selected pairs.
        # Reject-sample to skip selected pairs; cap attempts to avoid infinite loop.
        sampled: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        max_attempts = k_sample * 50
        attempts = 0
        while len(sampled) < k_sample and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, len(survivors_list), size=2)
            if i == j:
                continue
            a, b = survivors_list[i], survivors_list[j]
            if a > b:
                a, b = b, a
            key = (a, b)
            if key in seen:
                continue
            if (a, b) in sel or (b, a) in sel:
                continue  # excluded — was selected by fold N
            seen.add(key)
            sampled.append((a, b))

        if not sampled:
            print(f"  fold {fold_n}->{next_n}: no random pairs sampled", flush=True)
            continue

        n_pass_05 = 0
        n_pass_01 = 0
        n_pass_001 = 0
        n_valid = 0
        for a, b in sampled:
            pval = test_pair_johansen(resid_df, a, b)
            rows.append({
                "fold_n": fold_n, "ticker_a": a, "ticker_b": b,
                "pval_nplus1": pval,
            })
            if not np.isnan(pval):
                n_valid += 1
                if pval < 0.05:
                    n_pass_05 += 1
                if pval < 0.01:
                    n_pass_01 += 1
                if pval < 0.001:
                    n_pass_001 += 1

        elapsed = time.time() - t_fold
        print(
            f"  fold {fold_n:>2}->{next_n:>2}: sampled={len(sampled):>4} (cap={k_sample}) | "
            f"valid={n_valid:>4} | "
            f"p<0.05={n_pass_05:>4} ({100*n_pass_05/max(n_valid,1):.0f}%) | "
            f"p<0.01={n_pass_01:>4} | p<0.001={n_pass_001:>4} | {elapsed:.1f}s",
            flush=True,
        )

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)\n")

    valid = df.dropna(subset=["pval_nplus1"])

    print("=== AGGREGATE: RANDOM (non-selected) pairs ===")
    print(f"Total tests run                      : {len(df)}")
    print(f"Valid (Johansen converged)           : {len(valid)} "
          f"({100*len(valid)/max(len(df),1):.1f}%)")
    print()
    if len(valid):
        for thresh in (0.05, 0.01, 0.001):
            n_pass = int((valid["pval_nplus1"] < thresh).sum())
            print(f"  p < {thresh:<5}   : {n_pass} of {len(valid)} "
                  f"({100*n_pass/len(valid):.1f}%)")
        print()
        print(f"  median pval : {valid['pval_nplus1'].median():.4f}")
        print(f"  mean   pval : {valid['pval_nplus1'].mean():.4f}")

    print()
    print("=== COMPARISON vs SELECTED-pair test ===")
    print(f"  Selected pairs (audit_carryforward_upper_bound.py) : 78.0% pass p<0.05")
    print(f"  Random pairs   (this audit)                        : "
          f"{100*(valid['pval_nplus1']<0.05).sum()/max(len(valid),1):.1f}% pass p<0.05")
    selected_rate = 0.780
    random_rate = (valid["pval_nplus1"] < 0.05).sum() / max(len(valid), 1)
    gap = selected_rate - random_rate
    print(f"  Gap (selection signal)                             : "
          f"{100*gap:+.1f} percentage points")
    print()
    if gap >= 0.40:
        print("  -> Gate IS strongly informative (selection has large signal)")
    elif gap >= 0.20:
        print("  -> Gate has moderate signal")
    elif gap >= 0.05:
        print("  -> Gate has weak signal — selection barely beats random")
    else:
        print("  -> Gate RUBBER-STAMPS — random pairs pass nearly as often as selected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
