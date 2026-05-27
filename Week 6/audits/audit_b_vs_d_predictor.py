"""
audit_b_vs_d_predictor.py
==========================
Decide which carry-forward design hypothesis is more promising BEFORE
committing 25+ min to a full 39-fold re-run with bug fixes.

Hypotheses to compare:
    B: stricter gate (p < 0.01 instead of p < 0.05)
    D: gate also requires next-fold HL < 15 days

Methodology:
    For each carry candidate in audit_carryforward_upper_bound.csv:
        - already have gate pval (= Johansen pval on fold N+1's formation)
        - compute HL on fold N+1's formation residual (NEW)
    Then for each fold transition, predict:
        - which candidates would PASS each hypothesis
        - what fraction of the carry pool each hypothesis removes
        - which TICKER PAIRS are uniquely removed by B vs D vs both

Interpretation:
    - Hypothesis B is selective on STRENGTH of cointegration signal.
    - Hypothesis D is selective on SPEED of mean reversion.
    - If the audit shows weak overlap between B-cuts and D-cuts -> they
      target different failure modes (we'd want to combine).
    - If B-cuts >> D-cuts in cardinality -> B is much more restrictive
      and may starve the carry pool.

Runtime: ~5 min (38 folds × residual compute + HL compute for each pair).
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

from scripts.run_v4_pipeline import DATA_DAILY, FOLD_SCHEDULE, _load_all_daily, _slice_daily
from engine_daily.carry_forward import compute_residuals_for_gate
from engine.utils.stats import compute_ou_halflife


UPPER_BOUND_CSV = WEEK6 / "results" / "v4" / "audit_carryforward_upper_bound.csv"
OUT_CSV = WEEK6 / "results" / "v4" / "audit_b_vs_d_predictor.csv"


def main() -> int:
    print("=== B vs D predictor: HL + pval per carry candidate ===\n", flush=True)

    # Load existing upper-bound audit (already has pval_nplus1)
    ub = pd.read_csv(UPPER_BOUND_CSV)
    ub = ub[(ub["eligible"] == 1) & ub["pval_nplus1"].notna()].copy()
    print(f"upper-bound CSV: {len(ub)} eligible+valid records\n", flush=True)

    print("Loading daily cache...", flush=True)
    cache_daily = _load_all_daily(DATA_DAILY)
    print(f"  {len(cache_daily)} tickers\n", flush=True)

    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}

    rows: list[dict] = []
    t0 = time.time()

    for fold_n in sorted(ub["fold_n"].unique()):
        next_n = int(fold_n) + 1
        next_sched = sched_by_fold.get(next_n)
        if next_sched is None:
            continue

        _, fs_next, fe_next, _ = next_sched
        t_f = time.time()
        next_formation = _slice_daily(cache_daily, fs_next, fe_next)
        next_resid_df, _ = compute_residuals_for_gate(next_formation)
        if next_resid_df.empty:
            continue

        sub = ub[ub["fold_n"] == fold_n]
        n_hl_done = 0
        for _, r in sub.iterrows():
            ta, tb = r["ticker_a"], r["ticker_b"]
            if ta not in next_resid_df.columns or tb not in next_resid_df.columns:
                rows.append({"fold_n": int(fold_n), "ticker_a": ta, "ticker_b": tb,
                             "pval_nplus1": float(r["pval_nplus1"]),
                             "hl_nplus1": np.nan})
                continue
            s_a = next_resid_df[ta].dropna()
            s_b = next_resid_df[tb].dropna()
            aligned = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
            if len(aligned) < 50:
                rows.append({"fold_n": int(fold_n), "ticker_a": ta, "ticker_b": tb,
                             "pval_nplus1": float(r["pval_nplus1"]),
                             "hl_nplus1": np.nan})
                continue
            # Estimate beta via OLS on residuals (proxy for hedge ratio in next fold)
            X = aligned.values
            x, y = X[:, 0], X[:, 1]
            # beta = cov / var (regress a on b)
            denom = float(((y - y.mean()) ** 2).sum())
            if denom <= 1e-12:
                hl = np.nan
            else:
                beta = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
                spread = x - beta * y
                hl = compute_ou_halflife(spread, bars_per_day=1)
                n_hl_done += 1
            rows.append({"fold_n": int(fold_n), "ticker_a": ta, "ticker_b": tb,
                         "pval_nplus1": float(r["pval_nplus1"]),
                         "hl_nplus1": float(hl) if hl is not None else np.nan})

        elapsed = time.time() - t_f
        print(f"  fold {fold_n}->{next_n}: HL computed for {n_hl_done} / {len(sub)} pairs ({elapsed:.1f}s)",
              flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)\n")

    # --- B vs D analysis ---
    valid = df.dropna(subset=["pval_nplus1", "hl_nplus1"]).copy()
    print("=== HL distribution of carry candidates (pairs that pass current p<0.05) ===")
    eligible_05 = valid[valid["pval_nplus1"] < 0.05]
    print(f"  n            : {len(eligible_05)}")
    print(f"  HL min/med/max: {eligible_05['hl_nplus1'].min():.1f} / "
          f"{eligible_05['hl_nplus1'].median():.1f} / "
          f"{eligible_05['hl_nplus1'].max():.1f} days")
    for thresh in (5, 10, 15, 20, 30, 60):
        n = int((eligible_05["hl_nplus1"] < thresh).sum())
        print(f"  HL < {thresh:>3}d   : {n} ({100*n/len(eligible_05):.1f}%)")

    print()
    print("=== Hypothesis B vs D survival ===")
    n_b = int((valid["pval_nplus1"] < 0.01).sum())                          # B alone
    n_d = int(((valid["pval_nplus1"] < 0.05) & (valid["hl_nplus1"] < 15)).sum())  # D alone
    n_bd = int(((valid["pval_nplus1"] < 0.01) & (valid["hl_nplus1"] < 15)).sum())  # B AND D
    n_05 = int((valid["pval_nplus1"] < 0.05).sum())                         # current baseline
    print(f"  Baseline (p<0.05)              : {n_05} pairs survive ({100*n_05/len(valid):.1f}%)")
    print(f"  Hypothesis B (p<0.01)          : {n_b} ({100*n_b/len(valid):.1f}%)")
    print(f"  Hypothesis D (p<0.05 & HL<15)  : {n_d} ({100*n_d/len(valid):.1f}%)")
    print(f"  B AND D combined               : {n_bd} ({100*n_bd/len(valid):.1f}%)")

    # Overlap of B-cuts vs D-cuts
    b_cuts = set(map(tuple, valid[(valid["pval_nplus1"] >= 0.01) & (valid["pval_nplus1"] < 0.05)]
                     [["fold_n", "ticker_a", "ticker_b"]].values.tolist()))
    d_cuts = set(map(tuple, valid[(valid["pval_nplus1"] < 0.05) & (valid["hl_nplus1"] >= 15)]
                     [["fold_n", "ticker_a", "ticker_b"]].values.tolist()))
    overlap = b_cuts & d_cuts
    print()
    print(f"  B cuts ({len(b_cuts)}) and D cuts ({len(d_cuts)}) overlap: {len(overlap)} pairs")
    print(f"    B-only cuts: {len(b_cuts) - len(overlap)}")
    print(f"    D-only cuts: {len(d_cuts) - len(overlap)}")
    if max(len(b_cuts), len(d_cuts)) > 0:
        jaccard = len(overlap) / max(len(b_cuts | d_cuts), 1)
        print(f"    Jaccard similarity: {jaccard:.2f}  "
              f"({'targeting different failures' if jaccard < 0.3 else 'targeting same failures'})")

    # Correlation: pval vs HL — are slow-revert pairs the same as weak-cointegration pairs?
    print()
    corr = valid[["pval_nplus1", "hl_nplus1"]].corr().iloc[0, 1]
    print(f"=== Correlation pval vs HL on carry pool ===")
    print(f"  Pearson corr: {corr:+.3f}")
    print(f"  (high positive corr = pval and HL move together; one filter dominates)")
    print(f"  (low/zero corr = pval and HL pick different pairs; complementary filters)")

    # HL split by p-bucket
    print()
    print("=== HL distribution by gate pval bucket ===")
    for pmin, pmax in [(0.0, 0.001), (0.001, 0.01), (0.01, 0.05), (0.05, 1.0)]:
        sub = valid[(valid["pval_nplus1"] >= pmin) & (valid["pval_nplus1"] < pmax)]
        if len(sub) == 0:
            continue
        print(f"  pval in [{pmin:.3f}, {pmax:.3f}): n={len(sub)}, "
              f"HL median={sub['hl_nplus1'].median():.1f}d, "
              f"HL%<15={100*(sub['hl_nplus1']<15).mean():.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
