"""
Re-run E2 viability tests with FULL fold coverage (38/39).

Problem with previous test: 3 folds excluded (35, 36, 39).
- Fold 36: zero-trade fold (engine didn't enter, but discovery might still find pairs)
- Fold 35: fold 36 had no trades -> can't measure carry
- Fold 39: last fold, no future

Fix: run discovery for fold 36 only (no engine) to get its cointegrated pair set,
then use it to measure fold 35 carry. Fold 39 still excluded (no future possible).

Coverage: 38/39 folds for Tests A, B; 37/39 for carry-related (fold 39 excluded).
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))
sys.path.insert(0, str(WEEK6 / "scripts"))   # for `from run_v4_pipeline import ...`
logging.basicConfig(level=logging.WARNING)

from run_v4_pipeline import _load_all_daily, _slice_daily, FOLD_SCHEDULE, DATA_DAILY
from engine_daily import discovery_daily


def run_fold36_discovery() -> set[str]:
    """Run discovery for fold 36 only, return set of 'A/B' pair strings."""
    print("Loading daily cache...")
    cache = _load_all_daily(DATA_DAILY)

    # Find fold 36 in schedule
    fold_n, fs, fe, tm = next(s for s in FOLD_SCHEDULE if s[0] == 36)
    print(f"Running discovery for fold {fold_n} ({tm}) -- formation {fs}..{fe}")

    formation_daily = _slice_daily(cache, fs, fe)
    pairs_df, _ = discovery_daily.run(formation_data=formation_daily,
                                       hl_min=5.0, hl_max=30.0)
    if pairs_df.empty:
        print("Fold 36 discovery returned 0 pairs.")
        return set()

    pair_set = {f"{r['ticker_a']}/{r['ticker_b']}" for _, r in pairs_df.iterrows()}
    print(f"Fold 36: {len(pair_set)} cointegrated pairs (passed beta>0 + FDR + HL)")
    return pair_set


def main():
    fold36_coint_set = run_fold36_discovery()

    CSV = WEEK6 / "results" / "v4" / "audit_trades.csv"
    df = pd.read_csv(CSV)
    print(f"\naudit_trades.csv: {len(df)} trades, {df['fold'].nunique()} folds")

    # Per-fold sets: traded (engine entered) and cointegrated-by-discovery
    traded = df.groupby("fold")["pair"].apply(set).to_dict()
    open_at_eom = (df[df["exit_reason"] == "open_at_eom"]
                   .groupby("fold")["pair"].apply(set)).to_dict()

    # Inject fold 36's COINTEGRATED set as proxy for "would have been carryable".
    # Note: this is a STRICTER set than traded (no entry trigger filter), so it
    # gives a slightly more generous carry estimate. Mark it explicitly.
    coint = dict(traded)  # default: use traded as proxy
    coint[36] = fold36_coint_set  # fold 36: use discovery (no engine ran)

    print("\n" + "=" * 70)
    print("Updated Sanity Check (carry = open EOM N -> cointegrated N+1)")
    print("=" * 70)
    print(f"{'fold':>4} | {'open_at_eom':>12} | {'next_coint':>11} | "
          f"{'overlap':>8} | {'carry%':>7}")
    print("-" * 70)
    rows = []
    for fold_n in sorted(open_at_eom.keys()):
        next_fold = fold_n + 1
        if next_fold not in coint:
            continue  # next fold has no data (e.g., fold 39 -> 40)
        eom = open_at_eom[fold_n]
        next_set = coint[next_fold]
        overlap = eom & next_set
        pct = 100.0 * len(overlap) / len(eom) if eom else 0.0
        marker = " *" if fold_n == 35 else ""   # mark the newly covered one
        print(f"{fold_n:>4} | {len(eom):>12} | {len(next_set):>11} | "
              f"{len(overlap):>8} | {pct:>6.1f}%{marker}")
        rows.append({"fold": fold_n, "n_eom": len(eom),
                     "n_next": len(next_set), "n_overlap": len(overlap),
                     "carry_pct": pct, "newly_added": fold_n == 35})
    print("  (* = newly added with full-coverage rerun)")

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"Coverage: {len(summary)} / 39 folds")
    print(f"Mean carry %:               {summary['carry_pct'].mean():.1f}%")
    print(f"Median:                     {summary['carry_pct'].median():.1f}%")
    print(f"Max:                        {summary['carry_pct'].max():.1f}%")
    print(f"Folds carry > 70%:          {(summary['carry_pct'] > 70).sum()} / {len(summary)}")
    print(f"Folds carry < 30%:          {(summary['carry_pct'] < 30).sum()} / {len(summary)}")
    print("=" * 70)

    # Test B: dropped re-emergence with full coverage
    print("\n" + "=" * 70)
    print("Test B (full coverage): Dropped-pair re-emergence")
    print("=" * 70)
    horizons = [2, 3, 6, 12]
    for h in horizons:
        total_dropped = 0
        total_reemerged = 0
        for fold_n in sorted(open_at_eom.keys()):
            if fold_n + 1 not in coint:
                continue
            eom = open_at_eom[fold_n]
            dropped = eom - coint[fold_n + 1]
            if not dropped:
                continue
            reemerged = set()
            for future in range(fold_n + 2, fold_n + 1 + h):
                if future in coint:
                    reemerged |= (dropped & coint[future])
            total_dropped += len(dropped)
            total_reemerged += len(reemerged)
        pct = 100.0 * total_reemerged / total_dropped if total_dropped else 0.0
        print(f"  Within next {h:>2} folds: {total_reemerged:>4} / {total_dropped:>4} "
              f"= {pct:>5.1f}%")

    # Test C: carry outcome quality with full coverage
    print("\n" + "=" * 70)
    print("Test C (full coverage): Carried-pair gross_pnl")
    print("=" * 70)
    overlap_pnls, baseline_pnls = [], []
    for fold_n in sorted(open_at_eom.keys()):
        if fold_n + 1 not in coint:
            continue
        eom = open_at_eom[fold_n]
        next_fold_trades = df[df["fold"] == fold_n + 1]
        for _, t in next_fold_trades.iterrows():
            if t["pair"] in eom:
                overlap_pnls.append(t["gross_pnl"])
            else:
                baseline_pnls.append(t["gross_pnl"])
    # Note: fold 36 has no trades, so fold 35's carry can't add overlap_pnls
    # from fold 36. Carry outcome unchanged for that pair. Document this.
    over = np.array(overlap_pnls)
    base = np.array(baseline_pnls)
    print(f"  Carried trades:      {len(over):>5}  mean=${over.mean():>+9.0f}  "
          f"median=${np.median(over):>+9.0f}  %pos={100*(over>0).mean():>4.1f}%")
    print(f"  Baseline trades:     {len(base):>5}  mean=${base.mean():>+9.0f}  "
          f"median=${np.median(base):>+9.0f}  %pos={100*(base>0).mean():>4.1f}%")
    diff = over.mean() - base.mean()
    print(f"  Carry premium:       ${diff:>+9.0f}")
    print()
    print(f"  Note: Test C carry-trade count unchanged from previous run because")
    print(f"  fold 36 had zero trades -- pair X open EOM 35 + cointegrated in fold 36")
    print(f"  has NO measurable trade outcome (engine didn't enter).")


if __name__ == "__main__":
    main()
