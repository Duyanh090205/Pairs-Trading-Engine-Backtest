"""
Deeper sanity tests for E2 viability before committing 7-8h implementation.

Test A: Pair cointegration lifetime distribution
  - For each pair, what's the longest consecutive run of folds it appeared in?
  - Tells us: typical cointegration "shelf life"

Test B: Dropped-pair re-emergence rate
  - When a pair is OPEN at EOM fold N but NOT traded in fold N+1, does it
    re-appear in fold N+2, N+3, ... N+6?
  - If YES (high re-emergence): cointegration is just paused -> E2 closes too early
  - If NO: dropouts are permanent -> E2 is correct to close

Test C: Carried-pair outcome quality
  - For pairs in BOTH (open-at-EOM fold N) AND (traded fold N+1):
    what's the gross P&L distribution of fold N+1 trades?
  - Positive: E2 would have profited from carrying
  - Negative: E2 would have lost more (false carry)
"""
from pathlib import Path
import pandas as pd
import numpy as np

CSV = Path(r"d:\Quant Finance\Quant Program\Week 6\results\v4\audit_trades.csv")
df = pd.read_csv(CSV)
print(f"audit_trades.csv: {len(df)} trades, {df['fold'].nunique()} folds\n")

# Build per-fold pair sets
traded = df.groupby("fold")["pair"].apply(set)
open_at_eom = (df[df["exit_reason"] == "open_at_eom"]
               .groupby("fold")["pair"].apply(set))

print("=" * 70)
print("Test A: Pair cointegration lifetime distribution")
print("=" * 70)
# For each pair, find longest consecutive run of folds it was in `traded`
pair_runs = {}
all_pairs = set()
for folds in traded.values:
    all_pairs.update(folds)
for pair in all_pairs:
    folds_present = sorted(f for f in traded.index if pair in traded[f])
    if not folds_present:
        continue
    longest = 1
    curr = 1
    for i in range(1, len(folds_present)):
        if folds_present[i] == folds_present[i - 1] + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    pair_runs[pair] = longest

runs_series = pd.Series(pair_runs)
print(f"  Total unique pairs traded:       {len(runs_series):>5}")
print(f"  Mean longest consecutive run:    {runs_series.mean():>5.2f} folds")
print(f"  Median:                          {runs_series.median():>5.0f} folds")
print(f"  Max:                             {runs_series.max():>5.0f} folds")
print(f"  Pairs with run >= 3 folds:       {(runs_series >= 3).sum():>5} "
      f"({100 * (runs_series >= 3).mean():.1f}%)")
print(f"  Pairs with run = 1 fold only:    {(runs_series == 1).sum():>5} "
      f"({100 * (runs_series == 1).mean():.1f}%)")

print("\n" + "=" * 70)
print("Test B: Dropped-pair re-emergence rate")
print("=" * 70)
# For each pair open-at-EOM fold N but NOT in fold N+1 traded set,
# check if it reappears in fold N+2 .. N+6.
horizons = [2, 3, 6, 12]
results_b = {h: [] for h in horizons}
for fold_n in sorted(open_at_eom.index):
    if fold_n + 1 not in traded.index:
        continue
    eom_pairs = open_at_eom[fold_n]
    next_traded = traded[fold_n + 1]
    dropped = eom_pairs - next_traded  # not retraded in fold N+1
    if not dropped:
        continue
    for h in horizons:
        # Check folds N+2 .. N+h
        reemerged = set()
        for future_fold in range(fold_n + 2, fold_n + 1 + h):
            if future_fold in traded.index:
                reemerged |= (dropped & traded[future_fold])
        results_b[h].append({
            "fold": fold_n,
            "n_dropped": len(dropped),
            "n_reemerged": len(reemerged),
            "pct": 100.0 * len(reemerged) / len(dropped),
        })

for h in horizons:
    rows = results_b[h]
    if not rows:
        continue
    dfb = pd.DataFrame(rows)
    total_dropped = dfb["n_dropped"].sum()
    total_reemerged = dfb["n_reemerged"].sum()
    pct_overall = 100.0 * total_reemerged / total_dropped if total_dropped else 0.0
    print(f"  Within next {h:>2} folds: {total_reemerged:>4} / {total_dropped:>4} "
          f"= {pct_overall:>5.1f}%   (mean per-fold: {dfb['pct'].mean():>5.1f}%)")

print("\n" + "=" * 70)
print("Test C: Carried-pair outcome quality")
print("=" * 70)
# For each pair in (open-at-EOM fold N) AND (traded fold N+1):
# get the gross_pnl of fold N+1 trades for that pair.
overlap_pnls = []
non_overlap_pnls = []  # for comparison: fresh fold N+1 trades NOT from carry
for fold_n in sorted(open_at_eom.index):
    if fold_n + 1 not in traded.index:
        continue
    eom_pairs = open_at_eom[fold_n]
    next_fold_trades = df[df["fold"] == fold_n + 1]
    for _, trade in next_fold_trades.iterrows():
        if trade["pair"] in eom_pairs:
            overlap_pnls.append(trade["gross_pnl"])
        else:
            non_overlap_pnls.append(trade["gross_pnl"])

overlap_arr = np.array(overlap_pnls)
nonover_arr = np.array(non_overlap_pnls)
print(f"  Carried pair trades (fold N+1):  {len(overlap_arr):>5}")
print(f"    mean gross_pnl:                ${overlap_arr.mean():>+9.0f}")
print(f"    median gross_pnl:              ${np.median(overlap_arr):>+9.0f}")
print(f"    % positive:                    {100 * (overlap_arr > 0).mean():>5.1f}%")
print()
print(f"  Non-carried pair trades (baseline): {len(nonover_arr):>5}")
print(f"    mean gross_pnl:                ${nonover_arr.mean():>+9.0f}")
print(f"    median gross_pnl:              ${np.median(nonover_arr):>+9.0f}")
print(f"    % positive:                    {100 * (nonover_arr > 0).mean():>5.1f}%")
print()
diff = overlap_arr.mean() - nonover_arr.mean()
print(f"  Carry premium (mean diff):       ${diff:>+9.0f}")
if diff > 0:
    print("  -> Carried pairs OUTPERFORM baseline -> E2 has signal")
else:
    print("  -> Carried pairs UNDERPERFORM baseline -> E2 carrying bleeders")

print("\n" + "=" * 70)
print("SYNTHESIS for E2 go/no-go")
print("=" * 70)
print(f"  A: Median consecutive cointegration = {runs_series.median():.0f} fold(s)")
print(f"     -> {'short' if runs_series.median() <= 1 else 'multi-fold'} cointegration tenure")
print(f"  B: Dropped re-emergence within 6 folds = "
      f"{100 * sum(r['n_reemerged'] for r in results_b[6]) / max(sum(r['n_dropped'] for r in results_b[6]), 1):.1f}%")
print(f"     -> {'high false-dropout' if 100 * sum(r['n_reemerged'] for r in results_b[6]) / max(sum(r['n_dropped'] for r in results_b[6]), 1) > 30 else 'permanent dropouts'}")
print(f"  C: Carry premium = ${overlap_arr.mean() - nonover_arr.mean():+.0f}")
print(f"     -> {'E2 picks bleeders' if diff < 0 else 'E2 picks winners'}")
