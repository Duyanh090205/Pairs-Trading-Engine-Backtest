"""
run_divergence_audit.py  —  Measure post-entry spread behavior.

For each trade across all 32 Option A folds, classify the post-entry path:
  - REVERT   : |Z| moves toward 0 from entry
  - DIVERGE  : |Z| keeps growing (gets MORE extreme) before reverting
  - DRIFT    : |Z| stays roughly flat (no decisive move)

Compute per Z-bucket and per regime:
  - % of trades that diverge before reverting
  - Max adverse |Z| excursion vs entry |Z|
  - Time-to-revert (bars from entry to first |Z|<0.5 cross)
  - % of trades that NEVER revert (held to fold-end)
"""

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()

from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase4_defense.orchestrator import FOLD_SCHEDULE

PHASE1_DIR = "results/metrics/phase1_folds"
DATA_1MIN  = "data/validated/1min_phase2"
DATA_5MIN  = "data/validated/5min_phase1"
OUT_DIR    = "results/metrics/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA   = 1e-7
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
EXIT_Z        = 0.5  # consider "reverted" when |Z| crosses below this

# ---------------------------------------------------------------------------
# Helpers (cache, slice, gate)
# ---------------------------------------------------------------------------

def _load_cache(data_dir, compute_log_close=False):
    cache = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache

def _slice(cache, tickers, start, end):
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out

def _slice_month(cache, tickers, month):
    y, m = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        sliced = cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan

def apply_persistence_gate(pairs_df, form_5min, form_end):
    if pairs_df.empty:
        return pd.DataFrame()
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(
                fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                lsuffix="_a", rsuffix="_b", how="inner",
            ).dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pval) and pval < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading data caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")


# ---------------------------------------------------------------------------
# Trade-level extraction across all folds
# ---------------------------------------------------------------------------

def classify_trades(pair_df, exit_z=EXIT_Z):
    """
    For each trade in pair_df, extract:
      entry_z      : |Z| at entry bar
      max_adv_z    : max |Z| reached AFTER entry (worst-case adverse move)
      max_adv_pct  : (max_adv_z / entry_z) - 1  (positive = diverged further)
      time_to_rev  : bars from entry to first |Z|<exit_z (NaN if never reverts)
      reverted     : True if |Z| crossed below exit_z before exit
      diverged     : True if max_adv_z > entry_z * 1.10 (10% worse before reverting)
      hold_bars    : total bars in position
    """
    pos = pair_df["position"].values
    z   = pair_df["zscore"].values if "zscore" in pair_df.columns else None
    if z is None:
        return []

    trades = []
    n = len(pos)
    i = 0
    while i < n:
        if pos[i] != 0 and (i == 0 or pos[i-1] == 0):
            # entry at bar i
            entry_z = abs(z[i])
            j = i
            while j < n and pos[j] != 0:
                j += 1
            # j is first bar with pos==0 after entry, or n
            seg_z   = np.abs(z[i:j])
            max_adv = float(np.nanmax(seg_z))
            # time to revert (first |Z|<exit_z within hold window)
            below   = np.where(seg_z < exit_z)[0]
            time_to_rev = int(below[0]) if len(below) > 0 else -1
            reverted = time_to_rev >= 0
            diverged = max_adv > entry_z * 1.10
            never_rev = (j == n) and not reverted   # held to fold end without reverting

            trades.append({
                "entry_z":     entry_z,
                "max_adv_z":   max_adv,
                "max_adv_pct": (max_adv / entry_z - 1.0) if entry_z > 0 else 0.0,
                "time_to_rev": time_to_rev,
                "reverted":    reverted,
                "diverged":    diverged,
                "never_rev":   never_rev,
                "hold_bars":   int(j - i),
            })
            i = j
        else:
            i += 1
    return trades


all_trades = []
print("Running execution + classification across all folds...")
for spec in FOLD_SCHEDULE:
    fold_n        = spec.fold_n
    form_start    = spec.form_start
    form_end      = spec.form_end
    trading_month = spec.trading_month

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if pairs_df.empty:
            continue
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue

    tickers  = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min    = _slice_month(cache_1min, tickers, trading_month)
    f1min    = _slice(cache_1min, tickers, form_start, form_end)
    f5min    = _slice(cache_5min, tickers, form_start, form_end)

    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        continue

    exec_res = run_fold_execution(
        pairs_df=pairs_gated, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        continue

    regime = "Bear" if trading_month <= "2022-12" else "Bull"

    fold_n_trades = 0
    for (ta, tb), pair_df in exec_res.items():
        trades = classify_trades(pair_df, exit_z=EXIT_Z)
        for tr in trades:
            tr["fold"]    = fold_n
            tr["month"]   = trading_month
            tr["regime"]  = regime
            tr["pair"]    = f"{ta}/{tb}"
            all_trades.append(tr)
            fold_n_trades += 1

    print(f"  Fold {fold_n:02d} [{trading_month}] {regime}: {fold_n_trades} trades classified")

df = pd.DataFrame(all_trades)
df.to_csv(f"{OUT_DIR}/divergence_trades.csv", index=False)
print(f"\nTotal trades classified: {len(df)}")
print(f"Saved: {OUT_DIR}/divergence_trades.csv\n")

if len(df) == 0:
    print("No trades — exiting.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def summary_block(sub, label):
    n = len(sub)
    if n == 0:
        return f"{label:<30}  n=0"
    pct_div    = sub["diverged"].mean()
    pct_rev    = sub["reverted"].mean()
    pct_never  = sub["never_rev"].mean()
    med_adv    = sub["max_adv_pct"].median()
    p75_adv    = sub["max_adv_pct"].quantile(0.75)
    p90_adv    = sub["max_adv_pct"].quantile(0.90)
    med_ent    = sub["entry_z"].median()
    med_max    = sub["max_adv_z"].median()
    med_hold   = sub["hold_bars"].median()
    rev_only   = sub[sub["reverted"]]
    med_ttr    = rev_only["time_to_rev"].median() if len(rev_only) > 0 else float("nan")
    return (
        f"{label:<30}  n={n:>4}  "
        f"div={pct_div:>5.0%}  rev={pct_rev:>5.0%}  never_rev={pct_never:>5.0%}  "
        f"med_adv={med_adv:>+6.0%}  p75={p75_adv:>+5.0%}  p90={p90_adv:>+5.0%}  "
        f"med_entryZ={med_ent:>4.2f}  med_maxZ={med_max:>4.2f}  "
        f"med_hold={med_hold:>5.0f}b  med_ttr={med_ttr if not np.isnan(med_ttr) else 0:>5.0f}b"
    )

print("=" * 180)
print("LEGEND")
print("  div      = % of trades where max-adverse |Z| > entry |Z| * 1.10  (i.e. spread got >=10% worse)")
print("  rev      = % of trades where |Z| crossed below 0.5 at some point during hold")
print("  never_rev= % of trades that never reverted AND were held to fold-end (open at month close)")
print("  med_adv  = median (max_adverse_Z / entry_Z - 1)  — negative means typical trade reverts immediately")
print("  med_ttr  = median bars from entry to first |Z|<0.5 cross (only for trades that did revert)")
print("=" * 180)

print("\n--- ALL TRADES ---")
print(summary_block(df, "All trades"))

print("\n--- BY REGIME ---")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    print(summary_block(sub, reg))

print("\n--- BY ENTRY-Z BUCKET ---  (these are all entries at >=3.0; bucket is on the actual entry |Z| at the bar)")
buckets = [(3.0, 3.5), (3.5, 4.0), (4.0, 5.0), (5.0, 7.0), (7.0, 100.0)]
for lo, hi in buckets:
    sub = df[(df["entry_z"] >= lo) & (df["entry_z"] < hi)]
    print(summary_block(sub, f"Z in [{lo},{hi})"))

print("\n--- REGIME x ENTRY-Z ---")
for reg in ["Bear", "Bull"]:
    print(f"\n  {reg}:")
    for lo, hi in buckets:
        sub = df[(df["regime"] == reg) & (df["entry_z"] >= lo) & (df["entry_z"] < hi)]
        print("    " + summary_block(sub, f"Z in [{lo},{hi})"))

# Headline numbers
print("\n" + "=" * 80)
print("HEADLINE FINDINGS")
print("=" * 80)
total = len(df)
n_div = int(df["diverged"].sum())
n_rev = int(df["reverted"].sum())
n_never = int(df["never_rev"].sum())
print(f"Of {total} trades opened at |Z|>=3.0 across all folds:")
print(f"  {n_div:>4} ({n_div/total:.0%}) saw spread diverge >=10% further before reverting")
print(f"  {n_rev:>4} ({n_rev/total:.0%}) eventually reverted (|Z|<0.5)")
print(f"  {n_never:>4} ({n_never/total:.0%}) NEVER reverted — held open to end of fold")
print()

print(f"Median adverse excursion: spread moves {df['max_adv_pct'].median():+.0%} further before reverting/exit")
print(f"P75 adverse excursion:    {df['max_adv_pct'].quantile(0.75):+.0%}")
print(f"P90 adverse excursion:    {df['max_adv_pct'].quantile(0.90):+.0%}")
print()

# Bear vs Bull comparison
bear = df[df["regime"] == "Bear"]
bull = df[df["regime"] == "Bull"]
print("Bear vs Bull divergence:")
print(f"  Bear: {bear['diverged'].mean():.0%} diverged  |  {bear['reverted'].mean():.0%} reverted  |  {bear['never_rev'].mean():.0%} never reverted  |  median adv = {bear['max_adv_pct'].median():+.0%}")
print(f"  Bull: {bull['diverged'].mean():.0%} diverged  |  {bull['reverted'].mean():.0%} reverted  |  {bull['never_rev'].mean():.0%} never reverted  |  median adv = {bull['max_adv_pct'].median():+.0%}")
