"""
run_diagnostic2.py  —  Test 1 and Test 4 on 10 diagnostic folds.

Test 1 (Tight FDR)      : Re-apply BH-FDR at q=0.01 to existing Phase 1 pairs.
                          Keeps only the most statistically significant pairs.
                          No Phase 1 re-run needed.

Test 4 (Persistence Gate): Re-run Johansen on each pair using only the LAST MONTH
                           of the formation window. Drop pairs that fail (p>=0.05).
                           Directly attacks the 100%->11% persistence collapse.

Test 1+4 (Combined)     : Apply both filters together.

All tests use fixed delta=1e-7 and formation-locked Z (same as baseline).
Baseline is included for reference (same as run_diagnostic.py output).

Usage:
    python run_diagnostic2.py
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

print("Warming up Numba JIT...")
from src.phase2_execution.kalman import run_kalman, warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import (
    run_fold_execution,
    _N_OPEN_PAIRS_MAX,
    _TOTAL_CAPITAL,
)
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.utils.stats import bh_fdr_correct

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PHASE1_DIR = "results/metrics/phase1_folds"
DATA_1MIN  = "data/validated/1min_phase2"
DATA_5MIN  = "data/validated/5min_phase1"
OUT_DIR    = "results/metrics/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA       = 1e-7
TC_BPS            = 30.0
BORROW_BPS        = 50.0
ENTRY_Z           = 2.0
FDR_TIGHT_Q       = 0.01   # Test 1
PERSIST_PVAL_MAX  = 0.05   # Test 4: max p-value on 1-month holdout

TEST_FOLDS = [
    (1,  "2022-01-03", "2022-06-30", "2022-07", "Bear"),
    (2,  "2022-02-01", "2022-07-31", "2022-08", "Bear"),
    (5,  "2022-05-01", "2022-10-31", "2022-11", "Bear"),
    (6,  "2022-06-01", "2022-11-30", "2022-12", "Bear"),
    (8,  "2022-08-01", "2023-01-31", "2023-02", "Bull"),
    (10, "2022-10-01", "2023-03-31", "2023-04", "Bull"),
    (11, "2022-11-01", "2023-04-30", "2023-05", "Bull"),
    (12, "2022-12-01", "2023-05-31", "2023-06", "Bull"),
    (13, "2023-01-01", "2023-06-30", "2023-07", "Bull"),
    (17, "2023-05-01", "2023-10-31", "2023-11", "Bull"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cache(data_dir: str, compute_log_close: bool) -> dict:
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


def _slice(cache: dict, tickers: set, start: str, end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_month(cache: dict, tickers: set, month: str) -> dict:
    year, mon = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == mon)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ---------------------------------------------------------------------------
# Test 1: Tight FDR (q=0.01)
# ---------------------------------------------------------------------------

def apply_tight_fdr(pairs_df: pd.DataFrame, q: float = FDR_TIGHT_Q) -> pd.DataFrame:
    """Re-apply BH-FDR at tighter q on existing johansen_pval column."""
    if pairs_df.empty:
        return pairs_df
    pvals = pairs_df["johansen_pval"].values
    reject = bh_fdr_correct(pvals, q=q)
    result = pairs_df[reject].reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Test 4: Persistence gate — last month of formation
# ---------------------------------------------------------------------------

def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    X = np.column_stack([log_a, log_b])
    try:
        res = coint_johansen(X, det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate(
    pairs_df: pd.DataFrame,
    form_5min: dict,
    form_end: str,
    pval_max: float = PERSIST_PVAL_MAX,
) -> pd.DataFrame:
    """
    Re-test Johansen on each pair using only the last calendar month of
    the formation window. Drop pairs where p >= pval_max.

    Uses 5-min log_close (same data Phase 1 used, just a 1-month slice).
    """
    if pairs_df.empty:
        return pairs_df

    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

    survivors = []
    n_tested = n_passed = n_short = n_failed = 0

    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
        if fa is None or fb is None:
            continue

        fa_gate = fa[(fa.index >= gate_start) & (fa.index <= gate_end)]
        fb_gate = fb[(fb.index >= gate_start) & (fb.index <= gate_end)]

        aligned = (
            fa_gate[["log_close"]]
            .join(fb_gate[["log_close"]], lsuffix="_a", rsuffix="_b", how="inner")
            .dropna()
        )
        n_tested += 1
        if len(aligned) < 20:
            n_short += 1
            continue

        pval = _johansen_pval(aligned["log_close_a"].values, aligned["log_close_b"].values)
        if np.isnan(pval) or pval >= pval_max:
            n_failed += 1
            continue

        n_passed += 1
        survivors.append(row)

    return pd.DataFrame(survivors).reset_index(drop=True), n_tested, n_passed, n_failed


# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------

def _run_exec_and_sharpe(pairs_df, trading_1min, formation_1min):
    """Run fold execution + PnL, return (sharpe, n_pairs_exec)."""
    if pairs_df.empty:
        return float("nan"), 0
    exec_res = run_fold_execution(
        pairs_df      = pairs_df,
        trading_1min  = trading_1min,
        delta         = FIXED_DELTA,
        eos_flatten   = True,
        formation_ref = formation_1min,
    )
    if not exec_res:
        return float("nan"), 0
    metrics = run_fold_pnl(
        fold_results     = exec_res,
        trading_1min     = trading_1min,
        pairs_df         = pairs_df,
        delta            = FIXED_DELTA,
        delta_metrics    = {},
        total_capital    = _TOTAL_CAPITAL,
        n_open_pairs_max = _N_OPEN_PAIRS_MAX,
        tc_bps           = TC_BPS,
        borrow_bps_yr    = BORROW_BPS,
    )
    return float(metrics.get("sharpe", float("nan"))), len(exec_res)


# ---------------------------------------------------------------------------
# Load caches once
# ---------------------------------------------------------------------------

print("Loading data caches (1-min + 5-min)...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} tickers (1-min), {len(cache_5min)} tickers (5-min) — {time.time()-t0:.1f}s\n")

# Load all fold pairs upfront
fold_pairs = {}
for fold_n, *_ in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        df = pd.read_csv(csv)
        if len(df) > 500:
            df = df.nsmallest(500, "johansen_pval").reset_index(drop=True)
        fold_pairs[fold_n] = df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        fold_pairs[fold_n] = pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-fold loop
# ---------------------------------------------------------------------------

rows = []

for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    pairs_df = fold_pairs.get(fold_n, pd.DataFrame())
    if pairs_df.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: no pairs — skipped")
        continue

    tickers       = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min  = _slice_month(cache_1min, tickers, trading_month)
    formation_1min = _slice(cache_1min, tickers, form_start, form_end)
    form_5min     = _slice(cache_5min, tickers, form_start, form_end)

    n_base = len(pairs_df)
    print(f"Fold {fold_n:02d} [{trading_month}] {regime} — {n_base} pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime, "n_base": n_base}

    # Baseline
    t1 = time.time()
    s, n = _run_exec_and_sharpe(pairs_df, trading_1min, formation_1min)
    row["baseline_sharpe"] = s
    row["baseline_exec"]   = n
    print(f"  Baseline   : {s:+.2f}  ({n} exec pairs, {time.time()-t1:.0f}s)")

    # Test 1: Tight FDR
    t1 = time.time()
    pairs_t1 = apply_tight_fdr(pairs_df)
    s, n = _run_exec_and_sharpe(pairs_t1, trading_1min, formation_1min)
    row["t1_n_pairs"]  = len(pairs_t1)
    row["t1_sharpe"]   = s
    row["t1_exec"]     = n
    print(f"  Test1 FDR01: {s:+.2f}  ({len(pairs_t1)} pairs -> {n} exec, {time.time()-t1:.0f}s)")

    # Test 4: Persistence gate
    t1 = time.time()
    result = apply_persistence_gate(pairs_df, form_5min, form_end)
    pairs_t4, n_tested, n_passed, n_failed = result
    s, n = _run_exec_and_sharpe(pairs_t4, trading_1min, formation_1min)
    row["t4_n_pairs"]  = len(pairs_t4)
    row["t4_sharpe"]   = s
    row["t4_exec"]     = n
    row["t4_pct_pass"] = n_passed / n_tested if n_tested else float("nan")
    print(f"  Test4 Persist: {s:+.2f}  ({len(pairs_t4)}/{n_tested} pairs passed -> {n} exec, {time.time()-t1:.0f}s)")

    # Test 1+4 Combined
    t1 = time.time()
    pairs_t14 = apply_tight_fdr(pairs_t4) if not pairs_t4.empty else pd.DataFrame()
    s, n = _run_exec_and_sharpe(pairs_t14, trading_1min, formation_1min)
    row["t14_n_pairs"] = len(pairs_t14)
    row["t14_sharpe"]  = s
    row["t14_exec"]    = n
    print(f"  Test1+4 Combo: {s:+.2f}  ({len(pairs_t14)} pairs -> {n} exec, {time.time()-t1:.0f}s)")

    rows.append(row)
    print()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

results = pd.DataFrame(rows)

def fmt(v):
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:+.2f}"
    return "  n/a"

def col_mean(col):
    vals = results[col].dropna()
    return float(vals.mean()) if len(vals) else float("nan")

print("=" * 85)
print(f"{'Fold':<6} {'Month':<9} {'Reg':<5} {'N':>4} "
      f"{'Baseline':>9} {'T1 FDR01':>9} {'T4 Persist':>11} {'T1+4 Combo':>11}")
print("-" * 85)
for _, r in results.iterrows():
    print(
        f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5} {int(r['n_base']):>4} "
        f"{fmt(r['baseline_sharpe']):>9} "
        f"{fmt(r['t1_sharpe']):>9} ({int(r.get('t1_n_pairs',0)):>3}p) "
        f"{fmt(r['t4_sharpe']):>8} ({int(r.get('t4_n_pairs',0)):>3}p) "
        f"{fmt(r['t14_sharpe']):>8} ({int(r.get('t14_n_pairs',0)):>3}p)"
    )
print("-" * 85)
print(
    f"{'MEAN':<6} {'':9} {'':5} {'':>4} "
    f"{fmt(col_mean('baseline_sharpe')):>9} "
    f"{fmt(col_mean('t1_sharpe')):>9}       "
    f"{fmt(col_mean('t4_sharpe')):>8}       "
    f"{fmt(col_mean('t14_sharpe')):>8}"
)
print("=" * 85)

# Persistence gate pass rates
print("\nPersistence gate pass rates (% of pairs surviving 1-month holdout Johansen):")
for _, r in results.iterrows():
    pct = r.get("t4_pct_pass", float("nan"))
    pct_str = f"{pct:.0%}" if not np.isnan(pct) else "n/a"
    print(f"  Fold {int(r['fold']):02d} [{r['month']}]: {int(r.get('t4_n_pairs',0))}/{int(r['n_base'])} passed ({pct_str})")

# Decision
print("\nDiagnostic verdict:")
b  = col_mean("baseline_sharpe")
t1 = col_mean("t1_sharpe")
t4 = col_mean("t4_sharpe")
t14= col_mean("t14_sharpe")

if not np.isnan(t1) and t1 > b + 1.0:
    print(f"  Test 1 POSITIVE: tighter FDR improved Sharpe {b:+.2f} -> {t1:+.2f}")
    print("  Fewer but cleaner pairs help. BH-FDR q=0.01 recommended.")
else:
    print(f"  Test 1 NEGATIVE: tighter FDR did not help ({b:+.2f} -> {t1:+.2f})")
    print("  The surviving pairs at q=0.01 are still not mean-reverting.")

if not np.isnan(t4) and t4 > b + 1.0:
    print(f"  Test 4 POSITIVE: persistence gate improved Sharpe {b:+.2f} -> {t4:+.2f}")
    print("  Pairs that stay cointegrated in the final formation month are tradeable.")
else:
    print(f"  Test 4 NEGATIVE: persistence gate did not help ({b:+.2f} -> {t4:+.2f})")
    print("  Even pairs that pass last-month Johansen don't mean-revert in trading.")

if not np.isnan(t14) and t14 > b + 1.0:
    print(f"  Combined POSITIVE: {b:+.2f} -> {t14:+.2f}. Apply both filters.")
else:
    print(f"  Combined NEGATIVE: {b:+.2f} -> {t14:+.2f}. Filters don't rescue the strategy.")

out_csv = f"{OUT_DIR}/diagnostic2_results.csv"
results.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")
