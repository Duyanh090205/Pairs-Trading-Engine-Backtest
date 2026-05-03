"""
run_z_sweep.py  —  Find optimal Z entry threshold with No-EOS + persistence gate.

Sweeps Z from 1.0 to 5.0 on all 10 diagnostic folds.
Uses: persistence gate (p<0.05 last month), eos_flatten=False, TC=30 bps, delta=1e-7.
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase3_backtest.metrics_runner import run_fold_pnl

PHASE1_DIR = "results/metrics/phase1_folds"
DATA_1MIN  = "data/validated/1min_phase2"
DATA_5MIN  = "data/validated/5min_phase1"

FIXED_DELTA      = 1e-7
TC_BPS           = 30.0
BORROW_BPS       = 50.0
PERSIST_PVAL_MAX = 0.05

Z_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

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
# Data helpers
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

# ---------------------------------------------------------------------------
# Persistence gate
# ---------------------------------------------------------------------------

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
        fa_g = fa[(fa.index >= gate_start) & (fa.index <= gate_end)]
        fb_g = fb[(fb.index >= gate_start) & (fb.index <= gate_end)]
        aligned = (
            fa_g[["log_close"]]
            .join(fb_g[["log_close"]], lsuffix="_a", rsuffix="_b", how="inner")
            .dropna()
        )
        if len(aligned) < 20:
            continue
        pval = _johansen_pval(aligned["log_close_a"].values, aligned["log_close_b"].values)
        if not np.isnan(pval) and pval < PERSIST_PVAL_MAX:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

print("Loading data caches...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min) — {time.time()-t0:.1f}s\n")

# Pre-load and gate all fold pairs once
fold_data = {}  # fold_n -> (pairs_pg, t1min, f1min)
print("Applying persistence gate to all folds...")
for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue
    tickers  = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min    = _slice_month(cache_1min, tickers, trading_month)
    f1min    = _slice(cache_1min, tickers, form_start, form_end)
    f5min    = _slice(cache_5min, tickers, form_start, form_end)
    pairs_pg = apply_persistence_gate(pairs_df, f5min, form_end)
    fold_data[fold_n] = (pairs_pg, t1min, f1min, regime, trading_month)
    print(f"  Fold {fold_n:02d} [{trading_month}] {regime}: {len(pairs_df)} -> {len(pairs_pg)} pairs")

print()

# ---------------------------------------------------------------------------
# Z sweep
# ---------------------------------------------------------------------------

# z_results[z] = list of (sharpe, n_trades) per fold
z_results = {z: [] for z in Z_GRID}

for z in Z_GRID:
    t_z = time.time()
    fold_sharpes = []
    fold_trades  = []
    for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
        if fold_n not in fold_data:
            continue
        pairs_pg, t1min, f1min, regime, trading_month = fold_data[fold_n]
        if pairs_pg.empty:
            continue

        exec_res = run_fold_execution(
            pairs_df=pairs_pg, trading_1min=t1min, delta=FIXED_DELTA,
            eos_flatten=False, formation_ref=f1min, entry_z=z,
        )
        if not exec_res:
            fold_sharpes.append(float("nan"))
            fold_trades.append(0)
            continue

        m = run_fold_pnl(
            exec_res, t1min, pairs_pg, FIXED_DELTA, {},
            total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
        )
        fold_sharpes.append(float(m.get("sharpe", float("nan"))))
        fold_trades.append(int(m.get("n_trades", 0)))

    z_results[z] = (fold_sharpes, fold_trades)
    valid = [s for s in fold_sharpes if not np.isnan(s)]
    mean_sr   = float(np.mean(valid)) if valid else float("nan")
    median_sr = float(np.median(valid)) if valid else float("nan")
    pct_pos   = sum(1 for s in valid if s > 0) / len(valid) if valid else 0
    total_trades = sum(fold_trades)
    print(f"  Z={z:.1f}  mean={mean_sr:+.2f}  median={median_sr:+.2f}  "
          f"pos={pct_pos:.0%}  trades={total_trades:4d}  ({time.time()-t_z:.0f}s)")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print()
print("=" * 75)
print(f"{'Z':>5}  {'Mean SR':>8}  {'Median SR':>10}  {'% Pos':>7}  {'Total Trades':>13}")
print("-" * 75)

best_z    = None
best_mean = float("-inf")

for z in Z_GRID:
    fold_sharpes, fold_trades = z_results[z]
    valid = [s for s in fold_sharpes if not np.isnan(s)]
    if not valid:
        continue
    mean_sr   = float(np.mean(valid))
    median_sr = float(np.median(valid))
    pct_pos   = sum(1 for s in valid if s > 0) / len(valid)
    total_tr  = sum(fold_trades)
    marker = "  <-- BEST" if mean_sr > best_mean and total_tr >= 5 else ""
    if mean_sr > best_mean and total_tr >= 5:
        best_mean = mean_sr
        best_z    = z
    print(f"  {z:>3.1f}  {mean_sr:>+8.2f}  {median_sr:>+10.2f}  {pct_pos:>7.0%}  {total_tr:>13d}{marker}")

print("=" * 75)

# Per-fold breakdown at best Z
if best_z is not None:
    print(f"\nPer-fold Sharpe at optimal Z={best_z}:")
    fold_sharpes, fold_trades = z_results[best_z]
    for i, (fold_n, form_start, form_end, trading_month, regime) in enumerate(TEST_FOLDS):
        if i < len(fold_sharpes):
            s = fold_sharpes[i]
            nt = fold_trades[i]
            flag = " *" if not np.isnan(s) and s > 0 else ""
            print(f"  Fold {fold_n:02d} [{trading_month}] {regime:4s}: "
                  f"{s:+.2f}  ({nt} trades){flag}")
    valid = [s for s in fold_sharpes if not np.isnan(s)]
    print(f"\n  Mean={np.mean(valid):+.2f}  Median={np.median(valid):+.2f}  "
          f"Positive={sum(1 for s in valid if s>0)}/{len(valid)} folds")

print(f"\nOptimal Z (No-EOS, TC=30, persistence gate): {best_z}")
