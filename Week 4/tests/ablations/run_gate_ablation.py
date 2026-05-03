"""
run_gate_ablation.py  —  Isolated test of persistence gate's marginal value
on the CURRENT config (No-EOS + Z=3.0 + delta=1e-7).

Compares on the same 10 diagnostic folds:
  A) No gate    + No-EOS + Z=3.0
  B) With gate  + No-EOS + Z=3.0    <-- Option A config

Reports per-fold delta and aggregate stats.
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()

from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase3_backtest.metrics_runner import run_fold_pnl

PHASE1_DIR = "results/metrics/phase1_folds"
DATA_1MIN  = "data/validated/1min_phase2"
DATA_5MIN  = "data/validated/5min_phase1"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05

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


def run_config(pairs_df, t1min, f1min):
    if pairs_df is None or pairs_df.empty:
        return float("nan"), 0, 0
    exec_res = run_fold_execution(
        pairs_df=pairs_df, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        return float("nan"), 0, 0
    m = run_fold_pnl(
        exec_res, t1min, pairs_df, FIXED_DELTA, {},
        total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
        tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
    )
    return float(m.get("sharpe", float("nan"))), len(exec_res), int(m.get("n_trades", 0))


print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")


rows = []
for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue
    if pairs_df.empty:
        continue

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)
    n_base  = len(pairs_df)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_base} base pairs", flush=True)

    # A) NO GATE
    t = time.time()
    s_a, ne_a, nt_a = run_config(pairs_df, t1min, f1min)
    print(f"  No gate    ({n_base:>3} pairs): SR={s_a:+.2f}  exec={ne_a}  trades={nt_a}  ({time.time()-t:.0f}s)")

    # B) WITH GATE (Option A)
    t = time.time()
    pairs_g = apply_persistence_gate(pairs_df, f5min, form_end)
    s_b, ne_b, nt_b = run_config(pairs_g, t1min, f1min)
    print(f"  With gate  ({len(pairs_g):>3} pairs): SR={s_b:+.2f}  exec={ne_b}  trades={nt_b}  ({time.time()-t:.0f}s)")

    rows.append({
        "fold": fold_n, "month": trading_month, "regime": regime,
        "n_base": n_base, "n_gate": len(pairs_g),
        "sr_no_gate": s_a, "sr_gate": s_b,
        "trades_no_gate": nt_a, "trades_gate": nt_b,
    })
    print()


df = pd.DataFrame(rows)

print("=" * 90)
print("GATE ABLATION on No-EOS + Z=3.0 + delta=1e-7")
print("=" * 90)
print(f"{'Fold':<6} {'Mo':<9} {'Reg':<5} {'Base':>5} {'Gate':>5}  {'NoGate SR':>10} {'Trd':>5}  {'Gate SR':>9} {'Trd':>5}  {'Delta':>7}")
print("-" * 90)
for _, r in df.iterrows():
    delta = (r['sr_gate'] - r['sr_no_gate']) if not (np.isnan(r['sr_gate']) or np.isnan(r['sr_no_gate'])) else float('nan')
    print(f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5} {int(r['n_base']):>5} {int(r['n_gate']):>5}  "
          f"{r['sr_no_gate']:>+10.2f} {int(r['trades_no_gate']):>5}  "
          f"{r['sr_gate']:>+9.2f} {int(r['trades_gate']):>5}  "
          f"{delta:>+7.2f}")
print("-" * 90)
ng = df['sr_no_gate'].dropna()
g  = df['sr_gate'].dropna()
print(f"{'MEAN':<6} {'':9} {'':5} {df['n_base'].mean():>5.0f} {df['n_gate'].mean():>5.0f}  "
      f"{ng.mean():>+10.2f} {df['trades_no_gate'].sum():>5.0f}  "
      f"{g.mean():>+9.2f} {df['trades_gate'].sum():>5.0f}  "
      f"{(g.mean()-ng.mean()):>+7.2f}")
print(f"{'MED':<6} {'':9} {'':5} {'':>5} {'':>5}  "
      f"{ng.median():>+10.2f} {'':>5}  {g.median():>+9.2f} {'':>5}  "
      f"{(g.median()-ng.median()):>+7.2f}")
print(f"{'POS%':<6} {'':9} {'':5} {'':>5} {'':>5}  "
      f"{(ng>0).mean()*100:>9.0f}% {'':>5}  {(g>0).mean()*100:>8.0f}% {'':>5}")

# Per-regime delta
print()
print("Per regime:")
for reg in ['Bear', 'Bull']:
    sub = df[df['regime'] == reg]
    ng_r = sub['sr_no_gate'].dropna()
    g_r  = sub['sr_gate'].dropna()
    print(f"  {reg}: NoGate={ng_r.mean():+.2f}  WithGate={g_r.mean():+.2f}  Delta={g_r.mean()-ng_r.mean():+.2f}  (n={len(sub)})")
