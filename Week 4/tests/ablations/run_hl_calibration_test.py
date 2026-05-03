"""
run_hl_calibration_test.py  --  Test effect of tightening HL upper bound for Z=3.0.

Math: expected reversion time from Z_entry to 0 = HL * 2.585 days (for Z=3.0, exit at 0.5 sigma)
  HL=10d -> 25.8d expected  (exceeds 21d trading window)
  HL= 7d -> 18.1d expected  (just fits)
  HL= 6d -> 15.5d expected  (conservative)

Tests four configs on 10 diagnostic folds:
  A       -- Option A baseline  (HL <= 10d, gate, no-EOS, Z=3.0)
  A+HL7   -- same + HL <= 7d
  A+HL6   -- same + HL <= 6d
  A+HL5   -- same + HL <= 5d
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
from src.phase3_backtest.metrics_runner import run_fold_pnl

PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05

HL_CAPS = [10.0, 7.0, 6.0, 5.0]   # upper bound candidates

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
            .join(fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                  lsuffix="_a", rsuffix="_b", how="inner").dropna()
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

    # Persistence gate once (shared across all HL caps)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    n_gated = len(pairs_gated)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_gated} gated pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime, "n_gated": n_gated}

    for hl_cap in HL_CAPS:
        pairs_hl = pairs_gated[pairs_gated["half_life_days"] <= hl_cap].reset_index(drop=True)
        n_hl = len(pairs_hl)
        sr, ne, nt = run_config(pairs_hl, t1min, f1min)
        label = f"hl{int(hl_cap)}"
        row[f"sr_{label}"]     = sr
        row[f"trades_{label}"] = nt
        row[f"n_{label}"]      = n_hl
        print(f"  HL<={hl_cap:.0f}d ({n_hl:>3} pairs): SR={sr:+.2f}  trades={nt}", flush=True)

    rows.append(row)
    print()

df = pd.DataFrame(rows)

print("=" * 85)
print("HL CALIBRATION TEST  --  gate + no-EOS + Z=3.0 + delta=1e-7  (10 folds)")
print("=" * 85)
header = f"{'Fold':<6} {'Mo':<9} {'Reg':<5}"
for hl in HL_CAPS:
    header += f"  {'HL<='+str(int(hl))+'d SR':>11} {'Tr':>4}"
print(header)
print("-" * 85)

for _, r in df.iterrows():
    line = f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5}"
    for hl in HL_CAPS:
        label = f"hl{int(hl)}"
        sr = r[f"sr_{label}"]
        nt = r[f"trades_{label}"]
        sr_str = f"{sr:+.2f}" if not np.isnan(sr) else "  nan"
        line += f"  {sr_str:>10} {int(nt):>4}"
    print(line)

print("-" * 85)
line = f"{'MEAN':<6} {'':<9} {'':<5}"
for hl in HL_CAPS:
    label = f"hl{int(hl)}"
    mean_sr = df[f"sr_{label}"].dropna().mean()
    total_tr = df[f"trades_{label}"].sum()
    line += f"  {mean_sr:>+10.2f} {int(total_tr):>4}"
print(line)

line = f"{'MED':<6} {'':<9} {'':<5}"
for hl in HL_CAPS:
    label = f"hl{int(hl)}"
    med_sr = df[f"sr_{label}"].dropna().median()
    line += f"  {med_sr:>+10.2f} {'':>4}"
print(line)

line = f"{'POS%':<6} {'':<9} {'':<5}"
for hl in HL_CAPS:
    label = f"hl{int(hl)}"
    pos_pct = (df[f"sr_{label}"].dropna() > 0).mean() * 100
    line += f"  {pos_pct:>9.0f}% {'':>4}"
print(line)

print()
print("Per regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    line = f"  {reg}: "
    for hl in HL_CAPS:
        label = f"hl{int(hl)}"
        mean_sr = sub[f"sr_{label}"].dropna().mean()
        line += f"HL<={int(hl)}d={mean_sr:+.2f}  "
    print(line)

print()
print("Avg pairs remaining after each HL cap:")
line = "  "
for hl in HL_CAPS:
    label = f"hl{int(hl)}"
    avg_n = df[f"n_{label}"].mean()
    line += f"HL<={int(hl)}d={avg_n:.1f}  "
print(line)

print()
print("Expected reversion time from Z=3.0: HL * 2.585 days")
for hl in HL_CAPS:
    print(f"  HL={int(hl)}d ->{hl * 2.585:.1f}d expected  ({'OK' if hl * 2.585 <= 21 else 'EXCEEDS 21d window'})")
