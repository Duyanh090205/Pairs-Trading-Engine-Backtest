"""
run_diagnostic3.py  —  TC, EOS, and Z-threshold tests on persistence-gated pairs.

Root cause confirmed: strategy has real gross alpha (Fold 2 gross SR=+6.15) but
30 bps one-side TC (60 bps RT) destroys it. Break-even TC is ~8-10 bps one-side.

Three tests on persistence-gate filtered pairs (best pair set from diagnostic2):

Test A — No EOS flatten (eos_flatten=False):
    Positions carry overnight. Removes the -25 bps avg EOS drag.
    75% of trades currently exit via EOS at a loss.

Test B — TC sensitivity sweep (0, 5, 10, 15, 20, 25, 30 bps one-side):
    Finds break-even TC across all 10 folds.

Test C — Z=3.0 entry threshold (persistence-gate pairs):
    Fewer trades, larger avg gross per trade (~25 bps vs ~17 bps at Z=2.0).

All tests use persistence-gate filtered pairs (p<0.05 on last formation month).
Baseline = persistence-gate pairs with default params (eos_flatten=True, Z=2.0, TC=30).
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
OUT_DIR    = "results/metrics/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA      = 1e-7
TC_DEFAULT       = 30.0
BORROW_BPS       = 50.0
PERSIST_PVAL_MAX = 0.05

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
    return {tk: cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]
            for tk in tickers if tk in cache
            and len(cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]) > 0}


def _slice_month(cache, tickers, month):
    y, m = int(month[:4]), int(month[5:7])
    return {tk: cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]
            for tk in tickers if tk in cache
            and len(cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]) > 0}


# ---------------------------------------------------------------------------
# Persistence gate (reused from diagnostic2)
# ---------------------------------------------------------------------------

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate(pairs_df, form_5min, form_end, pval_max=PERSIST_PVAL_MAX):
    if pairs_df.empty:
        return pd.DataFrame()
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_g = fa[(fa.index >= gate_start) & (fa.index <= gate_end)]
        fb_g = fb[(fb.index >= gate_start) & (fb.index <= gate_end)]
        aligned = (fa_g[["log_close"]].join(fb_g[["log_close"]], lsuffix="_a", rsuffix="_b", how="inner").dropna())
        if len(aligned) < 20:
            continue
        pval = _johansen_pval(aligned["log_close_a"].values, aligned["log_close_b"].values)
        if not np.isnan(pval) and pval < pval_max:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution + Sharpe helper
# ---------------------------------------------------------------------------

def _exec_sharpe(pairs_df, t1min, f1min, entry_z=2.0, eos_flatten=True, tc_bps=TC_DEFAULT):
    """Returns (sharpe, n_exec_pairs, n_trades, exec_res_dict)."""
    if pairs_df is None or pairs_df.empty:
        return float("nan"), 0, 0, {}
    exec_res = run_fold_execution(
        pairs_df=pairs_df, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=eos_flatten, formation_ref=f1min, entry_z=entry_z,
    )
    if not exec_res:
        return float("nan"), 0, 0, {}
    m = run_fold_pnl(exec_res, t1min, pairs_df, FIXED_DELTA, {},
                     total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
                     tc_bps=tc_bps, borrow_bps_yr=BORROW_BPS)
    return float(m.get("sharpe", float("nan"))), len(exec_res), int(m.get("n_trades", 0)), exec_res


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading data caches...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min) — {time.time()-t0:.1f}s\n")

# ---------------------------------------------------------------------------
# Per-fold loop
# ---------------------------------------------------------------------------

rows = []

for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        continue

    tickers   = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min     = _slice_month(cache_1min, tickers, trading_month)
    f1min     = _slice(cache_1min, tickers, form_start, form_end)
    f5min     = _slice(cache_5min, tickers, form_start, form_end)

    # Apply persistence gate (best filter from diagnostic2)
    pairs_pg = apply_persistence_gate(pairs_df, f5min, form_end)
    n_pg = len(pairs_pg)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} — {len(pairs_df)} base -> {n_pg} persist-gate pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime,
           "n_base": len(pairs_df), "n_persist": n_pg}

    # Baseline: persist-gate, EOS=True, Z=2.0, TC=30
    # exec_res_eos reused by Test B TC sweep — avoids running execution twice.
    t1 = time.time()
    s, ne, nt, exec_res_eos = _exec_sharpe(pairs_pg, t1min, f1min, entry_z=2.0, eos_flatten=True, tc_bps=30.0)
    row["baseline_sharpe"] = s
    row["baseline_trades"] = nt
    print(f"  Baseline (EOS, Z=2, TC=30): {s:+.2f}  ({ne} exec, {nt} trades, {time.time()-t1:.0f}s)")

    # Test A: No EOS flatten
    t1 = time.time()
    s_noeos, ne_noeos, nt_noeos, _ = _exec_sharpe(pairs_pg, t1min, f1min, entry_z=2.0, eos_flatten=False, tc_bps=30.0)
    row["noeos_sharpe"] = s_noeos
    row["noeos_trades"] = nt_noeos
    print(f"  Test A No-EOS (TC=30)     : {s_noeos:+.2f}  ({ne_noeos} exec, {nt_noeos} trades, {time.time()-t1:.0f}s)")

    # Test B: TC sensitivity — reuses exec_res_eos from Baseline, no re-execution.
    t1 = time.time()
    tc_sharpes = {}
    for tc in [0, 5, 10, 15, 20, 25, 30]:
        if exec_res_eos:
            m = run_fold_pnl(exec_res_eos, t1min, pairs_pg, FIXED_DELTA, {},
                             tc_bps=float(tc), borrow_bps_yr=BORROW_BPS)
            tc_sharpes[tc] = float(m.get("sharpe", float("nan")))
        else:
            tc_sharpes[tc] = float("nan")
    row.update({f"tc{tc}_sharpe": v for tc, v in tc_sharpes.items()})
    tc_str = "  ".join(f"TC{tc}:{tc_sharpes[tc]:+.1f}" for tc in [0, 10, 20, 30])
    print(f"  Test B TC sweep           : {tc_str}  ({time.time()-t1:.0f}s)")

    # Test C: Z=3.0 with EOS and TC=30
    t1 = time.time()
    s_z3, ne_z3, nt_z3, _ = _exec_sharpe(pairs_pg, t1min, f1min, entry_z=3.0, eos_flatten=True, tc_bps=30.0)
    row["z3_sharpe"] = s_z3
    row["z3_trades"] = nt_z3
    print(f"  Test C Z=3.0 (EOS, TC=30) : {s_z3:+.2f}  ({ne_z3} exec, {nt_z3} trades, {time.time()-t1:.0f}s)")

    # Bonus: No-EOS + Z=3.0 + TC=30
    t1 = time.time()
    s_combo, ne_c, nt_c, _ = _exec_sharpe(pairs_pg, t1min, f1min, entry_z=3.0, eos_flatten=False, tc_bps=30.0)
    row["combo_sharpe"] = s_combo
    row["combo_trades"] = nt_c
    print(f"  Bonus: No-EOS + Z=3 TC=30 : {s_combo:+.2f}  ({ne_c} exec, {nt_c} trades, {time.time()-t1:.0f}s)")

    rows.append(row)
    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

results = pd.DataFrame(rows)

def fmt(v):
    return f"{v:+.2f}" if isinstance(v, float) and not np.isnan(v) else "  n/a"

def col_mean(col):
    vals = results[col].dropna()
    return float(vals.mean()) if len(vals) else float("nan")

print("=" * 90)
print(f"{'Fold':<6} {'Month':<9} {'Reg':<5} "
      f"{'Baseline':>9} {'NoEOS':>7} {'Z=3.0':>7} {'NoEOS+Z3':>9}")
print("-" * 90)
for _, r in results.iterrows():
    print(f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5} "
          f"{fmt(r['baseline_sharpe']):>9} "
          f"{fmt(r['noeos_sharpe']):>7} "
          f"{fmt(r['z3_sharpe']):>7} "
          f"{fmt(r['combo_sharpe']):>9}")
print("-" * 90)
print(f"{'MEAN':<6} {'':9} {'':5} "
      f"{fmt(col_mean('baseline_sharpe')):>9} "
      f"{fmt(col_mean('noeos_sharpe')):>7} "
      f"{fmt(col_mean('z3_sharpe')):>7} "
      f"{fmt(col_mean('combo_sharpe')):>9}")
print("=" * 90)

# TC break-even table
print("\n=== TC sensitivity (mean Sharpe across all folds) ===")
print(f"  {'TC(1side)':>10} {'TC(RT)':>8} {'Mean_SR':>9}")
print("  " + "-" * 32)
for tc in [0, 5, 10, 15, 20, 25, 30]:
    col = f"tc{tc}_sharpe"
    ms = col_mean(col)
    be_marker = "  <-- BREAK-EVEN" if not np.isnan(ms) and abs(ms) < 0.5 else ""
    print(f"  {tc:>10} {tc*2:>8} {fmt(ms):>9}{be_marker}")

# Verdict
print("\nDiagnostic verdict:")
b   = col_mean("baseline_sharpe")
noe = col_mean("noeos_sharpe")
z3  = col_mean("z3_sharpe")
cmb = col_mean("combo_sharpe")
tc0 = col_mean("tc0_sharpe")

print(f"  Gross SR (TC=0):   {fmt(tc0)}")
print(f"  Baseline (TC=30):  {fmt(b)}")
print(f"  No-EOS (TC=30):    {fmt(noe)}")
print(f"  Z=3.0  (TC=30):    {fmt(z3)}")
print(f"  NoEOS+Z3 (TC=30):  {fmt(cmb)}")

be_tc = None
for tc in [0, 5, 10, 15, 20, 25, 30]:
    ms = col_mean(f"tc{tc}_sharpe")
    if not np.isnan(ms) and ms > 0:
        be_tc = tc

if be_tc is not None and be_tc > 0:
    print(f"\n  Break-even TC: {be_tc} bps one-side ({be_tc*2} bps RT)")
    print(f"  Default pipeline TC of 30 bps is {30/be_tc:.1f}x above break-even.")
elif be_tc == 0:
    print(f"\n  Break-even TC: <5 bps one-side (only TC=0 gives positive Sharpe).")
    print(f"  Default pipeline TC of 30 bps is far above break-even.")
else:
    print("\n  No break-even TC found — strategy does not achieve positive Sharpe even at TC=0.")

out_csv = f"{OUT_DIR}/diagnostic3_results.csv"
results.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")
