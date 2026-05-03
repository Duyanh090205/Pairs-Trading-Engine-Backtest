"""
run_valid_gate.py  —  Valid persistence gate: 5-month re-test + month-6 holdout.

The original gate double-dipped: Phase 1 selected pairs on 6 months,
then we gated on the last month of those same 6 months.

The valid version avoids this by:
  Step 1 — Re-test each Phase 1 pair on only the first 5 months (simulate
            what a clean Phase 1 on 5-month window would have produced).
  Step 2 — Gate on month 6, which is now a true holdout relative to step 1.

No Phase 1 re-run needed — we just re-test the existing surviving pairs.

Comparison: Baseline  vs  Old gate  vs  Valid gate
All configs: No-EOS, Z=3.0, TC=30 bps, delta=1e-7.
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

PHASE1_DIR       = "results/metrics/phase1_folds"
DATA_1MIN        = "data/validated/1min_phase2"
DATA_5MIN        = "data/validated/5min_phase1"
OUT_DIR          = "results/metrics/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA      = 1e-7
TC_BPS           = 30.0
BORROW_BPS       = 50.0
ENTRY_Z          = 3.0
JOHANSEN_PVAL    = 0.05

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
# Johansen p-value
# ---------------------------------------------------------------------------

def _johansen_pval(log_a, log_b):
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def apply_old_gate(pairs_df, form_5min, form_end):
    """
    Original (invalid) gate: Johansen on last month of formation window.
    Month 6 was already seen by Phase 1 — double-dipping.
    """
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


def apply_valid_gate(pairs_df, form_5min, form_start, form_end):
    """
    Valid gate:
      Step 1 — Re-test Johansen on first 5 months only (months 1-5).
                Simulates clean Phase 1 on 5-month window without O(N^2) re-scan.
      Step 2 — Gate on month 6: true holdout relative to the 5-month test.

    Pairs must pass BOTH steps to survive.
    """
    # Month 6 boundaries (holdout)
    gate_end   = form_end
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")

    # First 5 months: formation start up to day before month 6
    five_month_end = (pd.Timestamp(gate_start) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")

    survivors = []
    n_fail_5m = n_fail_gate = 0

    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue

        # Step 1: 5-month re-test
        aln_5m = (
            fa[(fa.index >= form_start) & (fa.index <= five_month_end)][["log_close"]]
            .join(
                fb[(fb.index >= form_start) & (fb.index <= five_month_end)][["log_close"]],
                lsuffix="_a", rsuffix="_b", how="inner",
            ).dropna()
        )
        if len(aln_5m) < 20:
            n_fail_5m += 1
            continue
        pval_5m = _johansen_pval(aln_5m["log_close_a"].values, aln_5m["log_close_b"].values)
        if np.isnan(pval_5m) or pval_5m >= JOHANSEN_PVAL:
            n_fail_5m += 1
            continue

        # Step 2: month-6 holdout gate
        aln_g = (
            fa[(fa.index >= gate_start) & (fa.index <= gate_end)][["log_close"]]
            .join(
                fb[(fb.index >= gate_start) & (fb.index <= gate_end)][["log_close"]],
                lsuffix="_a", rsuffix="_b", how="inner",
            ).dropna()
        )
        if len(aln_g) < 20:
            n_fail_gate += 1
            continue
        pval_g = _johansen_pval(aln_g["log_close_a"].values, aln_g["log_close_b"].values)
        if np.isnan(pval_g) or pval_g >= JOHANSEN_PVAL:
            n_fail_gate += 1
            continue

        survivors.append(row)

    return pd.DataFrame(survivors).reset_index(drop=True), n_fail_5m, n_fail_gate


# ---------------------------------------------------------------------------
# Execution + Sharpe helper
# ---------------------------------------------------------------------------

def _exec_sharpe(pairs_df, t1min, f1min):
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

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)
    n_base  = len(pairs_df)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} — {n_base} base pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime, "n_base": n_base}

    # Baseline: no gate, no-EOS, Z=3.0
    t1 = time.time()
    s, ne, nt = _exec_sharpe(pairs_df, t1min, f1min)
    row["base_sharpe"] = s
    row["base_trades"] = nt
    print(f"  Baseline (no gate)  : {s:+.2f}  ({ne} exec, {nt} trades, {time.time()-t1:.0f}s)")

    # Old gate: last month only (double-dipping)
    t1 = time.time()
    pairs_old = apply_old_gate(pairs_df, f5min, form_end)
    s, ne, nt = _exec_sharpe(pairs_old, t1min, f1min)
    row["old_n"]      = len(pairs_old)
    row["old_sharpe"] = s
    row["old_trades"] = nt
    print(f"  Old gate  ({len(pairs_old):3d} pairs): {s:+.2f}  ({ne} exec, {nt} trades, {time.time()-t1:.0f}s)")

    # Valid gate: 5-month re-test + month-6 holdout
    t1 = time.time()
    pairs_valid, n_fail_5m, n_fail_g = apply_valid_gate(pairs_df, f5min, form_start, form_end)
    s, ne, nt = _exec_sharpe(pairs_valid, t1min, f1min)
    row["valid_n"]        = len(pairs_valid)
    row["valid_sharpe"]   = s
    row["valid_trades"]   = nt
    row["valid_fail_5m"]  = n_fail_5m
    row["valid_fail_gate"]= n_fail_g
    print(f"  Valid gate ({len(pairs_valid):3d} pairs): {s:+.2f}  ({ne} exec, {nt} trades, "
          f"[{n_fail_5m} fail 5m, {n_fail_g} fail holdout])  {time.time()-t1:.0f}s")

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

print("=" * 80)
print(f"{'Fold':<6} {'Month':<9} {'Reg':<5} {'N':>4}  "
      f"{'No Gate':>8}  {'OldGate':>8} {'N':>4}  {'ValidGate':>9} {'N':>4}")
print("-" * 80)
for _, r in results.iterrows():
    print(
        f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5} {int(r['n_base']):>4}  "
        f"{fmt(r['base_sharpe']):>8}  "
        f"{fmt(r['old_sharpe']):>8} {int(r.get('old_n',0)):>4}  "
        f"{fmt(r['valid_sharpe']):>9} {int(r.get('valid_n',0)):>4}"
    )
print("-" * 80)
print(
    f"{'MEAN':<6} {'':9} {'':5} {'':>4}  "
    f"{fmt(col_mean('base_sharpe')):>8}  "
    f"{fmt(col_mean('old_sharpe')):>8} {'':>4}  "
    f"{fmt(col_mean('valid_sharpe')):>9}"
)
print("=" * 80)

# Survival rates
print("\nPair survival rates:")
print(f"  {'Fold':<6} {'Base':>5}  {'OldGate':>8} {'Pct':>6}  {'ValidGate':>10} {'Pct':>6}")
for _, r in results.iterrows():
    nb = int(r['n_base'])
    no = int(r.get('old_n', 0))
    nv = int(r.get('valid_n', 0))
    po = f"{no/nb:.0%}" if nb > 0 else "n/a"
    pv = f"{nv/nb:.0%}" if nb > 0 else "n/a"
    print(f"  {int(r['fold']):<6} {nb:>5}  {no:>8} {po:>6}  {nv:>10} {pv:>6}")

# Verdict
print("\nVerdict:")
b  = col_mean("base_sharpe")
og = col_mean("old_sharpe")
vg = col_mean("valid_sharpe")
print(f"  No gate:    {fmt(b)}")
print(f"  Old gate:   {fmt(og)}  (double-dipping — Phase 1 saw month 6)")
print(f"  Valid gate: {fmt(vg)}  (5-month re-test + true month-6 holdout)")

if not np.isnan(vg) and not np.isnan(og):
    diff = vg - og
    if abs(diff) < 0.3:
        print(f"\n  Gates are SIMILAR (diff={diff:+.2f}): double-dipping concern is minor.")
        print("  Old gate is a reasonable approximation.")
    elif diff > 0:
        print(f"\n  Valid gate BETTER by {diff:+.2f}: old gate was over-selecting on recent data.")
    else:
        print(f"\n  Valid gate WORSE by {diff:+.2f}: old gate's recency bias was actually helpful.")

out_csv = f"{OUT_DIR}/valid_gate_results.csv"
results.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")
