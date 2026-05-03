"""
run_45fold_optionA.py  —  Full 45-fold run: persistence gate + No-EOS + Z=3.0.

Config:
  - Persistence gate: Johansen on last month of formation (p < 0.05)
  - eos_flatten=False  (positions carry overnight to natural exit)
  - entry_z=3.0        (fewer, higher-conviction entries)
  - delta=1e-7 fixed   (skip delta selector for speed ~15 min total)
  - TC=30 bps one-side, borrow=50 bps/yr

Outputs:
  results/metrics/optionA_fold_metrics.csv   — per-fold Sharpe/MaxDD/etc.
  results/metrics/optionA_summary.txt        — aggregate stats
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
from src.phase4_defense.orchestrator import FOLD_SCHEDULE

PHASE1_DIR    = "results/metrics/phase1_folds"
DATA_1MIN     = "data/validated/1min_phase2"
DATA_5MIN     = "data/validated/5min_phase1"
OUT_DIR       = "results/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
MAX_PAIRS     = 500
HL_MAX_DAYS   = 6.0   # recalibrated for Z=3.0: E[reversion]=HL*2.585d, 6d->15.5d < 21d window


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
# Load caches once
# ---------------------------------------------------------------------------

print("Loading data caches (1-min + 5-min)...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min) — {time.time()-t0:.1f}s\n")


# ---------------------------------------------------------------------------
# 45-fold loop
# ---------------------------------------------------------------------------

rows = []
t_total = time.time()

for spec in FOLD_SCHEDULE:
    fold_n         = spec.fold_n
    form_start     = spec.form_start
    form_end       = spec.form_end
    trading_month  = spec.trading_month

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if pairs_df.empty or len(pairs_df) == 0:
            print(f"Fold {fold_n:02d} [{trading_month}]: empty pairs — skip")
            continue
        if len(pairs_df) > MAX_PAIRS:
            pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"Fold {fold_n:02d} [{trading_month}]: no CSV — skip")
        continue

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)

    # Persistence gate
    t_fold = time.time()
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    n_base  = len(pairs_df)

    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 pairs after gate — skip")
        continue

    # HL recalibration for Z=3.0 entry
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    n_gated = len(pairs_gated)

    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 pairs after HL cap — skip")
        continue

    # Execution
    exec_res = run_fold_execution(
        pairs_df     = pairs_gated,
        trading_1min = t1min,
        delta        = FIXED_DELTA,
        eos_flatten  = False,
        formation_ref= f1min,
        entry_z      = ENTRY_Z,
    )

    if not exec_res:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 exec pairs — skip")
        continue

    # PnL + metrics
    metrics = run_fold_pnl(
        fold_results     = exec_res,
        trading_1min     = t1min,
        pairs_df         = pairs_gated,
        delta            = FIXED_DELTA,
        delta_metrics    = {},
        total_capital    = _TOTAL_CAPITAL,
        n_open_pairs_max = _N_OPEN_PAIRS_MAX,
        tc_bps           = TC_BPS,
        borrow_bps_yr    = BORROW_BPS,
    )

    elapsed = time.time() - t_fold
    sharpe  = metrics.get("sharpe", float("nan"))
    n_trades= int(metrics.get("n_trades", 0))
    flag    = " *" if not np.isnan(sharpe) and sharpe > 0 else ""

    print(f"Fold {fold_n:02d} [{trading_month}]  gate:{n_base}->{n_gated}  "
          f"exec:{len(exec_res)}  trades:{n_trades}  "
          f"SR:{sharpe:+.2f}{flag}  ({elapsed:.0f}s)")

    rows.append({
        "fold":          fold_n,
        "trading_month": trading_month,
        "form_start":    form_start,
        "form_end":      form_end,
        "n_base":        n_base,
        "n_gated":       n_gated,
        "n_exec":        len(exec_res),
        "n_trades":      n_trades,
        "sharpe":        sharpe,
        "max_dd":        metrics.get("max_dd",   float("nan")),
        "cagr":          metrics.get("cagr",     float("nan")),
        "calmar":        metrics.get("calmar",   float("nan")),
        "win_rate":      metrics.get("win_rate", float("nan")),
        "avg_net_bps":   metrics.get("avg_net_bps", float("nan")),
    })

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

if not rows:
    print("No results — exiting.")
    sys.exit(0)

results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/optionA_fold_metrics.csv", index=False)

valid = results["sharpe"].dropna()
pct_pos = (valid > 0).mean()
mean_sr = valid.mean()
med_sr  = valid.median()
std_sr  = valid.std()

# Regime breakdown (Bear=folds 1-6 trading Jul-Dec 2022, Bull=rest)
bear = results[results["trading_month"] <= "2022-12"]["sharpe"].dropna()
bull = results[results["trading_month"] >  "2022-12"]["sharpe"].dropna()

summary = f"""
=== Option A: Persistence Gate + No-EOS + Z=3.0 + Fixed Delta=1e-7 ===
Config: TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Folds completed : {len(results)} / {len(FOLD_SCHEDULE)}
Total trades    : {int(results['n_trades'].sum())}
Avg pairs/fold  : {results['n_exec'].mean():.1f} (after gate)

--- Sharpe Distribution ---
Mean            : {mean_sr:+.3f}
Median          : {med_sr:+.3f}
Std             : {std_sr:.3f}
% Positive folds: {pct_pos:.0%}
Min / Max       : {valid.min():+.2f} / {valid.max():+.2f}

--- Regime Breakdown ---
Bear (Jul-Dec 2022) : mean={bear.mean():+.3f}  n={len(bear)}  pos={( bear>0).mean():.0%}
Bull (2023+)        : mean={bull.mean():+.3f}  n={len(bull)}  pos={(bull>0).mean():.0%}

--- Comparison to Previous Runs ---
Original pipeline (45 folds)       : mean=-8.90  pos=0%
Diagnostic (10 folds, no gate)     : mean=-2.47  pos=30%
Diagnostic (10 folds, old gate)    : mean=-0.17  pos=40%
Option A   ({len(results)} folds)              : mean={mean_sr:+.3f}  pos={pct_pos:.0%}

Runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)
"""

print(summary)

# Per-fold table
print("Per-fold Sharpe:")
print(f"  {'Fold':<6} {'Month':<9} {'N_base':>7} {'N_gate':>7} {'N_exec':>7} {'Trades':>7} {'Sharpe':>8}")
print("  " + "-" * 58)
for _, r in results.iterrows():
    flag = " *" if r["sharpe"] > 0 else ""
    print(f"  {int(r['fold']):<6} {r['trading_month']:<9} "
          f"{int(r['n_base']):>7} {int(r['n_gated']):>7} {int(r['n_exec']):>7} "
          f"{int(r['n_trades']):>7} {r['sharpe']:>+8.2f}{flag}")

# Save summary
with open(f"{OUT_DIR}/optionA_summary.txt", "w") as f:
    f.write(summary)

print(f"\nSaved: {OUT_DIR}/optionA_fold_metrics.csv")
print(f"Saved: {OUT_DIR}/optionA_summary.txt")
