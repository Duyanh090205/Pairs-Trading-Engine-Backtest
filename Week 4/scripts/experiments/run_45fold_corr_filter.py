"""
run_45fold_corr_filter.py  --  Full 45-fold run with formation correlation floor.

Primary finding: CORR25 (formation 5-min returns corr >= 0.25)
  - Delta vs baseline: +0.84 mean SR (on 14-fold diagnostic set)
  - Retains 77% of trades
  - Bear regime: unchanged (+1.70 vs +1.70 baseline)
  - Bad folds (spurious cointegration): -3.83 vs -5.71 baseline
  - Key mechanism: Fold 34 (tariff shock, Apr 2025) had mean formation corr=0.04
    -- Johansen passed but pairs had near-zero returns co-movement -- CORR25 skips this fold

Secondary: CORR30 (threshold 0.30), marginally better SR but 63% trade retention.

Base config (Option A): persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7
TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Outputs:
  results/metrics/corr_filter_fold_metrics.csv
  results/metrics/corr_filter_summary.txt
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

PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"
OUT_DIR     = "results/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
MAX_PAIRS     = 500
HL_MAX_DAYS   = 6.0

CORR_CONFIGS = [
    ("CORR25", 0.25),   # primary
    ("CORR30", 0.30),   # secondary
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
        s = cache[tk][(cache[tk].index >= start) & (cache[tk].index <= end)]
        if len(s) > 0:
            out[tk] = s
    return out

def _slice_month(cache, tickers, month):
    y, m = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        s = cache[tk][(cache[tk].index.year == y) & (cache[tk].index.month == m)]
        if len(s) > 0:
            out[tk] = s
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
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        aln = (
            fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
            .join(fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]],
                  lsuffix="_a", rsuffix="_b", how="inner").dropna()
        )
        if len(aln) < 20:
            continue
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pval) and pval < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)

def compute_formation_corr(pairs_df, form_5min, form_start, form_end):
    """
    Adds 'formation_corr' column: full formation-window 5-min log-return correlation.
    Pairs missing 5-min data get NaN and are excluded by any positive threshold.
    """
    corrs = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
        if fa is None or fb is None:
            corrs.append(np.nan)
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            corrs.append(np.nan)
            continue
        ra = aln["log_close_a"].diff().dropna()
        rb = aln["log_close_b"].diff().dropna()
        al2 = ra.to_frame("ra").join(rb.to_frame("rb"), how="inner").dropna()
        corrs.append(float(al2["ra"].corr(al2["rb"])))
    out = pairs_df.copy()
    out["formation_corr"] = corrs
    return out

def run_fold(pairs_df, t1min, f1min):
    if pairs_df is None or pairs_df.empty:
        return None, float("nan"), 0
    exec_res = run_fold_execution(
        pairs_df     = pairs_df,
        trading_1min = t1min,
        delta        = FIXED_DELTA,
        eos_flatten  = False,
        formation_ref= f1min,
        entry_z      = ENTRY_Z,
    )
    if not exec_res:
        return None, float("nan"), 0
    m = run_fold_pnl(
        fold_results     = exec_res,
        trading_1min     = t1min,
        pairs_df         = pairs_df,
        delta            = FIXED_DELTA,
        delta_metrics    = {},
        total_capital    = _TOTAL_CAPITAL,
        n_open_pairs_max = _N_OPEN_PAIRS_MAX,
        tc_bps           = TC_BPS,
        borrow_bps_yr    = BORROW_BPS,
    )
    return m, float(m.get("sharpe", float("nan"))), int(m.get("n_trades", 0))

# ---------------------------------------------------------------------------
# Load caches once
# ---------------------------------------------------------------------------

print("Loading data caches (1-min + 5-min)...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min) -- {time.time()-t0:.1f}s\n")

# ---------------------------------------------------------------------------
# 45-fold loop
# ---------------------------------------------------------------------------

rows = []
t_total = time.time()

for spec in FOLD_SCHEDULE:
    fold_n        = spec.fold_n
    form_start    = spec.form_start
    form_end      = spec.form_end
    trading_month = spec.trading_month

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if pairs_df.empty or len(pairs_df) == 0:
            print(f"Fold {fold_n:02d} [{trading_month}]: empty pairs -- skip")
            continue
        if len(pairs_df) > MAX_PAIRS:
            pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"Fold {fold_n:02d} [{trading_month}]: no CSV -- skip")
        continue

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)

    t_fold = time.time()

    # Persistence gate + HL cap (shared across both configs)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    n_base = len(pairs_df)

    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 pairs after gate -- skip")
        continue

    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 pairs after HL cap -- skip")
        continue

    n_gated = len(pairs_gated)

    # Compute formation correlation once for all gated pairs
    pairs_with_corr = compute_formation_corr(pairs_gated, f5min, form_start, form_end)
    corr_vals = pairs_with_corr["formation_corr"].dropna()
    corr_summary = (f"corr_mean={corr_vals.mean():.2f} "
                    f"corr_p25={corr_vals.quantile(0.25):.2f} "
                    f"corr_min={corr_vals.min():.2f}") if len(corr_vals) > 0 else "no_corr_data"

    row = {
        "fold":          fold_n,
        "trading_month": trading_month,
        "form_start":    form_start,
        "form_end":      form_end,
        "n_base":        n_base,
        "n_gated":       n_gated,
        "corr_mean":     float(corr_vals.mean()) if len(corr_vals) > 0 else np.nan,
        "corr_p25":      float(corr_vals.quantile(0.25)) if len(corr_vals) > 0 else np.nan,
    }

    # Run both configs
    config_results = []
    for cfg_name, threshold in CORR_CONFIGS:
        pairs_cfg = pairs_with_corr[
            pairs_with_corr["formation_corr"].notna() &
            (pairs_with_corr["formation_corr"] >= threshold)
        ].reset_index(drop=True)
        n_cfg = len(pairs_cfg)

        _, sr, nt = run_fold(pairs_cfg, t1min, f1min)
        row[f"n_{cfg_name}"]      = n_cfg
        row[f"sharpe_{cfg_name}"] = sr
        row[f"trades_{cfg_name}"] = nt
        config_results.append((cfg_name, n_cfg, sr, nt))

    elapsed = time.time() - t_fold
    sr25 = row["sharpe_CORR25"]
    sr30 = row["sharpe_CORR30"]
    n25  = row["n_CORR25"]
    n30  = row["n_CORR30"]
    flag25 = " *" if not np.isnan(sr25) and sr25 > 0 else ""
    print(
        f"Fold {fold_n:02d} [{trading_month}]  gate:{n_base}->{n_gated}  {corr_summary}  "
        f"CORR25(n={n25},SR={sr25:+.2f}{flag25})  "
        f"CORR30(n={n30},SR={sr30:+.2f})  "
        f"({elapsed:.0f}s)"
    )

    rows.append(row)

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

if not rows:
    print("No results -- exiting.")
    sys.exit(0)

results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/corr_filter_fold_metrics.csv", index=False)

def _stats(series, label):
    s = series.dropna()
    pct_pos = (s > 0).mean()
    return {
        "label":   label,
        "n_folds": len(s),
        "mean":    s.mean(),
        "median":  s.median(),
        "std":     s.std(),
        "pct_pos": pct_pos,
        "min":     s.min(),
        "max":     s.max(),
    }

# Bear/Bull split (trading month determines regime label)
bear_mask = results["trading_month"] <= "2022-12"
bull_mask = ~bear_mask

print()
print("=" * 100)
print("FORMATION CORRELATION FILTER -- 45-FOLD RESULTS")
print("Base: gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7 | TC=30bps | Borrow=50bps/yr")
print("CORR25 = primary (formation 5-min returns corr >= 0.25)")
print("CORR30 = secondary (corr >= 0.30)")
print("=" * 100)

# Per-fold table
print(f"\n{'Fold':<6} {'Month':<9} {'N_gate':>7} {'N_c25':>6} {'N_c30':>6} "
      f"{'Tr25':>5} {'Tr30':>5} {'SR_C25':>8} {'SR_C30':>8}")
print("  " + "-" * 66)
for _, r in results.iterrows():
    sr25 = r["sharpe_CORR25"]
    sr30 = r["sharpe_CORR30"]
    f25 = " *" if not np.isnan(sr25) and sr25 > 0 else ""
    f30 = " *" if not np.isnan(sr30) and sr30 > 0 else ""
    sr25_str = f"{sr25:>+8.2f}{f25}" if not np.isnan(sr25) else f"{'nan':>8}  "
    sr30_str = f"{sr30:>+8.2f}{f30}" if not np.isnan(sr30) else f"{'nan':>8}  "
    print(f"  {int(r['fold']):<6} {r['trading_month']:<9} "
          f"{int(r['n_gated']):>7} {int(r['n_CORR25']):>6} {int(r['n_CORR30']):>6} "
          f"{int(r['trades_CORR25']):>5} {int(r['trades_CORR30']):>5} "
          f"{sr25_str} {sr30_str}")

print()
print("-" * 100)

for cfg_name in ["CORR25", "CORR30"]:
    st = _stats(results[f"sharpe_{cfg_name}"], cfg_name)
    total_trades = int(results[f"trades_{cfg_name}"].sum())
    bear_sr = results.loc[bear_mask, f"sharpe_{cfg_name}"].dropna()
    bull_sr = results.loc[bull_mask, f"sharpe_{cfg_name}"].dropna()
    print(f"\n--- {cfg_name} ---")
    print(f"  Folds completed : {st['n_folds']} / {len(FOLD_SCHEDULE)}")
    print(f"  Total trades    : {total_trades}")
    print(f"  Avg pairs/fold  : {results[f'n_{cfg_name}'].mean():.1f}")
    print(f"  Mean SR         : {st['mean']:+.3f}")
    print(f"  Median SR       : {st['median']:+.3f}")
    print(f"  Std SR          : {st['std']:.3f}")
    print(f"  % Positive folds: {st['pct_pos']:.0%}")
    print(f"  Min / Max SR    : {st['min']:+.2f} / {st['max']:+.2f}")
    print(f"  Bear folds      : mean={bear_sr.mean():+.3f}  n={len(bear_sr)}  pos={(bear_sr>0).mean():.0%}")
    print(f"  Bull folds      : mean={bull_sr.mean():+.3f}  n={len(bull_sr)}  pos={(bull_sr>0).mean():.0%}")

# Comparison block
print()
print("=" * 60)
print("COMPARISON SUMMARY")
print(f"{'Metric':<25} {'CORR25':>10} {'CORR30':>10}")
print("-" * 45)
metrics_to_compare = [
    ("Mean SR",     lambda c: results[f"sharpe_{c}"].dropna().mean()),
    ("Median SR",   lambda c: results[f"sharpe_{c}"].dropna().median()),
    ("% Positive",  lambda c: (results[f"sharpe_{c}"].dropna() > 0).mean()),
    ("Folds run",   lambda c: int(results[f"sharpe_{c}"].notna().sum())),
    ("Total trades",lambda c: int(results[f"trades_{c}"].sum())),
    ("Avg pairs",   lambda c: results[f"n_{c}"].mean()),
    ("Bear mean SR",lambda c: results.loc[bear_mask, f"sharpe_{c}"].dropna().mean()),
    ("Bull mean SR",lambda c: results.loc[bull_mask, f"sharpe_{c}"].dropna().mean()),
]
for metric_name, fn in metrics_to_compare:
    v25 = fn("CORR25")
    v30 = fn("CORR30")
    if isinstance(v25, float):
        print(f"  {metric_name:<23} {v25:>+10.3f} {v30:>+10.3f}")
    else:
        print(f"  {metric_name:<23} {v25:>10} {v30:>10}")

# Also include reference points
print()
print("Reference (from prior runs):")
print("  Original pipeline (45 folds) : mean=-8.90  pos=0%")
print("  Option A baseline (no filter): mean=-0.580  pos=36%  (from run_45fold_optionA.py)")
print("  CORR25 diagnostic (14 folds) : mean=-0.38   delta=+0.84 vs baseline")
print("  STRICT sector (14 folds)     : mean=-0.46   delta=+0.76 vs baseline, trades=22 (18%)")

summary_txt = f"""
=== CORR Filter: Formation 5-min Returns Correlation Floor ===
Base: persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7
TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

CORR25 (primary -- corr >= 0.25):
  Folds: {results['sharpe_CORR25'].notna().sum()} / {len(FOLD_SCHEDULE)}
  Trades: {int(results['trades_CORR25'].sum())}
  Mean SR: {results['sharpe_CORR25'].dropna().mean():+.3f}
  Median SR: {results['sharpe_CORR25'].dropna().median():+.3f}
  % Positive: {(results['sharpe_CORR25'].dropna() > 0).mean():.0%}
  Bear: {results.loc[bear_mask, 'sharpe_CORR25'].dropna().mean():+.3f}
  Bull: {results.loc[bull_mask, 'sharpe_CORR25'].dropna().mean():+.3f}

CORR30 (secondary -- corr >= 0.30):
  Folds: {results['sharpe_CORR30'].notna().sum()} / {len(FOLD_SCHEDULE)}
  Trades: {int(results['trades_CORR30'].sum())}
  Mean SR: {results['sharpe_CORR30'].dropna().mean():+.3f}
  Median SR: {results['sharpe_CORR30'].dropna().median():+.3f}
  % Positive: {(results['sharpe_CORR30'].dropna() > 0).mean():.0%}
  Bear: {results.loc[bear_mask, 'sharpe_CORR30'].dropna().mean():+.3f}
  Bull: {results.loc[bull_mask, 'sharpe_CORR30'].dropna().mean():+.3f}

Reference:
  Option A baseline (45 folds): mean=-0.580, pos=36%
  Diagnostic delta (14 folds): CORR25=+0.84, CORR30=+0.86 vs baseline

Runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)
"""

print(summary_txt)
with open(f"{OUT_DIR}/corr_filter_summary.txt", "w") as f:
    f.write(summary_txt)

print(f"Saved: {OUT_DIR}/corr_filter_fold_metrics.csv")
print(f"Saved: {OUT_DIR}/corr_filter_summary.txt")
