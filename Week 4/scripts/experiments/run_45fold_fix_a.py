"""
run_45fold_fix_a.py  --  Full 45-fold run: CORR25 baseline vs CORR25 + Fix A.

Fix A: Formation Spread-Trend Filter
  Compute the static PCA spread S = log_A - alpha_pca - beta_pca * log_B
  over the LAST 30 trading days of the 6-month formation window (5-min bars).
  Fit a linear trend and normalize by the spread std over the full formation window.

  normalized_slope = slope_per_bar * N_last_bars / std(spread_full_formation)

  This gives total drift in spread std-deviation units over the 30-day look-back.
  Reject pair if |normalized_slope| > tau.

  Two tau values tested:
    tau=1.0 (primary, tighter)
    tau=1.5 (secondary, looser)

Root cause addressed:
  Factor-momentum accumulates differential factor exposure during formation,
  producing a slowly trending spread.  CORR25 is a correlation gate (level);
  Fix A adds a DIRECTION gate -- pairs where the spread was already drifting
  at formation end are the most direct DRIFT predictor in the trading window.
  Addresses the same (a)+(c) root cause as Fix D but via the spread channel
  rather than the return-correlation channel.

Integration: appended after CORR25, before engine. Same 5-min data already loaded.
Parameters: one threshold tau per variant.

Base config: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7
TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Outputs:
  results/metrics/fix_a_fold_metrics.csv
  results/metrics/fix_a_summary.txt
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import chi2, linregress
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
HL_MAX_DAYS   = 6.0
CORR25_THRESH = 0.25

# Fix A thresholds
FIXA_TAU_PRIMARY   = 1.0   # tighter: reject if |norm_slope| > 1.0
FIXA_TAU_SECONDARY = 1.5   # looser:  reject if |norm_slope| > 1.5

# Last-N-days window for slope fit (30 trading days × 78 bars/day)
FIXA_LOOKBACK_DAYS = 30
FIXA_BARS_PER_DAY  = 78
FIXA_LOOKBACK_BARS = FIXA_LOOKBACK_DAYS * FIXA_BARS_PER_DAY   # 2340


# ---------------------------------------------------------------------------
# Data helpers  (identical to run_45fold_fix_d.py)
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


# ---------------------------------------------------------------------------
# Pair filters
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
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def _block_corr(fa, fb, start, end, min_bars=20):
    fa_sl = fa[(fa.index >= start) & (fa.index <= end)][["log_close"]]
    fb_sl = fb[(fb.index >= start) & (fb.index <= end)][["log_close"]]
    aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if len(aln) < min_bars:
        return np.nan
    ra = aln["log_close_a"].diff().dropna()
    rb = aln["log_close_b"].diff().dropna()
    al2 = ra.to_frame("ra").join(rb.to_frame("rb"), how="inner").dropna()
    if len(al2) < min_bars:
        return np.nan
    return float(al2["ra"].corr(al2["rb"]))


def compute_formation_corr(pairs_df, form_5min, form_start, form_end):
    corrs = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            corrs.append(np.nan)
            continue
        corrs.append(_block_corr(fa, fb, form_start, form_end))
    out = pairs_df.copy()
    out["formation_corr"] = corrs
    return out


# ---------------------------------------------------------------------------
# Fix A: Spread-Trend Filter
# ---------------------------------------------------------------------------

def _compute_spread_slope(fa, fb, alpha, beta, form_start, form_end, min_bars=78):
    """
    Compute normalized linear slope of the static PCA spread over the last
    FIXA_LOOKBACK_BARS bars of the formation window.

    Returns:
        norm_slope : float or nan
            slope_per_bar * N_last_bars / std(spread_full_formation)
            Positive = spread trending up (A outperforming B relative to formation)
            Negative = spread trending down
    """
    fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
    fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
    aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if len(aln) < min_bars:
        return np.nan

    # Static PCA spread over full formation window
    spread_full = aln["log_close_a"].values - alpha - beta * aln["log_close_b"].values
    std_full = float(np.std(spread_full, ddof=1))
    if std_full < 1e-12:
        return np.nan

    # Last-30-day slice for slope fit
    n_last = min(FIXA_LOOKBACK_BARS, len(spread_full))
    if n_last < min_bars:
        return np.nan
    spread_last = spread_full[-n_last:]

    t_arr = np.arange(n_last, dtype=float)
    try:
        slope, _, _, _, _ = linregress(t_arr, spread_last)
    except Exception:
        return np.nan

    # Normalize: total drift over the window in std-deviation units
    norm_slope = slope * n_last / std_full
    return float(norm_slope)


def compute_spread_slopes(pairs_df, form_5min, form_start, form_end):
    """Add 'norm_slope' column to pairs_df."""
    slopes = []
    for _, row in pairs_df.iterrows():
        ta, tb    = row["ticker_a"], row["ticker_b"]
        alpha     = float(row["alpha_pca"])
        beta      = float(row["beta_pca"])
        fa, fb    = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            slopes.append(np.nan)
            continue
        slopes.append(_compute_spread_slope(fa, fb, alpha, beta, form_start, form_end))
    out = pairs_df.copy()
    out["norm_slope"] = slopes
    return out


def apply_trend_gate(pairs_df, tau):
    """
    Fix A filter: reject pair if |norm_slope| > tau.
    Pairs with NaN norm_slope are also rejected (insufficient data).
    """
    mask = (
        pairs_df["norm_slope"].notna() &
        (pairs_df["norm_slope"].abs() <= tau)
    )
    return pairs_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------

def run_fold(pairs_df, t1min, f1min):
    if pairs_df is None or pairs_df.empty:
        return None, float("nan"), 0
    exec_res = run_fold_execution(
        pairs_df      = pairs_df,
        trading_1min  = t1min,
        delta         = FIXED_DELTA,
        eos_flatten   = False,
        formation_ref = f1min,
        entry_z       = ENTRY_Z,
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

rows    = []
t_total = time.time()

for spec in FOLD_SCHEDULE:
    fold_n        = spec.fold_n
    form_start    = spec.form_start
    form_end      = spec.form_end
    trading_month = spec.trading_month

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if pairs_df.empty:
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
    n_base = len(pairs_df)

    # 1. Persistence gate
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after persistence gate -- skip")
        continue

    # 2. HL cap
    pairs_gated = pairs_gated[
        pairs_gated["half_life_days"] <= HL_MAX_DAYS
    ].reset_index(drop=True)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after HL cap -- skip")
        continue

    # 3. CORR25 filter (level gate)
    pairs_with_corr = compute_formation_corr(pairs_gated, f5min, form_start, form_end)
    pairs_c25 = pairs_with_corr[
        pairs_with_corr["formation_corr"].notna() &
        (pairs_with_corr["formation_corr"] >= CORR25_THRESH)
    ].reset_index(drop=True)
    n_c25 = len(pairs_c25)

    if pairs_c25.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after CORR25 -- skip")
        continue

    # 4. Fix A: spread-trend slope filter
    pairs_with_slopes = compute_spread_slopes(pairs_c25, f5min, form_start, form_end)

    pairs_fixa1 = apply_trend_gate(pairs_with_slopes, FIXA_TAU_PRIMARY)    # tau=1.0
    pairs_fixa2 = apply_trend_gate(pairs_with_slopes, FIXA_TAU_SECONDARY)  # tau=1.5
    n_fixa1 = len(pairs_fixa1)
    n_fixa2 = len(pairs_fixa2)

    # Diagnostics: how many rejected by each threshold
    n_valid_slopes   = int(pairs_with_slopes["norm_slope"].notna().sum())
    n_reject_fixa1   = n_c25 - n_fixa1
    n_reject_fixa2   = n_c25 - n_fixa2
    slope_abs_mean   = float(pairs_with_slopes["norm_slope"].abs().mean())

    # 5. Run engine for all configs
    _, sr_c25,   tr_c25   = run_fold(pairs_c25,   t1min, f1min)
    _, sr_fixa1, tr_fixa1 = run_fold(pairs_fixa1, t1min, f1min) if n_fixa1 > 0 else (None, float("nan"), 0)
    _, sr_fixa2, tr_fixa2 = run_fold(pairs_fixa2, t1min, f1min) if n_fixa2 > 0 else (None, float("nan"), 0)

    elapsed = time.time() - t_fold

    flag_c25  = " *" if not np.isnan(sr_c25)   and sr_c25   > 0 else ""
    flag_fa1  = " *" if not np.isnan(sr_fixa1) and sr_fixa1 > 0 else ""
    flag_fa2  = " *" if not np.isnan(sr_fixa2) and sr_fixa2 > 0 else ""

    print(f"Fold {fold_n:02d} [{trading_month}]  "
          f"gate:{n_c25}->FixA1:{n_fixa1}->FixA2:{n_fixa2}  "
          f"(rej1={n_reject_fixa1} rej2={n_reject_fixa2} |slope|_mean={slope_abs_mean:.2f})  "
          f"CORR25:{sr_c25:+.2f}{flag_c25}  "
          f"FixA1:{sr_fixa1:+.2f}{flag_fa1}  "
          f"FixA2:{sr_fixa2:+.2f}{flag_fa2}  "
          f"({elapsed:.0f}s)")

    rows.append({
        "fold":           fold_n,
        "trading_month":  trading_month,
        "n_base":         n_base,
        "n_c25":          n_c25,
        "n_fixa1":        n_fixa1,
        "n_fixa2":        n_fixa2,
        "n_reject_fixa1": n_reject_fixa1,
        "n_reject_fixa2": n_reject_fixa2,
        "slope_abs_mean": slope_abs_mean,
        "sharpe_CORR25":  sr_c25,
        "trades_CORR25":  tr_c25,
        "sharpe_FIXA1":   sr_fixa1,
        "trades_FIXA1":   tr_fixa1,
        "sharpe_FIXA2":   sr_fixa2,
        "trades_FIXA2":   tr_fixa2,
    })

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


# ---------------------------------------------------------------------------
# Aggregate and report
# ---------------------------------------------------------------------------

if not rows:
    print("No results.")
    sys.exit(0)

results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/fix_a_fold_metrics.csv", index=False)


def _agg(col):
    v = results[col].dropna()
    v = v[v != 0.0]
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "pct_pos": float("nan"),
                "bear": float("nan"), "bull": float("nan"), "n": 0}
    bear = results.loc[results["trading_month"] <= "2022-12", col].dropna().replace(0.0, float("nan")).dropna()
    bull = results.loc[results["trading_month"] >  "2022-12", col].dropna().replace(0.0, float("nan")).dropna()
    return {
        "mean":    float(v.mean()),
        "median":  float(v.median()),
        "pct_pos": float((v > 0).mean()),
        "bear":    float(bear.mean()) if len(bear) else float("nan"),
        "bull":    float(bull.mean()) if len(bull) else float("nan"),
        "n":       len(v),
    }

a_c25  = _agg("sharpe_CORR25")
a_fa1  = _agg("sharpe_FIXA1")
a_fa2  = _agg("sharpe_FIXA2")

total_tr_c25  = int(results["trades_CORR25"].sum())
total_tr_fa1  = int(results["trades_FIXA1"].sum())
total_tr_fa2  = int(results["trades_FIXA2"].sum())

lines = [
    "=== Fix A: Formation Spread-Trend Filter ===",
    f"norm_slope = slope_per_bar * N_last_bars / std(spread_full_formation)",
    f"Last-{FIXA_LOOKBACK_DAYS}d window ({FIXA_LOOKBACK_BARS} 5-min bars), static PCA spread",
    f"FixA1 tau={FIXA_TAU_PRIMARY:.1f}  (reject if |norm_slope| > {FIXA_TAU_PRIMARY:.1f})",
    f"FixA2 tau={FIXA_TAU_SECONDARY:.1f}  (reject if |norm_slope| > {FIXA_TAU_SECONDARY:.1f})",
    f"Base: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7",
    "",
    f"Folds completed   : {len(results)} / {len(FOLD_SCHEDULE)}",
    f"Trade count       : CORR25={total_tr_c25}  FixA1={total_tr_fa1}  FixA2={total_tr_fa2}",
    "",
    "--- CORR25 (baseline) ---",
    f"  Folds   : {a_c25['n']}",
    f"  Mean SR : {a_c25['mean']:+.3f}   Median: {a_c25['median']:+.3f}",
    f"  % Pos   : {a_c25['pct_pos']:.0%}",
    f"  Bear    : {a_c25['bear']:+.3f}   Bull: {a_c25['bull']:+.3f}",
    "",
    f"--- CORR25 + Fix A (tau={FIXA_TAU_PRIMARY:.1f}) ---",
    f"  Folds   : {a_fa1['n']}",
    f"  Mean SR : {a_fa1['mean']:+.3f}   Median: {a_fa1['median']:+.3f}",
    f"  % Pos   : {a_fa1['pct_pos']:.0%}",
    f"  Bear    : {a_fa1['bear']:+.3f}   Bull: {a_fa1['bull']:+.3f}",
    "",
    f"--- CORR25 + Fix A (tau={FIXA_TAU_SECONDARY:.1f}) ---",
    f"  Folds   : {a_fa2['n']}",
    f"  Mean SR : {a_fa2['mean']:+.3f}   Median: {a_fa2['median']:+.3f}",
    f"  % Pos   : {a_fa2['pct_pos']:.0%}",
    f"  Bear    : {a_fa2['bear']:+.3f}   Bull: {a_fa2['bull']:+.3f}",
    "",
    f"--- Delta (Fix A1 vs CORR25) ---",
    f"  Mean SR : {a_fa1['mean'] - a_c25['mean']:+.3f}",
    f"  % Pos   : {a_fa1['pct_pos'] - a_c25['pct_pos']:+.0%}",
    f"  Bull SR : {a_fa1['bull'] - a_c25['bull']:+.3f}",
    f"  Bear SR : {a_fa1['bear'] - a_c25['bear']:+.3f}",
    "",
    f"--- Delta (Fix A2 vs CORR25) ---",
    f"  Mean SR : {a_fa2['mean'] - a_c25['mean']:+.3f}",
    f"  % Pos   : {a_fa2['pct_pos'] - a_c25['pct_pos']:+.0%}",
    f"  Bull SR : {a_fa2['bull'] - a_c25['bull']:+.3f}",
    f"  Bear SR : {a_fa2['bear'] - a_c25['bear']:+.3f}",
    "",
    f"Runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
]

summary = "\n".join(lines)
print("\n" + summary)

# Per-fold table
print("\nPer-fold Sharpe (CORR25 vs Fix A1 vs Fix A2):")
hdr = (f"  {'Fold':<6} {'Month':<9} {'nC25':>5} {'nFA1':>5} {'nFA2':>5} "
       f"{'|slp|':>6} {'CORR25':>8} {'FixA1':>8} {'FixA2':>8} {'D1':>7} {'D2':>7}")
print(hdr)
print("  " + "-" * 82)
for _, r in results.iterrows():
    c25  = r["sharpe_CORR25"]
    fa1  = r["sharpe_FIXA1"]
    fa2  = r["sharpe_FIXA2"]
    d1   = fa1 - c25 if not (np.isnan(fa1) or np.isnan(c25)) else float("nan")
    d2   = fa2 - c25 if not (np.isnan(fa2) or np.isnan(c25)) else float("nan")
    fl1  = " *" if fa1 > 0 else ""
    fl2  = " *" if fa2 > 0 else ""
    slp  = r["slope_abs_mean"]
    print(f"  {int(r['fold']):<6} {r['trading_month']:<9} "
          f"{int(r['n_c25']):>5} {int(r['n_fixa1']):>5} {int(r['n_fixa2']):>5} "
          f"{slp:>6.2f} "
          f"{c25:>+8.2f} {fa1:>+8.2f}{fl1} {fa2:>+8.2f}{fl2} "
          f"{d1:>+7.2f} {d2:>+7.2f}")

with open(f"{OUT_DIR}/fix_a_summary.txt", "w") as f:
    f.write(summary)

print(f"\nSaved: {OUT_DIR}/fix_a_fold_metrics.csv")
print(f"Saved: {OUT_DIR}/fix_a_summary.txt")
