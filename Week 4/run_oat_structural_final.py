"""
run_oat_structural_final.py  —  §6 Structural OAT on Option A + CORR25 Final Config

Varies structural parameters one-at-a-time around the final config defaults:
  Default: Z_entry=3.0, eos_flatten=False, stop_loss=None, CORR25>=0.25,
           persistence gate, HL<=6d, delta=1e-7, TC=30bps/side

Parameters swept:
  Z_entry     : [2.5, 2.75, 3.0*, 3.25, 3.5]
  stop_loss   : [None*, -2.5%, -5.0%]  (post-hoc on Z=3.0 default run)

NOTE: max_holding variations omitted — the engine has no separate max-hold cap;
"EOS" vs "no-EOS" is already baked into the final config as eos_flatten=False,
and adding a day-cap requires engine changes not in scope.

Runs on all 23 completed folds (same fold set as run_final_pipeline.py).

Outputs:
  results/metrics/final/phase4/oat_structural.csv
  Appends structural rows to results/metrics/final/phase4/oat_sensitivity.csv
"""

import sys, os, logging, time, traceback
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("oat_structural")

from src.phase2_execution.kalman import warmup_kalman
print("Warming up Numba JIT...")
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.utils.metrics import compute_sharpe, compute_max_dd, compute_cagr

# ── Constants (must match run_final_pipeline.py exactly) ───────────────────
PHASE1_DIR   = "results/metrics/phase1_folds"
DATA_1MIN    = "data/validated/1min_phase2"
DATA_5MIN    = "data/validated/5min_phase1"
OUT_DIR      = Path("results/metrics/final/phase4")

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
DEFAULT_Z        = 3.0
TC_BPS           = 30.0
BORROW_BPS_YR    = 50.0
FIXED_DELTA      = 1e-7
HL_MAX_DAYS      = 6.0
CORR25_THRESH    = 0.25
JOHANSEN_PVAL    = 0.05
MAX_PAIRS        = 500
PER_PAIR_DOLLAR  = TOTAL_CAPITAL / N_OPEN_PAIRS_MAX

# OAT sweep values
# Z=3.75 and Z=4.0 only — extending past Z=3.5 to confirm where the post-fix optimum lies.
# (Z in {2.5, 2.75, 3.0, 3.25, 3.5} already measured in the prior run.)
Z_SWEEP  = [3.75, 4.0]
SL_SWEEP = []                       # no SL re-test; SL is only meaningful at default Z=3.0
SL_K     = 5                        # rolling window for stop-loss check

# ── Fold schedule — completed folds only ──────────────────────────────────
COMPLETED_FOLDS = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
]


# ── Data helpers ───────────────────────────────────────────────────────────

def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict:
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


def _slice_1min(cache, tickers, trading_month):
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    return {
        tk: df[(df.index.year == year) & (df.index.month == month)]
        for tk in tickers if tk in cache
        if len(df := cache[tk][(cache[tk].index.year == year) &
                               (cache[tk].index.month == month)]) > 0
    }


def _slice_range(cache, tickers, start, end):
    return {
        tk: sl for tk in tickers if tk in cache
        for sl in [cache[tk][(cache[tk].index >= start) &
                              (cache[tk].index <= end)]]
        if len(sl) > 0
    }


# ── Filter chain (identical to run_final_pipeline.py) ─────────────────────

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
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_hl_cap(pairs_df):
    return pairs_df[pairs_df["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)


def apply_corr25(pairs_df, form_5min, form_start, form_end):
    if pairs_df.empty:
        return pd.DataFrame()
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        ra = aln["log_close_a"].diff()
        rb = aln["log_close_b"].diff()
        valid = ra.notna() & rb.notna()
        if valid.sum() < 20:
            continue
        if float(ra[valid].corr(rb[valid])) >= CORR25_THRESH:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ── Post-hoc stop-loss ─────────────────────────────────────────────────────

def apply_stop_loss(metrics, fold_results, sl_pct):
    """Re-compute Sharpe after applying rolling K-bar stop-loss post-hoc."""
    pair_pnls = metrics.get("pair_pnls", {})
    if not pair_pnls:
        return metrics["sharpe"]

    adj_pnls = []
    for (ta, tb), bar_pnl in pair_pnls.items():
        if (ta, tb) not in fold_results:
            adj_pnls.append(bar_pnl)
            continue
        pos = fold_results[(ta, tb)]["position"].reindex(bar_pnl.index).fillna(0).astype(int)
        rolling_pct = bar_pnl.rolling(SL_K, min_periods=1).sum() / PER_PAIR_DOLLAR

        adj = bar_pnl.copy()
        in_sl = False
        prev_pos = 0
        for i in range(len(adj)):
            cur_pos = int(pos.iloc[i])
            if prev_pos != 0 and cur_pos == 0:
                in_sl = False
            if cur_pos != 0 and not in_sl and float(rolling_pct.iloc[i]) < sl_pct:
                in_sl = True
            if in_sl and cur_pos != 0:
                adj.iloc[i] = 0.0
            prev_pos = cur_pos
        adj_pnls.append(adj)

    if not adj_pnls:
        return metrics["sharpe"]

    combined = pd.concat(adj_pnls).sort_index().groupby(level=0).sum()
    daily = combined.resample("D").sum().dropna()
    return float(compute_sharpe(daily))


# ── Per-fold runner ────────────────────────────────────────────────────────

def run_fold_filtered(fold_n, form_start, form_end, trading_month,
                      cache_1min, cache_5min, entry_z):
    """Apply full filter chain + run engine. Returns (metrics, fold_results) or (None, None)."""
    try:
        pairs_df = pd.read_csv(f"{PHASE1_DIR}/fold_{fold_n:02d}.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None, None
    if len(pairs_df) == 0:
        return None, None
    if len(pairs_df) > MAX_PAIRS:
        pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_range(cache_5min, tickers, form_start, form_end)
    formation_1min = _slice_range(cache_1min, tickers, form_start, form_end)

    if not trading_1min:
        return None, None

    pairs_df = apply_persistence_gate(pairs_df, formation_5min, form_end)
    if pairs_df.empty:
        return None, None
    pairs_df = apply_hl_cap(pairs_df)
    if pairs_df.empty:
        return None, None
    pairs_df = apply_corr25(pairs_df, formation_5min, form_start, form_end)
    if pairs_df.empty:
        return None, None

    try:
        fold_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            entry_z=entry_z,
            eos_flatten=False,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        log.warning("Fold %d Z=%.2f engine FAILED: %s", fold_n, entry_z, e)
        return None, None

    if not fold_results:
        return None, None

    try:
        metrics = run_fold_pnl(
            fold_results=fold_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        log.warning("Fold %d Z=%.2f pnl FAILED: %s", fold_n, entry_z, e)
        return None, None

    return metrics, fold_results


# ── Main ───────────────────────────────────────────────────────────────────

print("Loading data caches...")
t0 = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} 1-min, {len(cache_5min)} 5-min tickers  [{time.time()-t0:.1f}s]\n")

# Store results: {z_value: {fold_n: sharpe}}
z_sharpes   = {z: {} for z in Z_SWEEP}
sl_sharpes  = {sl: {} for sl in SL_SWEEP if sl is not None}
default_metrics = {}  # fold_n -> (metrics, fold_results) for SL post-hoc

t_total = time.time()
n_folds = len(COMPLETED_FOLDS)

for i, (fold_n, form_start, form_end, trading_month) in enumerate(COMPLETED_FOLDS, 1):
    print(f"Fold {fold_n:02d} [{trading_month}]  ({i}/{n_folds})")

    for z in Z_SWEEP:
        t1 = time.time()
        metrics, fold_results = run_fold_filtered(
            fold_n, form_start, form_end, trading_month,
            cache_1min, cache_5min, entry_z=z
        )
        if metrics is not None:
            sr = metrics["sharpe"]
            z_sharpes[z][fold_n] = sr
            # Cache default-Z run for stop-loss post-hoc
            if abs(z - DEFAULT_Z) < 1e-9:
                default_metrics[fold_n] = (metrics, fold_results)
        else:
            # Use NaN so fold is excluded from aggregate (same as skip)
            z_sharpes[z][fold_n] = np.nan
        print(f"  Z={z:.2f}  SR={sr:+.3f}  [{time.time()-t1:.0f}s]"
              if metrics else f"  Z={z:.2f}  skip")

    # Apply stop-loss post-hoc to the Z=3.0 run
    if fold_n in default_metrics:
        m, fr = default_metrics[fold_n]
        for sl in SL_SWEEP:
            if sl is None:
                continue
            sl_sr = apply_stop_loss(m, fr, sl)
            sl_sharpes[sl][fold_n] = sl_sr
            print(f"  SL={sl:.1%}  SR={sl_sr:+.3f}")

print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

# ── Build output DataFrame (extension run: Z in [3.75, 4.0] only) ──────────

rows = []

# Use the prior Z=3.0 mean SR from the existing oat_structural.csv as the delta reference
default_mean = np.nan
prior_path = OUT_DIR / "oat_structural.csv"
if prior_path.exists():
    prior = pd.read_csv(prior_path)
    prior_z3 = prior[(prior["param"] == "Z_entry") & (prior["value"].astype(str) == "3.0")]
    if len(prior_z3):
        default_mean = float(prior_z3["mean_sharpe"].iloc[0])

# Z_entry variations (extension only)
for z in Z_SWEEP:
    arr = np.array([v for v in z_sharpes[z].values() if not np.isnan(v)])
    if len(arr) == 0:
        continue
    label = f"Z_entry={z:.2f}"
    rows.append({
        "param":          "Z_entry",
        "value":          str(z),
        "label":          label,
        "mean_sharpe":    float(np.mean(arr)),
        "median_sharpe":  float(np.median(arr)),
        "pct_positive":   float(np.mean(arr > 0)),
        "n_folds":        len(arr),
        "mode":           "structural",
    })

structural_df = pd.DataFrame(rows)

# Write to a separate extension file, do NOT overwrite the prior results
out_structural = OUT_DIR / "oat_structural_extended.csv"
structural_df.to_csv(out_structural, index=False)
print(f"\nSaved: {out_structural}")

# Append (do not replace) to oat_sensitivity.csv as new structural rows
oat_combined_path = OUT_DIR / "oat_sensitivity.csv"
if oat_combined_path.exists():
    existing = pd.read_csv(oat_combined_path)
    # Drop any prior rows for the new Z values only (3.75, 4.0) to avoid duplication
    drop_mask = (existing["param"] == "Z_entry") & (existing["value"].astype(str).isin([str(z) for z in Z_SWEEP]))
    existing_clean = existing[~drop_mask]
    combined = pd.concat([existing_clean, structural_df], ignore_index=True)
else:
    combined = structural_df
combined.to_csv(oat_combined_path, index=False)
print(f"Updated: {oat_combined_path}")

print(f"\n=== §6 Structural OAT EXTENSION — Option A+CORR25 (Z in {Z_SWEEP}) ===")
if not np.isnan(default_mean):
    print(f"  Reference: prior Z=3.0 mean SR = {default_mean:+.3f}\n")
else:
    print(f"  No prior Z=3.0 reference found in {prior_path}; deltas will be NaN\n")

print(f"  {'Label':<25} {'Mean SR':>8}  {'Delta':>7}  {'%Pos':>5}  {'N':>4}")
print("  " + "-" * 60)
for _, r in structural_df.iterrows():
    delta = r["mean_sharpe"] - default_mean
    print(f"  {r['label']:<25} {r['mean_sharpe']:>+8.3f}  {delta:>+7.3f}  "
          f"{r['pct_positive']:>5.0%}  {int(r['n_folds']):>4}")
