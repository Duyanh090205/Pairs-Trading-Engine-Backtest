"""
run_45fold_fix_d.py  --  Full 45-fold run: CORR25 baseline vs CORR25 + Fix D.

Fix D: Rolling Correlation Stability Gate
  Partition 6-month formation into 3 contiguous 2-month blocks B1, B2, B3.
  Compute rho1, rho2, rho3 (Pearson on 5-min log-returns, same estimator as CORR25).
  Reject pair if EITHER:
    (i)  rho3 < 0.15               -- final-block floor: catches end-of-formation collapse
    (ii) rho1 > rho2 > rho3
         AND rho1 - rho3 >= 0.15   -- monotonic decay: catches slow-bleed the floor misses

Root cause addressed:
  Factor-momentum accumulates differential factor exposure across formation,
  producing a monotonically decaying within-pair return correlation.
  CORR25 (full-window average 0.35-0.40) hides the 0.50->0.35->0.20 decay path.
  Fix D adds the TREND dimension to the existing LEVEL gate.

Integration: appended after CORR25, before engine. Same 5-min data already loaded.
Parameters: only 0.15 amplitude threshold, anchored to CORR25 scale. No new free param.

Base config: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7
TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Outputs:
  results/metrics/fix_d_fold_metrics.csv
  results/metrics/fix_d_summary.txt
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
HL_MAX_DAYS   = 6.0
CORR25_THRESH = 0.25

# Fix D thresholds — anchored to CORR25 scale
FIXD_FLOOR       = 0.15   # rho3 floor (final block)
FIXD_DECAY_DELTA = 0.15   # minimum amplitude for monotonic decay clause


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
    """Compute 5-min log-return correlation for a single pair in a date window."""
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
    """Full-window CORR25 computation. Adds 'formation_corr' column."""
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


def compute_block_correlations(pairs_df, form_5min, form_start, form_end):
    """
    Fix D: Compute 3-block rolling correlations.

    Partition formation into 3 equal 2-month blocks:
      B1: [form_start,       form_start+2m)
      B2: [form_start+2m,    form_start+4m)
      B3: [form_start+4m,    form_end]

    Adds columns: rho1, rho2, rho3.
    Pairs with any NaN block are flagged (insufficient data in that block).
    """
    t0   = pd.Timestamp(form_start)
    b1_end  = (t0 + pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    b2_start = (t0 + pd.DateOffset(months=2) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    b2_end  = (t0 + pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    b3_start = (t0 + pd.DateOffset(months=4) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    b3_end  = form_end

    block_bounds = [
        (form_start, b1_end),    # B1: [form_start, +2m]
        (b2_start,   b2_end),    # B2: (+2m+1d, +4m]  -- no overlap with B1
        (b3_start,   b3_end),    # B3: (+4m+1d, form_end]  -- no overlap with B2
    ]

    rho1_list, rho2_list, rho3_list = [], [], []

    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            rho1_list.append(np.nan)
            rho2_list.append(np.nan)
            rho3_list.append(np.nan)
            continue
        rhos = [_block_corr(fa, fb, bs, be) for bs, be in block_bounds]
        rho1_list.append(rhos[0])
        rho2_list.append(rhos[1])
        rho3_list.append(rhos[2])

    out = pairs_df.copy()
    out["rho1"] = rho1_list
    out["rho2"] = rho2_list
    out["rho3"] = rho3_list
    return out


def apply_stability_gate(pairs_df):
    """
    Fix D filter: reject pair if EITHER clause fires.

    Clause (i)  -- final-block floor:    rho3 < 0.15
    Clause (ii) -- monotonic decay:      rho1 > rho2 > rho3 AND rho1-rho3 >= 0.15

    Pairs with any NaN block correlation are also rejected (can't verify stability).
    Returns filtered DataFrame + two diagnostic columns: fixd_reject (bool), fixd_reason (str).
    """
    survivors = []
    for _, row in pairs_df.iterrows():
        r1 = row.get("rho1", np.nan)
        r2 = row.get("rho2", np.nan)
        r3 = row.get("rho3", np.nan)

        # Reject if any block is missing
        if any(pd.isna(x) for x in [r1, r2, r3]):
            continue

        # Clause (i): final-block floor
        if r3 < FIXD_FLOOR:
            continue

        # Clause (ii): monotonic decay with material amplitude
        if r1 > r2 > r3 and (r1 - r3) >= FIXD_DECAY_DELTA:
            continue

        survivors.append(row)

    return pd.DataFrame(survivors).reset_index(drop=True)


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

rows       = []
t_total    = time.time()

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

    # 4. Fix D: 3-block stability gate (trend gate, appended to CORR25)
    pairs_with_blocks = compute_block_correlations(pairs_c25, f5min, form_start, form_end)
    pairs_fixd = apply_stability_gate(pairs_with_blocks)
    n_fixd = len(pairs_fixd)

    # Diagnostic: count how many were rejected and why
    n_rejected_floor  = int((
        pairs_with_blocks["rho3"].notna() &
        (pairs_with_blocks["rho3"] < FIXD_FLOOR)
    ).sum())
    n_rejected_decay  = int((
        pairs_with_blocks["rho1"].notna() &
        pairs_with_blocks["rho2"].notna() &
        pairs_with_blocks["rho3"].notna() &
        (pairs_with_blocks["rho3"] >= FIXD_FLOOR) &   # didn't fail floor
        (pairs_with_blocks["rho1"] > pairs_with_blocks["rho2"]) &
        (pairs_with_blocks["rho2"] > pairs_with_blocks["rho3"]) &
        ((pairs_with_blocks["rho1"] - pairs_with_blocks["rho3"]) >= FIXD_DECAY_DELTA)
    ).sum())

    # 5. Run engine for both configs (shares the same formation data)
    _, sr_c25,  tr_c25  = run_fold(pairs_c25,  t1min, f1min)
    _, sr_fixd, tr_fixd = run_fold(pairs_fixd, t1min, f1min) if n_fixd > 0 else (None, float("nan"), 0)

    elapsed = time.time() - t_fold

    # Print block corr stats for insight
    rho3_mean = float(pairs_with_blocks["rho3"].mean()) if not pairs_with_blocks.empty else float("nan")
    flag_c25  = " *" if not np.isnan(sr_c25)  and sr_c25  > 0 else ""
    flag_fd   = " *" if not np.isnan(sr_fixd) and sr_fixd > 0 else ""

    print(f"Fold {fold_n:02d} [{trading_month}]  "
          f"gate:{n_c25}->FixD:{n_fixd}  "
          f"(floor={n_rejected_floor} decay={n_rejected_decay})  "
          f"rho3_mean={rho3_mean:+.2f}  "
          f"CORR25:{sr_c25:+.2f}{flag_c25}  "
          f"FixD:{sr_fixd:+.2f}{flag_fd}  "
          f"({elapsed:.0f}s)")

    rows.append({
        "fold":           fold_n,
        "trading_month":  trading_month,
        "n_base":         n_base,
        "n_c25":          n_c25,
        "n_fixd":         n_fixd,
        "n_rejected_floor":  n_rejected_floor,
        "n_rejected_decay":  n_rejected_decay,
        "rho3_mean":      rho3_mean,
        "sharpe_CORR25":  sr_c25,
        "trades_CORR25":  tr_c25,
        "sharpe_FIXD":    sr_fixd,
        "trades_FIXD":    tr_fixd,
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
results.to_csv(f"{OUT_DIR}/fix_d_fold_metrics.csv", index=False)


def _agg(col):
    v = results[col].dropna()
    v = v[v != 0.0]   # exclude zero-trade folds from SR stats
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
a_fixd = _agg("sharpe_FIXD")

total_rejected_floor = int(results["n_rejected_floor"].sum())
total_rejected_decay = int(results["n_rejected_decay"].sum())
total_tr_c25  = int(results["trades_CORR25"].sum())
total_tr_fixd = int(results["trades_FIXD"].sum())

lines = [
    "=== Fix D: Rolling Correlation Stability Gate ===",
    "Clause (i):  rho3 < 0.15              (final-block floor)",
    "Clause (ii): rho1 > rho2 > rho3 AND rho1-rho3 >= 0.15 (monotonic decay)",
    f"Base: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7",
    "",
    f"Folds completed   : {len(results)} / {len(FOLD_SCHEDULE)}",
    f"Pairs rejected    : {total_rejected_floor} floor  +  {total_rejected_decay} decay",
    f"Trade count       : CORR25={total_tr_c25}  FixD={total_tr_fixd}",
    "",
    "--- CORR25 (baseline) ---",
    f"  Folds   : {a_c25['n']}",
    f"  Mean SR : {a_c25['mean']:+.3f}   Median: {a_c25['median']:+.3f}",
    f"  % Pos   : {a_c25['pct_pos']:.0%}",
    f"  Bear    : {a_c25['bear']:+.3f}   Bull: {a_c25['bull']:+.3f}",
    "",
    "--- CORR25 + Fix D ---",
    f"  Folds   : {a_fixd['n']}",
    f"  Mean SR : {a_fixd['mean']:+.3f}   Median: {a_fixd['median']:+.3f}",
    f"  % Pos   : {a_fixd['pct_pos']:.0%}",
    f"  Bear    : {a_fixd['bear']:+.3f}   Bull: {a_fixd['bull']:+.3f}",
    "",
    "--- Delta (Fix D vs CORR25) ---",
    f"  Mean SR : {a_fixd['mean'] - a_c25['mean']:+.3f}",
    f"  % Pos   : {a_fixd['pct_pos'] - a_c25['pct_pos']:+.0%}",
    f"  Bull SR : {a_fixd['bull'] - a_c25['bull']:+.3f}",
    f"  Bear SR : {a_fixd['bear'] - a_c25['bear']:+.3f}",
    "",
    f"Runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
]

summary = "\n".join(lines)
print("\n" + summary)

# Per-fold table
print("\nPer-fold Sharpe (CORR25 vs Fix D):")
hdr = f"  {'Fold':<6} {'Month':<9} {'nC25':>5} {'nFD':>5} {'Floor':>6} {'Decay':>6} {'rho3':>6} {'CORR25':>8} {'FixD':>8} {'Delta':>8}"
print(hdr)
print("  " + "-" * 78)
for _, r in results.iterrows():
    c25  = r["sharpe_CORR25"]
    fd   = r["sharpe_FIXD"]
    dlt  = fd - c25 if not (np.isnan(fd) or np.isnan(c25)) else float("nan")
    fl   = " *" if fd > 0 else ""
    print(f"  {int(r['fold']):<6} {r['trading_month']:<9} "
          f"{int(r['n_c25']):>5} {int(r['n_fixd']):>5} "
          f"{int(r['n_rejected_floor']):>6} {int(r['n_rejected_decay']):>6} "
          f"{r['rho3_mean']:>+6.2f} "
          f"{c25:>+8.2f} {fd:>+8.2f}{fl} {dlt:>+8.2f}")

with open(f"{OUT_DIR}/fix_d_summary.txt", "w") as f:
    f.write(summary)

print(f"\nSaved: {OUT_DIR}/fix_d_fold_metrics.csv")
print(f"Saved: {OUT_DIR}/fix_d_summary.txt")
