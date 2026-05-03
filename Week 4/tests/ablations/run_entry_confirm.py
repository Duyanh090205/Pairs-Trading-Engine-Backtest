"""
run_entry_confirm.py  --  Entry confirmation ablation: Options 1 and 3.

Post-processes existing engine output (zscore + beta_post already computed)
to replay positions with two additional entry conditions:

  Option 1 -- Velocity flip : enter only when |Z[t]| < |Z[t-1]|
              (spread peaked and is pulling back, not still diverging)

  Option 3 -- Beta drift    : block entry when Kalman posterior beta has
              drifted > X% from frozen Phase 1 beta_pca

Tests on same 10 diagnostic folds as previous ablations.
Compares four configs:
  A    -- Option A baseline (no confirmation)
  A+V  -- Option A + velocity only
  A+B  -- Option A + beta drift only
  A+VB -- Option A + both

No engine code is modified. Post-processing only.
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

PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"

FIXED_DELTA = 1e-7
TC_BPS      = 30.0
BORROW_BPS  = 50.0
ENTRY_Z     = 3.0
JOHANSEN_PVAL = 0.05

# Beta drift thresholds to sweep for Option 3
BETA_DRIFT_THRESHOLDS = [0.10, 0.20, 0.30]

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


# ---------------------------------------------------------------------------
# Post-processing: replay positions with entry confirmations
# ---------------------------------------------------------------------------

def apply_entry_confirmations(pair_df, beta_pca,
                               velocity_confirm=False,
                               max_beta_drift=1.0):
    """
    Replay signal from existing engine output using additional entry conditions.
    Reads zscore and beta_post — no re-execution needed.

    velocity_confirm : enter only when |Z[t]| < |Z[t-1]|
    max_beta_drift   : block entry if |beta_post - beta_pca| / |beta_pca| > threshold
                       (1.0 = no filter)
    """
    z  = pair_df["zscore"].values
    bp = pair_df["beta_post"].values
    n  = len(z)

    signal    = np.zeros(n, dtype=np.int8)
    state     = np.int8(0)
    prev_abs_z = np.nan

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            signal[i] = state
            continue

        abs_zi = abs(zi)

        if state == 0:
            can_enter = True

            # Option 1: velocity confirmation
            if velocity_confirm and not np.isnan(prev_abs_z):
                if abs_zi >= prev_abs_z:      # Z still moving away from 0
                    can_enter = False

            # Option 3: beta drift at entry
            if can_enter and max_beta_drift < 1.0:
                denom = abs(beta_pca) if abs(beta_pca) > 1e-10 else 1e-10
                drift = abs(bp[i] - beta_pca) / denom
                if drift > max_beta_drift:
                    can_enter = False

            if can_enter:
                if zi < -ENTRY_Z:
                    state = np.int8(1)
                elif zi > ENTRY_Z:
                    state = np.int8(-1)
        else:
            if (state == 1 and zi >= 0.0) or (state == -1 and zi <= 0.0):
                state = np.int8(0)

        signal[i] = state
        prev_abs_z = abs_zi

    # 1-bar execution lag (same as engine)
    position = np.zeros(n, dtype=np.int8)
    position[1:] = signal[:-1]

    out = pair_df.copy()
    out["signal"]   = signal
    out["position"] = position
    return out


def exec_res_with_confirm(exec_res, pairs_gated,
                           velocity_confirm=False, max_beta_drift=1.0):
    """Apply entry confirmations to a full fold's exec_res dict."""
    new_res = {}
    beta_map = {(r["ticker_a"], r["ticker_b"]): r["beta_pca"]
                for _, r in pairs_gated.iterrows()}
    for (ta, tb), pair_df in exec_res.items():
        bp = beta_map.get((ta, tb), None)
        if bp is None:
            new_res[(ta, tb)] = pair_df
            continue
        new_res[(ta, tb)] = apply_entry_confirmations(
            pair_df, float(bp),
            velocity_confirm=velocity_confirm,
            max_beta_drift=max_beta_drift,
        )
    return new_res


def sharpe_from_exec(exec_res, t1min, pairs_gated):
    if not exec_res:
        return float("nan"), 0
    m = run_fold_pnl(
        exec_res, t1min, pairs_gated, FIXED_DELTA, {},
        total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
        tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
    )
    n_trades = int(m.get("n_trades", 0))
    return float(m.get("sharpe", float("nan"))), n_trades


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")


# ---------------------------------------------------------------------------
# Main loop
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
    if pairs_df.empty:
        continue

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)

    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        continue

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} -- {len(pairs_gated)} gated pairs", flush=True)

    # Run engine ONCE -- all confirmations post-process this output
    t = time.time()
    exec_res = run_fold_execution(
        pairs_df=pairs_gated, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        continue

    # A -- baseline (no confirmation)
    s_A,  nt_A  = sharpe_from_exec(exec_res, t1min, pairs_gated)

    # A+V -- velocity only
    er_V = exec_res_with_confirm(exec_res, pairs_gated, velocity_confirm=True, max_beta_drift=1.0)
    s_V,  nt_V  = sharpe_from_exec(er_V, t1min, pairs_gated)

    # A+B -- beta drift sweep (best threshold chosen later)
    beta_results = {}
    for bdt in BETA_DRIFT_THRESHOLDS:
        er_B = exec_res_with_confirm(exec_res, pairs_gated, velocity_confirm=False, max_beta_drift=bdt)
        s_B, nt_B = sharpe_from_exec(er_B, t1min, pairs_gated)
        beta_results[bdt] = (s_B, nt_B)

    # A+VB -- both (using 0.20 for first look; show all in table)
    er_VB = exec_res_with_confirm(exec_res, pairs_gated, velocity_confirm=True, max_beta_drift=0.20)
    s_VB, nt_VB = sharpe_from_exec(er_VB, t1min, pairs_gated)

    elapsed = time.time() - t

    print(f"  A  (baseline)        : SR={s_A:+.2f}  trades={nt_A}")
    print(f"  A+V (velocity)       : SR={s_V:+.2f}  trades={nt_V}  delta={s_V-s_A:+.2f}")
    for bdt, (s_B, nt_B) in beta_results.items():
        print(f"  A+B (beta<={bdt:.0%})     : SR={s_B:+.2f}  trades={nt_B}  delta={s_B-s_A:+.2f}")
    print(f"  A+VB (both, b<=20%)  : SR={s_VB:+.2f}  trades={nt_VB}  delta={s_VB-s_A:+.2f}  ({elapsed:.0f}s)")
    print()

    row = {
        "fold": fold_n, "month": trading_month, "regime": regime,
        "n_gated": len(pairs_gated),
        "sr_A":  s_A,  "nt_A":  nt_A,
        "sr_V":  s_V,  "nt_V":  nt_V,
        "sr_VB": s_VB, "nt_VB": nt_VB,
    }
    for bdt, (s_B, nt_B) in beta_results.items():
        row[f"sr_B{int(bdt*100)}"] = s_B
        row[f"nt_B{int(bdt*100)}"] = nt_B
    rows.append(row)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

df = pd.DataFrame(rows)

def col_mean(col):
    v = df[col].dropna()
    return v.mean() if len(v) else float("nan")

def col_pos(col):
    v = df[col].dropna()
    return (v > 0).mean() if len(v) else float("nan")

print("=" * 100)
print("SUMMARY -- Entry Confirmation Ablation (10 folds, Option A config)")
print("=" * 100)
print(f"\n{'Config':<25} {'Mean SR':>9} {'Median SR':>10} {'% Pos':>7} {'Total Trades':>13} {'Delta vs A':>11}")
print("-" * 75)

base_mean = col_mean("sr_A")
configs = [
    ("A  (baseline)",       "sr_A",   "nt_A"),
    ("A+V (velocity)",      "sr_V",   "nt_V"),
    ("A+B (beta<=10%)",     "sr_B10", "nt_B10"),
    ("A+B (beta<=20%)",     "sr_B20", "nt_B20"),
    ("A+B (beta<=30%)",     "sr_B30", "nt_B30"),
    ("A+VB (both, b<=20%)", "sr_VB",  "nt_VB"),
]
for label, scol, ntcol in configs:
    m  = col_mean(scol)
    md = df[scol].dropna().median() if scol in df else float("nan")
    pp = col_pos(scol)
    nt = int(df[ntcol].sum()) if ntcol in df else 0
    d  = m - base_mean
    flag = " <--" if scol != "sr_A" and m > base_mean else ""
    print(f"  {label:<23} {m:>+9.2f} {md:>+10.2f} {pp:>6.0%} {nt:>13} {d:>+11.2f}{flag}")

print("\nPer-fold breakdown:")
print(f"{'Fold':<6} {'Mo':<9} {'Reg':<5}  {'A':>7} {'A+V':>7} {'A+B10':>7} {'A+B20':>7} {'A+B30':>7} {'A+VB':>7}")
print("-" * 70)
for _, r in df.iterrows():
    def f(v): return f"{v:+.2f}" if not np.isnan(v) else "  n/a"
    print(f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5}  "
          f"{f(r['sr_A']):>7} {f(r['sr_V']):>7} "
          f"{f(r.get('sr_B10', float('nan'))):>7} "
          f"{f(r.get('sr_B20', float('nan'))):>7} "
          f"{f(r.get('sr_B30', float('nan'))):>7} "
          f"{f(r['sr_VB']):>7}")
print("-" * 70)
print(f"{'MEAN':<6} {'':9} {'':5}  "
      f"{col_mean('sr_A'):>+7.2f} {col_mean('sr_V'):>+7.2f} "
      f"{col_mean('sr_B10'):>+7.2f} {col_mean('sr_B20'):>+7.2f} "
      f"{col_mean('sr_B30'):>+7.2f} {col_mean('sr_VB'):>+7.2f}")

print("\nPer regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    def rm(col): return sub[col].dropna().mean() if col in sub else float("nan")
    print(f"  {reg}: A={rm('sr_A'):+.2f}  A+V={rm('sr_V'):+.2f}  "
          f"A+B20={rm('sr_B20'):+.2f}  A+VB={rm('sr_VB'):+.2f}")

print()
print("Trade count comparison:")
print(f"  A  baseline : {int(df['nt_A'].sum())} trades")
print(f"  A+V velocity: {int(df['nt_V'].sum())} trades  ({int(df['nt_V'].sum())/max(int(df['nt_A'].sum()),1):.0%} of baseline)")
print(f"  A+B beta20% : {int(df['nt_B20'].sum())} trades  ({int(df['nt_B20'].sum())/max(int(df['nt_A'].sum()),1):.0%} of baseline)")
print(f"  A+VB both   : {int(df['nt_VB'].sum())} trades  ({int(df['nt_VB'].sum())/max(int(df['nt_A'].sum()),1):.0%} of baseline)")
