"""
run_model_invalidation_test.py  --  Model invalidation exit ablation.

Instead of a price-based stop-loss (shown to have no discriminating power),
exit a position when the STATISTICAL MODEL assumptions break down mid-trade.

Two signals, both computed bar-by-bar during an active position:

  Signal 1 -- Beta drift
    At entry: record beta_post[entry_bar].
    Each subsequent bar: if |beta_post[i] - beta_entry| / |beta_entry| > threshold,
    the Kalman hedge ratio has shifted -- the spread definition is wrong. Exit.

  Signal 2 -- Rolling mean shift (spread trending, not reverting)
    At entry: record spread_prior[entry_bar].
    Each bar: compute rolling mean of spread_prior over last N bars.
    If rolling mean has shifted > K*spread_std in the WRONG direction
    (i.e. away from zero, not toward it), OU stationarity is violated. Exit.

Configs tested (10 diagnostic folds, Option A base: gate+noEOS+Z=3.0+HL<=6d+delta=1e-7):
  A          -- baseline
  A+BD20     -- beta drift > 20%
  A+BD30     -- beta drift > 30%
  A+MS60_1   -- rolling mean (60-bar) shifts > 1.0 sigma wrong way
  A+MS60_15  -- rolling mean (60-bar) shifts > 1.5 sigma wrong way
  A+MS120_15 -- rolling mean (120-bar) shifts > 1.5 sigma wrong way
  A+COMBO    -- BD25 + MS60_1.5
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
HL_MAX_DAYS   = 6.0

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

CONFIGS = [
    # (label, beta_drift_threshold, mean_shift_window, mean_shift_sigma)
    ("A",           None,  None, None),
    ("A+BD20",      0.20,  None, None),
    ("A+BD30",      0.30,  None, None),
    ("A+MS60_1",    None,  60,   1.0),
    ("A+MS60_15",   None,  60,   1.5),
    ("A+MS120_15",  None,  120,  1.5),
    ("A+COMBO",     0.25,  60,   1.5),
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
# Model invalidation exit — pure post-processing
# ---------------------------------------------------------------------------

def apply_model_invalidation_exit(
    pair_df: pd.DataFrame,
    beta_drift_threshold: float | None,
    mean_shift_window: int | None,
    mean_shift_sigma: float | None,
) -> pd.DataFrame:
    """
    Replay signal/position from existing engine output, inserting forced exits
    when model assumptions break during an active position.

    Reads: signal, spread_prior, beta_post (all already in pair_df).
    Writes: new signal + position (1-bar lag reapplied).
    """
    orig_signal = pair_df["signal"].values.copy()
    spread      = pair_df["spread_prior"].values
    beta        = pair_df["beta_post"].values
    n           = len(orig_signal)

    # Spread std over full trading window (normaliser for mean-shift check)
    valid_spread = spread[~np.isnan(spread)]
    spread_std   = float(np.std(valid_spread)) if len(valid_spread) > 10 else 1.0
    if spread_std < 1e-10:
        spread_std = 1.0

    new_signal    = np.zeros(n, dtype=np.int8)
    state         = np.int8(0)
    beta_entry    = np.nan
    spread_entry  = np.nan

    for i in range(n):
        orig = orig_signal[i]

        if state == 0:
            # Entry: follow original signal
            if orig != 0:
                state        = np.int8(orig)
                beta_entry   = beta[i]
                spread_entry = spread[i] if not np.isnan(spread[i]) else 0.0
        else:
            # In trade: check natural exit first
            if orig == 0:
                state = np.int8(0)
            else:
                invalidated = False

                # Signal 1: Kalman beta has drifted too far from entry hedge ratio
                if beta_drift_threshold is not None:
                    denom = abs(beta_entry) if abs(beta_entry) > 1e-10 else 1e-10
                    drift = abs(beta[i] - beta_entry) / denom
                    if drift > beta_drift_threshold:
                        invalidated = True

                # Signal 2: Rolling mean of spread has shifted in wrong direction
                if not invalidated and mean_shift_window is not None and mean_shift_sigma is not None:
                    w_start     = max(0, i - mean_shift_window + 1)
                    window_vals = spread[w_start: i + 1]
                    valid       = window_vals[~np.isnan(window_vals)]
                    if len(valid) >= mean_shift_window // 2:
                        roll_mean  = float(np.mean(valid))
                        mean_shift = (roll_mean - spread_entry) / spread_std
                        # state=+1: entered low, expect spread UP -> exit if mean drifts lower
                        # state=-1: entered high, expect spread DOWN -> exit if mean drifts higher
                        if state == 1 and mean_shift < -mean_shift_sigma:
                            invalidated = True
                        elif state == -1 and mean_shift > mean_shift_sigma:
                            invalidated = True

                if invalidated:
                    state = np.int8(0)

        new_signal[i] = state

    # Reapply 1-bar execution lag
    new_position    = np.zeros(n, dtype=np.int8)
    new_position[1:] = new_signal[:-1]

    out              = pair_df.copy()
    out["signal"]    = new_signal
    out["position"]  = new_position
    return out


def run_with_invalidation(exec_res, pairs_df, t1min, beta_drift, ms_window, ms_sigma):
    """Apply model invalidation exit to all pairs, then score PnL."""
    if not exec_res:
        return float("nan"), 0

    modified = {}
    for key, pdf in exec_res.items():
        ta, tb = key
        row = pairs_df[
            (pairs_df["ticker_a"] == ta) & (pairs_df["ticker_b"] == tb)
        ]
        if row.empty:
            modified[key] = pdf
            continue
        modified[key] = apply_model_invalidation_exit(
            pdf,
            beta_drift_threshold = beta_drift,
            mean_shift_window    = ms_window,
            mean_shift_sigma     = ms_sigma,
        )

    m = run_fold_pnl(
        modified, t1min, pairs_df, FIXED_DELTA, {},
        total_capital    = _TOTAL_CAPITAL,
        n_open_pairs_max = _N_OPEN_PAIRS_MAX,
        tc_bps           = TC_BPS,
        borrow_bps_yr    = BORROW_BPS,
    )
    n_trades = sum(
        int((pdf["position"].diff().abs() > 0).sum() // 2)
        for pdf in modified.values()
    )
    return float(m.get("sharpe", float("nan"))), n_trades


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")

fold_results = []

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
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        continue

    # Run engine ONCE
    exec_res = run_fold_execution(
        pairs_df     = pairs_gated,
        trading_1min = t1min,
        delta        = FIXED_DELTA,
        eos_flatten  = False,
        formation_ref= f1min,
        entry_z      = ENTRY_Z,
    )

    n_gated = len(pairs_gated)
    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_gated} pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime}

    for label, bd, msw, mss in CONFIGS:
        sr, nt = run_with_invalidation(exec_res, pairs_gated, t1min, bd, msw, mss)
        row[f"sr_{label}"]     = sr
        row[f"trades_{label}"] = nt
        baseline_sr = row.get("sr_A", float("nan"))
        delta_str   = f"  delta={sr - baseline_sr:+.2f}" if label != "A" and not np.isnan(baseline_sr) else ""
        print(f"  {label:<14}: SR={sr:+.2f}  trades={nt}{delta_str}", flush=True)

    fold_results.append(row)
    print()

df = pd.DataFrame(fold_results)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
labels = [c[0] for c in CONFIGS]

print("=" * 90)
print("MODEL INVALIDATION EXIT  --  gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7  (10 folds)")
print("=" * 90)

header = f"{'Fold':<6} {'Mo':<9} {'Reg':<5}"
for lbl in labels:
    header += f"  {lbl:>11}"
print(header)
print("-" * 90)

for _, r in df.iterrows():
    line = f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5}"
    for lbl in labels:
        sr = r[f"sr_{lbl}"]
        line += f"  {sr:>+11.2f}" if not np.isnan(sr) else f"  {'nan':>11}"
    print(line)

print("-" * 90)

stats_rows = [
    ("MEAN", lambda s: s.mean()),
    ("MED",  lambda s: s.median()),
]
for stat_name, fn in stats_rows:
    line = f"{stat_name:<6} {'':<9} {'':<5}"
    for lbl in labels:
        val = fn(df[f"sr_{lbl}"].dropna())
        line += f"  {val:>+11.2f}"
    print(line)

line = f"{'POS%':<6} {'':<9} {'':<5}"
for lbl in labels:
    pct = (df[f"sr_{lbl}"].dropna() > 0).mean() * 100
    line += f"  {pct:>10.0f}%"
print(line)

line = f"{'TRADES':<6} {'':<9} {'':<5}"
for lbl in labels:
    tot = df[f"trades_{lbl}"].sum()
    line += f"  {int(tot):>11}"
print(line)

print()
print("Per regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    line = f"  {reg}: "
    for lbl in labels:
        val = sub[f"sr_{lbl}"].dropna().mean()
        line += f"{lbl}={val:+.2f}  "
    print(line)

print()
print("Delta vs baseline A:")
baseline_mean = df["sr_A"].dropna().mean()
for lbl in labels[1:]:
    val  = df[f"sr_{lbl}"].dropna().mean()
    tr_a = df["trades_A"].sum()
    tr   = df[f"trades_{lbl}"].sum()
    print(f"  {lbl:<14}: mean delta={val - baseline_mean:+.2f}  trades={tr} ({100*tr/tr_a:.0f}% of baseline)")
