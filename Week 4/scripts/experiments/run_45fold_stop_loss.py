"""
run_45fold_stop_loss.py  --  Full 45-fold stop loss comparison.

Base config: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7
             TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Stop loss applied post-hoc on bar-level PnL per trade:
  BASELINE  -- natural zero-cross exit (no stop loss)
  SL2pct    -- exit trade when cumulative PnL < -$400  (2% of $20K per-pair budget)
  SL3pct    -- exit trade when cumulative PnL < -$600  (3% of $20K per-pair budget)

TC at early SL exit approximated as 30bps * $20K = $60 (one-way exit leg).

Key question: does SL meaningfully improve 45-fold mean Sharpe, or only helps
the worst late-2025 folds while hurting good reverting folds?

Outputs:
  results/metrics/stop_loss_fold_metrics.csv
  results/metrics/stop_loss_summary.txt
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
from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics
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
PER_PAIR_DOL  = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX   # $20,000
TC_APPROX     = (TC_BPS / 10_000) * PER_PAIR_DOL     # ~$60 per-leg exit TC

# (name, stop_loss_dollars)  -- None means no stop loss
SL_CONFIGS = [
    ("BASELINE", None),
    ("SL2pct",  -400.0),
    ("SL3pct",  -600.0),
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


# ---------------------------------------------------------------------------
# Pair filters (same as other scripts)
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

def compute_formation_corr(pairs_df, form_5min, form_start, form_end):
    """Adds 'formation_corr' column: full formation-window 5-min log-return correlation."""
    corrs = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
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


# ---------------------------------------------------------------------------
# Stop loss application
# ---------------------------------------------------------------------------

def apply_stop_loss_to_pair(bar_pnl_series, trade_log, sl_dollars):
    """
    Apply stop loss to a pair's bar_pnl Series post-hoc.

    For each trade: find first bar where cumulative trade PnL <= sl_dollars.
    If found and it's before natural exit: zero out bars after it and
    charge TC at the SL exit bar.

    Returns a modified copy of bar_pnl_series.
    sl_dollars: negative float (e.g. -400.0); None means no-op.
    """
    if sl_dollars is None:
        return bar_pnl_series

    result = bar_pnl_series.values.copy()
    idx    = bar_pnl_series.index

    for trade in trade_log:
        entry_ts = pd.Timestamp(trade["entry_ts"])
        exit_ts  = pd.Timestamp(trade["exit_ts"])

        try:
            entry_pos = idx.get_loc(entry_ts)
            exit_pos  = idx.get_loc(exit_ts)
        except KeyError:
            continue

        if exit_pos <= entry_pos:
            continue

        trade_bar_pnl = result[entry_pos: exit_pos + 1]
        cum_pnl = np.cumsum(trade_bar_pnl)

        # Find first bar where cumulative PnL hits stop loss
        sl_bar = None
        for j, c in enumerate(cum_pnl):
            if c <= sl_dollars:
                sl_bar = j
                break

        # Only act if SL triggers before the natural exit
        if sl_bar is not None and sl_bar < len(trade_bar_pnl) - 1:
            # Zero out all bars after the SL exit bar (removes natural exit TC + remaining gross)
            result[entry_pos + sl_bar + 1: exit_pos + 1] = 0.0
            # Charge exit TC at the SL exit bar
            result[entry_pos + sl_bar] -= TC_APPROX

    return pd.Series(result, index=idx)


def _sum_series(series_list):
    """Sum bar_pnl series across pairs on their union index (fill 0 for missing bars)."""
    if not series_list:
        return pd.Series(dtype=float)
    combined = pd.concat(series_list, axis=1).fillna(0.0)
    return combined.sum(axis=1)


# ---------------------------------------------------------------------------
# Per-fold runner: engine + PnL + stop loss per config
# ---------------------------------------------------------------------------

def run_fold_with_sl(pairs_df, t1min, f1min):
    """
    Run engine once, then compute fold metrics for all SL configs.

    Returns: dict of {config_name: {"sharpe": ..., "n_trades": ..., "n_sl": ...}}
    """
    exec_res = run_fold_execution(
        pairs_df     = pairs_df,
        trading_1min = t1min,
        delta        = FIXED_DELTA,
        eos_flatten  = False,
        formation_ref= f1min,
        entry_z      = ENTRY_Z,
    )
    if not exec_res:
        return None, 0

    # Collect bar_pnl + trade_log per pair (shared across all configs)
    pair_results = {}
    for (ta, tb), pair_df in exec_res.items():
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))
        if ca.empty or cb.empty:
            continue
        res = compute_pair_pnl(
            pair_df, ca, cb,
            per_pair_dollar = PER_PAIR_DOL,
            tc_bps          = TC_BPS,
            borrow_bps_yr   = BORROW_BPS,
        )
        pair_results[(ta, tb)] = res

    if not pair_results:
        return None, 0

    # Count total trades (same for all configs at baseline)
    all_trades = sum(len(r["trade_log"]) for r in pair_results.values())

    config_metrics = {}
    for cfg_name, sl_dollars in SL_CONFIGS:
        modified_series = []
        all_trade_log   = []
        n_sl_exits      = 0

        for (ta, tb), res in pair_results.items():
            trade_log     = res["trade_log"]
            bar_pnl_orig  = res["bar_pnl"]
            bar_pnl_mod   = apply_stop_loss_to_pair(bar_pnl_orig, trade_log, sl_dollars)
            modified_series.append(bar_pnl_mod)
            all_trade_log.extend(trade_log)

            # Count how many trades were stopped out
            if sl_dollars is not None:
                for trade in trade_log:
                    entry_ts = pd.Timestamp(trade["entry_ts"])
                    exit_ts  = pd.Timestamp(trade["exit_ts"])
                    try:
                        ep = bar_pnl_orig.index.get_loc(entry_ts)
                        xp = bar_pnl_orig.index.get_loc(exit_ts)
                    except KeyError:
                        continue
                    if xp <= ep:
                        continue
                    cum = np.cumsum(bar_pnl_orig.values[ep: xp + 1])
                    if any(c <= sl_dollars for c in cum[:-1]):
                        n_sl_exits += 1

        total_bar_pnl = _sum_series(modified_series)
        m = compute_metrics(total_bar_pnl, _TOTAL_CAPITAL, all_trade_log)

        config_metrics[cfg_name] = {
            "sharpe":   float(m.get("sharpe",   float("nan"))),
            "max_dd":   float(m.get("max_dd",   float("nan"))),
            "cagr":     float(m.get("cagr",     float("nan"))),
            "calmar":   float(m.get("calmar",   float("nan"))),
            "win_rate": float(m.get("win_rate", float("nan"))),
            "n_trades": int(m.get("n_trades",   0)),
            "n_sl":     n_sl_exits,
        }

    return config_metrics, all_trades


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

    # Persistence gate
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    n_base = len(pairs_df)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after gate -- skip")
        continue

    # HL cap
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after HL cap -- skip")
        continue

    # CORR25 filter
    pairs_with_corr = compute_formation_corr(pairs_gated, f5min, form_start, form_end)
    pairs_filtered  = pairs_with_corr[
        pairs_with_corr["formation_corr"].notna() &
        (pairs_with_corr["formation_corr"] >= CORR25_THRESH)
    ].reset_index(drop=True)
    n_corr = len(pairs_filtered)

    if pairs_filtered.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 after CORR25 -- skip")
        continue

    # Run engine + all SL configs
    config_metrics, n_total_trades = run_fold_with_sl(pairs_filtered, t1min, f1min)
    if config_metrics is None:
        print(f"Fold {fold_n:02d} [{trading_month}]: 0 exec pairs -- skip")
        continue

    elapsed = time.time() - t_fold

    # Print one-line summary
    b_sr = config_metrics["BASELINE"]["sharpe"]
    s2_sr = config_metrics["SL2pct"]["sharpe"]
    s3_sr = config_metrics["SL3pct"]["sharpe"]
    n_sl2 = config_metrics["SL2pct"]["n_sl"]
    n_sl3 = config_metrics["SL3pct"]["n_sl"]
    flag  = " *" if not np.isnan(b_sr) and b_sr > 0 else ""
    print(f"Fold {fold_n:02d} [{trading_month}]  n={n_corr}  trades={n_total_trades}  "
          f"BASE:{b_sr:+.2f}{flag}  SL2:{s2_sr:+.2f}(sl={n_sl2})  SL3:{s3_sr:+.2f}(sl={n_sl3})  "
          f"({elapsed:.0f}s)")

    row = {
        "fold":          fold_n,
        "trading_month": trading_month,
        "n_base":        n_base,
        "n_corr":        n_corr,
        "n_trades":      n_total_trades,
    }
    for cfg, cm in config_metrics.items():
        row[f"sharpe_{cfg}"]   = cm["sharpe"]
        row[f"max_dd_{cfg}"]   = cm["max_dd"]
        row[f"cagr_{cfg}"]     = cm["cagr"]
        row[f"n_sl_{cfg}"]     = cm["n_sl"]
        row[f"win_rate_{cfg}"] = cm["win_rate"]
    rows.append(row)

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


# ---------------------------------------------------------------------------
# Aggregate and report
# ---------------------------------------------------------------------------

if not rows:
    print("No results.")
    sys.exit(0)

results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/stop_loss_fold_metrics.csv", index=False)


def _agg(col):
    v = results[col].dropna()
    pct_pos = (v > 0).mean()
    bear = results.loc[results["trading_month"] <= "2022-12", col].dropna()
    bull = results.loc[results["trading_month"] >  "2022-12", col].dropna()
    return {
        "mean": v.mean(), "median": v.median(), "std": v.std(),
        "pct_pos": pct_pos, "min": v.min(), "max": v.max(),
        "bear_mean": bear.mean(), "bull_mean": bull.mean(),
        "n": len(v),
    }


lines = [
    "=== 45-Fold Stop Loss Comparison ===",
    "Base config: CORR25 + persistence gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7",
    f"TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M",
    f"SL2pct=-$400 per trade, SL3pct=-$600 per trade, TC_exit_approx=${TC_APPROX:.0f}",
    "",
]

total_sl2 = int(results["n_sl_SL2pct"].sum())
total_sl3 = int(results["n_sl_SL3pct"].sum())
total_trades = int(results["n_trades"].sum())
lines.append(f"Folds completed : {len(results)} / {len(FOLD_SCHEDULE)}")
lines.append(f"Total trades    : {total_trades}")
lines.append(f"SL2pct stops    : {total_sl2} ({total_sl2/max(total_trades,1):.0%} of trades)")
lines.append(f"SL3pct stops    : {total_sl3} ({total_sl3/max(total_trades,1):.0%} of trades)")
lines.append("")

for cfg, sl_d in SL_CONFIGS:
    a = _agg(f"sharpe_{cfg}")
    sl_label = "no SL" if sl_d is None else f"SL=-${abs(sl_d):.0f}/trade"
    lines.append(f"--- {cfg} ({sl_label}) ---")
    lines.append(f"  Folds  : {a['n']}")
    lines.append(f"  Mean SR: {a['mean']:+.3f}   Median: {a['median']:+.3f}   Std: {a['std']:.3f}")
    lines.append(f"  % Pos  : {a['pct_pos']:.0%}   Min: {a['min']:+.2f}   Max: {a['max']:+.2f}")
    lines.append(f"  Bear   : {a['bear_mean']:+.3f}   Bull: {a['bull_mean']:+.3f}")
    lines.append("")

lines.append("--- Delta vs BASELINE ---")
for cfg, sl_d in SL_CONFIGS:
    if sl_d is None:
        continue
    a_base = _agg("sharpe_BASELINE")
    a_cfg  = _agg(f"sharpe_{cfg}")
    delta_mean   = a_cfg["mean"] - a_base["mean"]
    delta_pctpos = a_cfg["pct_pos"] - a_base["pct_pos"]
    delta_bull   = a_cfg["bull_mean"] - a_base["bull_mean"]
    delta_bear   = a_cfg["bear_mean"] - a_base["bear_mean"]
    lines.append(f"  {cfg}: mean SR delta={delta_mean:+.3f}  "
                 f"%pos delta={delta_pctpos:+.0%}  "
                 f"Bull delta={delta_bull:+.3f}  Bear delta={delta_bear:+.3f}")

lines.append("")
lines.append(f"Runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

summary = "\n".join(lines)
print("\n" + summary)

# Per-fold table
header = f"  {'Fold':<6} {'Month':<9} {'Trades':>7} {'BASE':>8} {'SL2':>8} {'SL3':>8} {'nSL2':>6} {'nSL3':>6}"
print("\nPer-fold Sharpe:")
print(header)
print("  " + "-" * 66)
for _, r in results.iterrows():
    b  = r["sharpe_BASELINE"]
    s2 = r["sharpe_SL2pct"]
    s3 = r["sharpe_SL3pct"]
    fl = " *" if b > 0 else ""
    print(f"  {int(r['fold']):<6} {r['trading_month']:<9} "
          f"{int(r['n_trades']):>7} {b:>+8.2f}{fl} {s2:>+8.2f} {s3:>+8.2f} "
          f"{int(r['n_sl_SL2pct']):>6} {int(r['n_sl_SL3pct']):>6}")

with open(f"{OUT_DIR}/stop_loss_summary.txt", "w") as f:
    f.write(summary)

print(f"\nSaved: {OUT_DIR}/stop_loss_fold_metrics.csv")
print(f"Saved: {OUT_DIR}/stop_loss_summary.txt")
