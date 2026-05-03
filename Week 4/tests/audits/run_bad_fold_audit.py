"""
run_bad_fold_audit.py  --  Deep audit of catastrophic folds 23, 34, 39, 40.

For each bad fold:
  1. Re-run execution (Option A + HL<=6d config)
  2. Find top losing pairs via compute_pair_pnl
  3. Classify each trade's loss mechanism:
       JUMP    -- spread moved adversely in < 30 bars (sudden shock, earnings, news)
       TREND   -- spread moved monotonically against position (structural drift)
       DRIFT   -- spread drifted gradually, recovered partially but not fully
  4. Show price trajectories of underlying stocks during the losing trade
  5. Propose counter-mechanisms based on what we find

Folds:
  23 [2024-05] -- May 2024
  34 [2025-04] -- April 2025 (Trump tariff shock period)
  39 [2025-09] -- September 2025
  40 [2025-10] -- October 2025
"""

import sys, os
import numpy as np
import pandas as pd
from scipy.stats import chi2, linregress
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()

from src.phase2_execution.engine import run_fold_execution, _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL
from src.phase3_backtest.pnl import compute_pair_pnl

PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
HL_MAX_DAYS   = 6.0
PER_PAIR_DOLLAR = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX

BAD_FOLDS = [
    (23, "2023-11-01", "2024-04-30", "2024-05", "Bull"),
    (34, "2024-10-01", "2025-03-31", "2025-04", "Bull"),
    (39, "2025-03-01", "2025-08-31", "2025-09", "Bull"),
    (40, "2025-04-01", "2025-09-30", "2025-10", "Bull"),
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
    survivors  = []
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
# Trade classifier
# ---------------------------------------------------------------------------

def classify_trade_mechanism(pair_df, trade, spread_std):
    """
    Classify the loss mechanism of a single trade.

    Returns:
      mechanism : JUMP | TREND | DRIFT | REVERT (successful)
      details   : dict with diagnostics
    """
    entry_ts = pd.Timestamp(trade["entry_ts"])
    exit_ts  = pd.Timestamp(trade["exit_ts"])
    direction = trade["direction"]   # +1 or -1

    window = pair_df.loc[entry_ts:exit_ts]
    if len(window) < 2:
        return "UNKNOWN", {}

    z = window["zscore"].values
    s = window["spread_prior"].values
    n = len(z)

    # Max adverse Z excursion (how far did it go against us)
    if direction == 1:   # entered short (Z was high), expect Z to fall
        # adverse = Z going higher still
        max_adv_z = float(np.nanmax(z)) if len(z) > 0 else np.nan
        final_z   = float(z[-1]) if len(z) > 0 else np.nan
    else:                # entered long (Z was low), expect Z to rise
        max_adv_z = float(np.nanmin(z)) if len(z) > 0 else np.nan
        final_z   = float(z[-1]) if len(z) > 0 else np.nan

    # Time to reach max adverse excursion
    valid_z = z[~np.isnan(z)]
    if direction == 1:
        peak_idx = int(np.nanargmax(z)) if len(valid_z) > 0 else 0
    else:
        peak_idx = int(np.nanargmin(z)) if len(valid_z) > 0 else 0

    # Spread trend during trade: fit linear regression
    t_arr  = np.arange(len(s), dtype=float)
    valid  = ~np.isnan(s)
    if valid.sum() > 5:
        slope, _, _, _, _ = linregress(t_arr[valid], s[valid])
        # Normalise slope: bps per hour (60 1-min bars)
        slope_per_hr = slope * 60 / max(spread_std, 1e-10)
    else:
        slope_per_hr = 0.0

    # Classification
    pct_peak_early = peak_idx / max(n - 1, 1)  # how early did we hit peak adverse
    net_pnl = trade["net_pnl"]

    details = {
        "n_bars":        n,
        "max_adv_z":     round(max_adv_z, 2),
        "final_z":       round(final_z, 2),
        "peak_bar":      peak_idx,
        "pct_peak_early": round(pct_peak_early, 2),
        "slope_per_hr":  round(slope_per_hr, 3),
        "net_pnl":       round(net_pnl, 1),
    }

    if net_pnl >= 0:
        return "REVERT", details

    # JUMP: adverse excursion reached within first 10% of holding period
    if pct_peak_early < 0.10 and n > 30:
        return "JUMP", details

    # TREND: spread has strong monotonic slope against position throughout trade
    wrong_direction = (direction == 1 and slope_per_hr > 0.3) or \
                      (direction == -1 and slope_per_hr < -0.3)
    if wrong_direction and abs(slope_per_hr) > 0.3:
        return "TREND", details

    return "DRIFT", details


# ---------------------------------------------------------------------------
# Price move analysis
# ---------------------------------------------------------------------------

def analyse_price_moves(ta, tb, t1min, entry_ts, exit_ts):
    """Show price returns of A and B during the trade window."""
    da = t1min.get(ta)
    db = t1min.get(tb)
    if da is None or db is None:
        return {}

    wa = da[(da.index >= entry_ts) & (da.index <= exit_ts)]["close"]
    wb = db[(db.index >= entry_ts) & (db.index <= exit_ts)]["close"]

    if len(wa) < 2 or len(wb) < 2:
        return {}

    ret_a = (wa.iloc[-1] / wa.iloc[0] - 1) * 100
    ret_b = (wb.iloc[-1] / wb.iloc[0] - 1) * 100

    # Max intraperiod drawdown of each
    dd_a = float((wa / wa.cummax() - 1).min() * 100)
    dd_b = float((wb / wb.cummax() - 1).min() * 100)

    # Correlation of 1-min returns during trade
    r_a = wa.pct_change().dropna()
    r_b = wb.pct_change().dropna()
    aligned = r_a.align(r_b, join="inner")
    corr = float(aligned[0].corr(aligned[1])) if len(aligned[0]) > 10 else np.nan

    return {
        "ret_a_pct":  round(float(ret_a), 2),
        "ret_b_pct":  round(float(ret_b), 2),
        "divergence": round(float(ret_a - ret_b), 2),
        "dd_a_pct":   round(dd_a, 2),
        "dd_b_pct":   round(dd_b, 2),
        "intra_corr": round(corr, 3) if not np.isnan(corr) else None,
    }


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")

all_mechanisms = []

for fold_n, form_start, form_end, trading_month, regime in BAD_FOLDS:

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(csv)
        if len(pairs_df) > 500:
            pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"Fold {fold_n:02d}: no CSV -- skip")
        continue
    if pairs_df.empty:
        continue

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min   = _slice_month(cache_1min, tickers, trading_month)
    f1min   = _slice(cache_1min, tickers, form_start, form_end)
    f5min   = _slice(cache_5min, tickers, form_start, form_end)

    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d}: 0 after gate -- skip")
        continue
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        print(f"Fold {fold_n:02d}: 0 after HL cap -- skip")
        continue

    exec_res = run_fold_execution(
        pairs_df=pairs_gated, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        print(f"Fold {fold_n:02d}: 0 exec pairs -- skip")
        continue

    print(f"\n{'='*80}")
    print(f"FOLD {fold_n:02d}  [{trading_month}]  {regime}  -- {len(pairs_gated)} gated pairs, {len(exec_res)} exec pairs")
    print(f"{'='*80}")

    # Compute per-pair PnL
    pair_summaries = []
    for (ta, tb), pair_df in exec_res.items():
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))
        if ca.empty or cb.empty:
            continue

        result    = compute_pair_pnl(pair_df, ca, cb, PER_PAIR_DOLLAR, tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS)
        net_pnl   = float(result["bar_pnl"].sum())
        trade_log = result["trade_log"]
        spread_std = float(np.nanstd(pair_df["spread_prior"].values))

        pair_summaries.append({
            "pair":       f"{ta}/{tb}",
            "ta": ta, "tb": tb,
            "net_pnl":   net_pnl,
            "n_trades":  len(trade_log),
            "open_bars": int((pair_df["position"] != 0).sum()),
            "pair_df":   pair_df,
            "trade_log": trade_log,
            "spread_std": spread_std,
        })

    if not pair_summaries:
        print("  No pairs with price data.")
        continue

    pair_summaries.sort(key=lambda x: x["net_pnl"])

    fold_total = sum(p["net_pnl"] for p in pair_summaries)
    print(f"  Total fold PnL: ${fold_total:,.0f}  across {len(pair_summaries)} pairs  ({sum(p['n_trades'] for p in pair_summaries)} trades)\n")

    # Show top 5 losers in detail
    n_show = min(5, len(pair_summaries))
    print(f"  TOP {n_show} LOSING PAIRS:")
    print(f"  {'Pair':<18} {'NetPnL':>9} {'Trades':>7} {'Bars':>6}  Mechanisms")
    print(f"  {'-'*75}")

    for p in pair_summaries[:n_show]:
        if not p["trade_log"]:
            print(f"  {p['pair']:<18} {p['net_pnl']:>+9,.0f}   (no completed trades -- position open at fold end)")
            continue

        mechs = []
        for trade in p["trade_log"]:
            mech, det = classify_trade_mechanism(p["pair_df"], trade, p["spread_std"])
            mechs.append((mech, det, trade))
            all_mechanisms.append({"fold": fold_n, "month": trading_month, "pair": p["pair"], "mechanism": mech, **det})

        mech_str = ", ".join(f"{m}(Z={d['max_adv_z']:.1f},n={d['n_bars']})" for m, d, _ in mechs)
        print(f"  {p['pair']:<18} {p['net_pnl']:>+9,.0f}  {p['n_trades']:>6}  {p['open_bars']:>5}  {mech_str}")

        # For each trade, show price moves and detail
        for mech, det, trade in mechs:
            entry_ts = pd.Timestamp(trade["entry_ts"])
            exit_ts  = pd.Timestamp(trade["exit_ts"])
            pm = analyse_price_moves(p["ta"], p["tb"], t1min, entry_ts, exit_ts)

            dir_str = "LONG A/SHORT B" if trade["direction"] == -1 else "SHORT A/LONG B"
            print(f"    [{mech}] {dir_str}  {entry_ts.strftime('%m-%d %H:%M')} -> {exit_ts.strftime('%m-%d %H:%M')}  ({det['n_bars']} bars)")
            entry_z_val = trade.get('entry_z', float('nan'))
            entry_z_str = f"{entry_z_val:.2f}" if isinstance(entry_z_val, (int, float)) and not (isinstance(entry_z_val, float) and np.isnan(entry_z_val)) else str(entry_z_val)
            print(f"      Z at entry: {entry_z_str}  MaxAdverseZ={det['max_adv_z']:.2f}  FinalZ={det['final_z']:.2f}  Slope/hr={det['slope_per_hr']:+.3f}sigma")
            if pm:
                print(f"      {p['ta']} return: {pm['ret_a_pct']:+.2f}%  {p['tb']} return: {pm['ret_b_pct']:+.2f}%  divergence: {pm['divergence']:+.2f}pp  intra-corr: {pm['intra_corr']}")
            print(f"      Net PnL: ${trade['net_pnl']:+,.0f}  (gross: ${trade['gross_pnl']:+,.0f})")

    # Show winners for context
    winners = [p for p in pair_summaries if p["net_pnl"] > 0]
    if winners:
        print(f"\n  PROFITABLE PAIRS ({len(winners)} total, shown for context):")
        for p in winners[:3]:
            mechs = []
            for trade in p["trade_log"]:
                mech, _ = classify_trade_mechanism(p["pair_df"], trade, p["spread_std"])
                mechs.append(mech)
            print(f"  {p['pair']:<18} {p['net_pnl']:>+9,.0f}  {p['n_trades']:>6} trades")


# ---------------------------------------------------------------------------
# Aggregate mechanism breakdown
# ---------------------------------------------------------------------------

if all_mechanisms:
    mech_df = pd.DataFrame(all_mechanisms)

    print(f"\n\n{'='*80}")
    print("AGGREGATE MECHANISM BREAKDOWN (all bad folds, losing trades only)")
    print(f"{'='*80}")

    losing = mech_df[mech_df["net_pnl"] < 0]
    print(f"\nLosing trades: {len(losing)} total")
    if len(losing) > 0:
        for mech, grp in losing.groupby("mechanism"):
            avg_adv  = grp["max_adv_z"].mean()
            avg_bars = grp["n_bars"].mean()
            avg_loss = grp["net_pnl"].mean()
            print(f"  {mech:<8}: {len(grp):>3} trades  avg_adv_Z={avg_adv:.2f}  avg_bars={avg_bars:.0f}  avg_loss=${avg_loss:+,.0f}")

    print(f"\nMechanism by fold:")
    for fold_n, grp in mech_df[mech_df["net_pnl"] < 0].groupby("fold"):
        counts = grp["mechanism"].value_counts().to_dict()
        print(f"  Fold {fold_n:02d} [{grp['month'].iloc[0]}]: ", end="")
        print("  ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\n{'='*80}")
    print("COUNTER-MECHANISM PROPOSALS")
    print(f"{'='*80}")

    jump_n  = len(losing[losing["mechanism"] == "JUMP"])
    trend_n = len(losing[losing["mechanism"] == "TREND"])
    drift_n = len(losing[losing["mechanism"] == "DRIFT"])
    total_l = len(losing)

    print(f"\nLoss breakdown:")
    print(f"  JUMP  (sudden shock  < 10% into hold): {jump_n:>3} trades  ({100*jump_n/max(total_l,1):.0f}%)")
    print(f"  TREND (monotonic drift against pos)  : {trend_n:>3} trades  ({100*trend_n/max(total_l,1):.0f}%)")
    print(f"  DRIFT (partial recovery, stuck)      : {drift_n:>3} trades  ({100*drift_n/max(total_l,1):.0f}%)")

    print(f"""
Proposals by mechanism:

JUMP losses -- idiosyncratic events (earnings, news, sector shocks):
  * Cannot be predicted from spread/beta signals post-entry
  * Pre-entry filters: add earnings blackout window (skip entry within N days of
    known earnings date -- requires earnings calendar data)
  * Alternatively: skip entry if either ticker's 5-day realised vol spike > 2x
    its formation-window baseline (detects pre-announcement IV inflation)
  * Impact: reduces entry count, but removes the trades most likely to gap

TREND losses -- spread structurally drifting (OU assumption broken):
  * These ARE catchable mid-trade
  * Signal: fit OLS slope to spread_prior from entry bar onward every K bars
  * If slope (in sigma/hr) exceeds threshold and has been consistent for 2+ hours
    -> exit: the spread is trending, not mean-reverting
  * The rolling mean shift signal we tested fires too late (waits for 60-bar window)
  * A faster signal: cumulative Z-score drift (sum of Z values since entry > threshold)
  * Note: this is different from what we tested -- we were checking rolling mean vs
    entry-bar level, not cumulative directional drift

DRIFT losses -- slow grind, recovery stalled:
  * Max holding cap per trade (e.g. 3 days = 1170 bars) regardless of position
  * Different from EOS flatten -- this exits intraday multi-day trades
  * At Z=3.0 + HL<=6d, expected max hold should be ~15 trading days -- but
    some pairs still grind for entire month under no-EOS
  * A per-trade max-hold cap of 5 days would cap these without killing fast trades
""")
