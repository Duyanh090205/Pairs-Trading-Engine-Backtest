"""
run_exit_strategy_test.py  --  Max hold vs stop loss deep comparison.

For Folds 38, 39, 40 (CORR25 filtered), simulate multiple exit strategies
on the actual bar-level PnL of every trade — both losing AND winning trades.

Why this matters:
  HL <= 6d -> expected reversion from Z=3.0 is 15.5 days (6d * 2.585)
  A 3-day max hold exits at 20% of the expected window.
  A stop loss exits based on actual loss, not calendar time.
  Both risk cutting winning trades that would have reverted if held longer.

Strategies simulated:
  BASELINE  -- actual exit (zero-cross), what we currently have
  MH3d      -- force exit after 1170 bars (~3 trading days)
  MH5d      -- force exit after 1950 bars (~5 trading days)
  SL1pct    -- exit when cumulative trade PnL < -$200  (1% of $20K)
  SL2pct    -- exit when cumulative trade PnL < -$400  (2% of $20K)
  SL3pct    -- exit when cumulative trade PnL < -$600  (3% of $20K)
  SL5pct    -- exit when cumulative trade PnL < -$1000 (5% of $20K)

For each strategy, reports:
  - PnL on losing trades (how much loss prevented)
  - PnL on winning trades (how much profit sacrificed)
  - Net change vs baseline
  - Trades killed that would have reverted

EG/GPN excluded (confirmed calculation anomaly -- $187K loss on $20K budget).
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

FIXED_DELTA    = 1e-7
TC_BPS         = 30.0
BORROW_BPS     = 50.0
ENTRY_Z        = 3.0
JOHANSEN_PVAL  = 0.05
HL_MAX_DAYS    = 6.0
CORR25_THRESH  = 0.25
PER_PAIR_DOLLAR = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX   # $20,000

AUDIT_FOLDS = [
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
]

EXCLUDE_PAIRS = {("EG", "GPN")}   # confirmed calculation anomaly

# Exit strategy configs: (name, max_hold_bars, stop_loss_dollars)
STRATEGIES = [
    ("BASELINE", None,  None),
    ("MH3d",     1170,  None),
    ("MH5d",     1950,  None),
    ("SL1pct",   None,  -200),
    ("SL2pct",   None,  -400),
    ("SL3pct",   None,  -600),
    ("SL5pct",   None, -1000),
    ("MH3d+SL2", 1170,  -400),   # combo: whichever triggers first
]

# ---------------------------------------------------------------------------
# Helpers
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

def formation_corr_val(ta, tb, form_5min, form_start, form_end):
    fa = form_5min.get(ta)
    fb = form_5min.get(tb)
    if fa is None or fb is None:
        return np.nan
    fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
    fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
    aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if len(aln) < 20:
        return np.nan
    ra = aln["log_close_a"].diff().dropna()
    rb = aln["log_close_b"].diff().dropna()
    al2 = ra.to_frame("ra").join(rb.to_frame("rb"), how="inner").dropna()
    return float(al2["ra"].corr(al2["rb"]))


def simulate_exit(bar_pnl_trade: np.ndarray,
                  max_hold_bars: int | None,
                  stop_loss_dollars: float | None) -> tuple[int, float]:
    """
    Given bar-level PnL for a single trade (starting at 0 at entry),
    find the bar where the chosen exit strategy would trigger.

    Returns (exit_bar_index, pnl_at_exit).
    exit_bar_index: index into bar_pnl_trade array where we exit.
                    len(bar_pnl_trade)-1 means no early exit (natural exit).
    """
    n = len(bar_pnl_trade)
    cum_pnl = np.cumsum(bar_pnl_trade)

    for i in range(n):
        # Stop loss check
        if stop_loss_dollars is not None and cum_pnl[i] <= stop_loss_dollars:
            return i, float(cum_pnl[i])
        # Max hold check
        if max_hold_bars is not None and i >= max_hold_bars:
            return i, float(cum_pnl[i])

    return n - 1, float(cum_pnl[-1])


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} 1-min  {len(cache_5min)} 5-min\n")

# ---------------------------------------------------------------------------
# Collect all trades with bar-level PnL
# ---------------------------------------------------------------------------

trade_records = []

for fold_n, form_start, form_end, trading_month in AUDIT_FOLDS:

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    pairs_df = pd.read_csv(csv)
    if len(pairs_df) > 500:
        pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min = _slice_month(cache_1min, tickers, trading_month)
    f1min = _slice(cache_1min, tickers, form_start, form_end)
    f5min = _slice(cache_5min, tickers, form_start, form_end)

    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)

    fcorrs = [formation_corr_val(r.ticker_a, r.ticker_b, f5min, form_start, form_end)
              for _, r in pairs_gated.iterrows()]
    pairs_gated["fcorr"] = fcorrs
    pairs_filtered = pairs_gated[
        pairs_gated["fcorr"].notna() & (pairs_gated["fcorr"] >= CORR25_THRESH)
    ].reset_index(drop=True)

    print(f"Fold {fold_n} [{trading_month}]: {len(pairs_filtered)} pairs after CORR25", flush=True)

    exec_res = run_fold_execution(
        pairs_df=pairs_filtered, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )

    for (ta, tb), pair_df in exec_res.items():
        if (ta, tb) in EXCLUDE_PAIRS or (tb, ta) in EXCLUDE_PAIRS:
            continue
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))
        if ca.empty or cb.empty:
            continue

        result = compute_pair_pnl(pair_df, ca, cb, PER_PAIR_DOLLAR,
                                   tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS)
        bar_pnl_series = result["bar_pnl"]
        bar_pnl_arr    = bar_pnl_series.values
        pos_arr        = pair_df["position"].values

        for trade in result["trade_log"]:
            entry_ts = pd.Timestamp(trade["entry_ts"])
            exit_ts  = pd.Timestamp(trade["exit_ts"])

            # Find entry/exit indices in bar_pnl
            try:
                entry_idx = bar_pnl_series.index.get_loc(entry_ts)
                exit_idx  = bar_pnl_series.index.get_loc(exit_ts)
            except KeyError:
                continue

            # Bar-level PnL for this trade only (from entry to natural exit)
            trade_bar_pnl = bar_pnl_arr[entry_idx: exit_idx + 1].copy()
            n_bars_natural = len(trade_bar_pnl)

            # Include TC at entry (already in bar_pnl[entry_idx])
            baseline_pnl = float(trade["net_pnl"])

            # Spread slope (for mechanism classification)
            window = pair_df.loc[entry_ts:exit_ts]
            spread_std = float(np.nanstd(pair_df["spread_prior"].values))
            s = window["spread_prior"].values
            t_arr = np.arange(len(s), dtype=float)
            valid = ~np.isnan(s)
            slope_hr = 0.0
            if valid.sum() > 5:
                sl, *_ = linregress(t_arr[valid], s[valid])
                slope_hr = sl * 60 / max(spread_std, 1e-10)

            rec = {
                "fold":          fold_n,
                "month":         trading_month,
                "pair":          f"{ta}/{tb}",
                "direction":     trade["direction"],
                "baseline_pnl":  baseline_pnl,
                "n_bars":        n_bars_natural,
                "n_days":        round(n_bars_natural / 390, 1),
                "slope_hr":      round(slope_hr, 3),
                "trade_bar_pnl": trade_bar_pnl,
            }

            # Simulate each strategy
            for strat_name, mh_bars, sl_dollars in STRATEGIES:
                exit_bar, pnl_at_exit = simulate_exit(trade_bar_pnl, mh_bars, sl_dollars)
                # Add exit TC if we're exiting early (not at natural exit)
                early_exit = exit_bar < n_bars_natural - 1
                if early_exit:
                    # Approximate: charge same TC as entry (we don't have price at that bar easily)
                    tc_exit = abs(baseline_pnl - trade["gross_pnl"]) / 2  # rough half of total TC
                    pnl_at_exit -= tc_exit
                rec[f"pnl_{strat_name}"]    = pnl_at_exit
                rec[f"early_{strat_name}"]  = early_exit
                rec[f"exitbar_{strat_name}"] = exit_bar

            trade_records.append(rec)

df = pd.DataFrame(trade_records)
df = df.drop(columns=["trade_bar_pnl"])

print(f"\nTotal trades across 3 folds (ex EG/GPN): {len(df)}")

# ---------------------------------------------------------------------------
# Print per-trade detail
# ---------------------------------------------------------------------------

strat_names = [s[0] for s in STRATEGIES]

for fold_n in [38, 39, 40]:
    fdf = df[df["fold"] == fold_n].sort_values("baseline_pnl")
    print(f"\n{'='*110}")
    print(f"FOLD {fold_n:02d} [{fdf['month'].iloc[0]}]  {len(fdf)} trades  "
          f"baseline total=${fdf['baseline_pnl'].sum():+,.0f}")
    print(f"{'='*110}")
    hdr = f"{'Pair':<15} {'Dir':>4} {'Days':>5} {'BASELINE':>9}"
    for s in strat_names[1:]:
        hdr += f" {s:>9}"
    print(hdr)
    print("-" * 110)
    for _, r in fdf.iterrows():
        line = f"{r['pair']:<15} {int(r['direction']):>+4} {r['n_days']:>5.1f} {r['baseline_pnl']:>+9,.0f}"
        for s in strat_names[1:]:
            pnl = r[f"pnl_{s}"]
            early = r[f"early_{s}"]
            flag = "^" if early else " "
            line += f" {pnl:>+8,.0f}{flag}"
        print(line)
    print("-" * 110)
    tot = f"{'TOTAL':<15} {'':>4} {'':>5} {fdf['baseline_pnl'].sum():>+9,.0f}"
    for s in strat_names[1:]:
        tot += f" {fdf[f'pnl_{s}'].sum():>+9,.0f} "
    print(tot)

# ---------------------------------------------------------------------------
# Aggregate: losing trades, winning trades, net
# ---------------------------------------------------------------------------

print(f"\n\n{'='*90}")
print("AGGREGATE COMPARISON  (^ = early exit triggered, EG/GPN excluded)")
print(f"{'='*90}")

losing = df[df["baseline_pnl"] < 0].copy()
winning = df[df["baseline_pnl"] >= 0].copy()

print(f"\nTotal trades: {len(df)}  |  Losing: {len(losing)}  |  Winning: {len(winning)}")
print(f"Baseline total PnL: ${df['baseline_pnl'].sum():+,.0f}")
print(f"  Losing trades PnL: ${losing['baseline_pnl'].sum():+,.0f}")
print(f"  Winning trades PnL: ${winning['baseline_pnl'].sum():+,.0f}")

print(f"\n{'Strategy':<12} {'LossPnL':>10} {'WinPnL':>10} {'NetPnL':>10} "
      f"{'vsBase':>8} {'EarlyL':>7} {'EarlyW':>7} {'WinsKilled':>10}")
print("-" * 85)

base_net = df["baseline_pnl"].sum()
base_loss = losing["baseline_pnl"].sum()
base_win  = winning["baseline_pnl"].sum()
print(f"{'BASELINE':<12} {base_loss:>+10,.0f} {base_win:>+10,.0f} {base_net:>+10,.0f} "
      f"{'':>8} {'':>7} {'':>7} {'':>10}")

for s_name, mh, sl in STRATEGIES[1:]:
    loss_pnl  = losing[f"pnl_{s_name}"].sum()
    win_pnl   = winning[f"pnl_{s_name}"].sum()
    net_pnl   = df[f"pnl_{s_name}"].sum()
    delta     = net_pnl - base_net
    n_early_l = losing[f"early_{s_name}"].sum()
    n_early_w = winning[f"early_{s_name}"].sum()
    wins_killed = int((winning[f"pnl_{s_name}"] < 0).sum())
    print(f"{s_name:<12} {loss_pnl:>+10,.0f} {win_pnl:>+10,.0f} {net_pnl:>+10,.0f} "
          f"{delta:>+8,.0f} {int(n_early_l):>7} {int(n_early_w):>7} {wins_killed:>10}")

# ---------------------------------------------------------------------------
# Intraday P&L trajectory analysis for selected trades
# ---------------------------------------------------------------------------

print(f"\n\n{'='*90}")
print("INTRADAY P&L TRAJECTORY -- where does loss accumulate?")
print(f"{'='*90}")
print("For each losing trade: P&L at day 1, 2, 3, 5 vs final")
print()

losing_sorted = df[df["baseline_pnl"] < 0].sort_values("baseline_pnl")
# Re-run to get bar-level PnL per trade
trade_records2 = []
for fold_n, form_start, form_end, trading_month in AUDIT_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    pairs_df = pd.read_csv(csv)
    if len(pairs_df) > 500:
        pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)
    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min = _slice_month(cache_1min, tickers, trading_month)
    f1min = _slice(cache_1min, tickers, form_start, form_end)
    f5min = _slice(cache_5min, tickers, form_start, form_end)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    fcorrs = [formation_corr_val(r.ticker_a, r.ticker_b, f5min, form_start, form_end)
              for _, r in pairs_gated.iterrows()]
    pairs_gated["fcorr"] = fcorrs
    pairs_filtered = pairs_gated[
        pairs_gated["fcorr"].notna() & (pairs_gated["fcorr"] >= CORR25_THRESH)
    ].reset_index(drop=True)
    exec_res = run_fold_execution(
        pairs_df=pairs_filtered, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    for (ta, tb), pair_df in exec_res.items():
        if (ta, tb) in EXCLUDE_PAIRS or (tb, ta) in EXCLUDE_PAIRS:
            continue
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))
        if ca.empty or cb.empty:
            continue
        result = compute_pair_pnl(pair_df, ca, cb, PER_PAIR_DOLLAR,
                                   tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS)
        bar_pnl_series = result["bar_pnl"]
        for trade in result["trade_log"]:
            if trade["net_pnl"] >= 0:
                continue
            entry_ts = pd.Timestamp(trade["entry_ts"])
            exit_ts  = pd.Timestamp(trade["exit_ts"])
            try:
                ei = bar_pnl_series.index.get_loc(entry_ts)
                xi = bar_pnl_series.index.get_loc(exit_ts)
            except KeyError:
                continue
            tbp = bar_pnl_series.values[ei: xi + 1]
            cum = np.cumsum(tbp)
            n = len(cum)
            pnl_d1 = float(cum[min(389, n-1)])
            pnl_d2 = float(cum[min(779, n-1)])
            pnl_d3 = float(cum[min(1169, n-1)])
            pnl_d5 = float(cum[min(1949, n-1)])
            pnl_fin = float(cum[-1])
            # fraction of final loss already at day3
            frac_d3 = pnl_d3 / pnl_fin if abs(pnl_fin) > 1 else np.nan
            print(f"  {fold_n} {ta}/{tb:<14} n={n:>5}b ({n/390:.1f}d)  "
                  f"D1={pnl_d1:>+7,.0f}  D2={pnl_d2:>+7,.0f}  D3={pnl_d3:>+7,.0f}  "
                  f"D5={pnl_d5:>+7,.0f}  Final={pnl_fin:>+7,.0f}  "
                  f"@D3={100*frac_d3:.0f}%" if not np.isnan(frac_d3) else
                  f"  {fold_n} {ta}/{tb:<14} n={n:>5}b  D1={pnl_d1:>+7,.0f}  D3={pnl_d3:>+7,.0f}  Final={pnl_fin:>+7,.0f}")

# ---------------------------------------------------------------------------
# Final comparison and recommendation
# ---------------------------------------------------------------------------

print(f"\n\n{'='*90}")
print("VERDICT: MAX HOLD 3d vs STOP LOSS")
print(f"{'='*90}")

mh3_delta  = df["pnl_MH3d"].sum()  - base_net
sl2_delta  = df["pnl_SL2pct"].sum() - base_net
sl3_delta  = df["pnl_SL3pct"].sum() - base_net
combo_delta= df["pnl_MH3d+SL2"].sum() - base_net

mh3_wins_killed  = int((winning["pnl_MH3d"]  < 0).sum())
sl2_wins_killed  = int((winning["pnl_SL2pct"] < 0).sum())
sl3_wins_killed  = int((winning["pnl_SL3pct"] < 0).sum())

print(f"""
MH3d  (3-day max hold):
  Net improvement vs baseline: ${mh3_delta:+,.0f}
  Winning trades killed (turned negative): {mh3_wins_killed}
  Risk: exits at calendar time, indifferent to whether loss is $10 or $1,000

SL2pct  (stop at -$400 = -2% of $20K):
  Net improvement vs baseline: ${sl2_delta:+,.0f}
  Winning trades killed: {sl2_wins_killed}
  Risk: may exit too early before spread has time to revert

SL3pct  (stop at -$600 = -3% of $20K):
  Net improvement vs baseline: ${sl3_delta:+,.0f}
  Winning trades killed: {sl3_wins_killed}
  Risk: looser, allows more recovery room

MH3d+SL2  (combo -- whichever triggers first):
  Net improvement vs baseline: ${combo_delta:+,.0f}
  Winning trades killed: {int((winning['pnl_MH3d+SL2'] < 0).sum())}
""")

# Key insight: what fraction of final loss was present at day 3?
print("Key diagnostic -- how much of each losing trade's final loss was already locked in by day 3?")
print("(If >80% already at day 3 -> MH3d exits too late to matter)")
print("(If <50% at day 3 -> most loss accrues after day 3, MH3d helps)")
