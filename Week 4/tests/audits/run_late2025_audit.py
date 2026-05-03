"""
run_late2025_audit.py  --  Deep audit of Folds 38, 39, 40 (late-2025 losses).

These folds lose heavily (SR -4.77, -4.31, -4.98) despite passing the CORR25
formation correlation filter (mean corr 0.31-0.39).  Four hypotheses tested:

  H1  Loss concentrated by ticker  -> ticker exposure cap
  H2  Correlation collapses during trade  -> live correlation decay exit
  H3  Loss from long drift (direction bias + long hold)  -> 3-day max hold
  H4  None of the above  -> regime limitation, stop optimizing

Verdict printed at the end: which hypothesis is supported, or none.
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
PER_PAIR_DOLLAR = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX

AUDIT_FOLDS = [
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
]

# Thresholds for verdict
TICKER_CONC_THRESH  = 0.40   # >40% of total loss from 1-2 tickers -> concentrated
CORR_DROP_THRESH    = 0.20   # avg (formation_corr - intra_corr) > 0.20 -> collapse
CORR_COLLAPSE_PCT   = 0.50   # >50% of trades have intra_corr < 0.15 -> collapse
DIRECTION_BIAS      = 0.70   # >70% of losses one direction -> directional bias
MAX_HOLD_BARS_3D    = 1170   # 3 trading days * 390 bars

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

def formation_corr(ta, tb, form_5min, form_start, form_end):
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

def intra_trade_corr(ta, tb, t1min, entry_ts, exit_ts):
    da = t1min.get(ta)
    db = t1min.get(tb)
    if da is None or db is None:
        return np.nan
    wa = da[(da.index >= entry_ts) & (da.index <= exit_ts)]["close"]
    wb = db[(db.index >= entry_ts) & (db.index <= exit_ts)]["close"]
    if len(wa) < 10 or len(wb) < 10:
        return np.nan
    ra = wa.pct_change().dropna()
    rb = wb.pct_change().dropna()
    al = ra.align(rb, join="inner")
    if len(al[0]) < 10:
        return np.nan
    return float(al[0].corr(al[1]))

def classify_mechanism(pair_df, trade, spread_std):
    entry_ts  = pd.Timestamp(trade["entry_ts"])
    exit_ts   = pd.Timestamp(trade["exit_ts"])
    direction = trade["direction"]
    window    = pair_df.loc[entry_ts:exit_ts]
    if len(window) < 2:
        return "UNKNOWN", {}
    z = window["zscore"].values
    s = window["spread_prior"].values
    n = len(z)
    if direction == 1:
        max_adv_z = float(np.nanmax(z))
        peak_idx  = int(np.nanargmax(z))
    else:
        max_adv_z = float(np.nanmin(z))
        peak_idx  = int(np.nanargmin(z))
    final_z = float(z[-1])
    pct_early = peak_idx / max(n - 1, 1)
    t_arr = np.arange(len(s), dtype=float)
    valid = ~np.isnan(s)
    slope_hr = 0.0
    if valid.sum() > 5:
        sl, *_ = linregress(t_arr[valid], s[valid])
        slope_hr = sl * 60 / max(spread_std, 1e-10)
    det = {
        "n_bars": n, "max_adv_z": round(max_adv_z, 2),
        "final_z": round(final_z, 2), "pct_early": round(pct_early, 2),
        "slope_hr": round(slope_hr, 3), "net_pnl": round(trade["net_pnl"], 1),
    }
    if trade["net_pnl"] >= 0:
        return "REVERT", det
    if pct_early < 0.10 and n > 30:
        return "JUMP", det
    wrong = (direction == 1 and slope_hr > 0.3) or (direction == -1 and slope_hr < -0.3)
    if wrong and abs(slope_hr) > 0.3:
        return "TREND", det
    return "DRIFT", det


# ---------------------------------------------------------------------------
# Load caches
# ---------------------------------------------------------------------------

print("Loading caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} 1-min  {len(cache_5min)} 5-min\n")

# ---------------------------------------------------------------------------
# Per-fold audit -- collect all trade records
# ---------------------------------------------------------------------------

all_trades = []   # flat list of trade dicts across all 3 folds

for fold_n, form_start, form_end, trading_month in AUDIT_FOLDS:

    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    pairs_df = pd.read_csv(csv)
    if len(pairs_df) > 500:
        pairs_df = pairs_df.nsmallest(500, "johansen_pval").reset_index(drop=True)

    tickers = set(pairs_df.ticker_a.tolist() + pairs_df.ticker_b.tolist())
    t1min = _slice_month(cache_1min, tickers, trading_month)
    f1min = _slice(cache_1min, tickers, form_start, form_end)
    f5min = _slice(cache_5min, tickers, form_start, form_end)

    # Gate + HL + CORR25
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)

    # Compute formation corr and apply CORR25
    fcorrs = []
    for _, row in pairs_gated.iterrows():
        fcorrs.append(formation_corr(row.ticker_a, row.ticker_b, f5min, form_start, form_end))
    pairs_gated["fcorr"] = fcorrs
    pairs_filtered = pairs_gated[
        pairs_gated["fcorr"].notna() & (pairs_gated["fcorr"] >= CORR25_THRESH)
    ].reset_index(drop=True)

    n_filt = len(pairs_filtered)
    print(f"Fold {fold_n:02d} [{trading_month}]: {len(pairs_df)} -> gate -> {len(pairs_gated)} -> CORR25 -> {n_filt} pairs")

    exec_res = run_fold_execution(
        pairs_df=pairs_filtered, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )

    # Collect trade records
    for (ta, tb), pair_df in exec_res.items():
        ca = t1min.get(ta, pd.DataFrame()).get("close", pd.Series(dtype=float))
        cb = t1min.get(tb, pd.DataFrame()).get("close", pd.Series(dtype=float))
        if ca.empty or cb.empty:
            continue
        result = compute_pair_pnl(pair_df, ca, cb, PER_PAIR_DOLLAR,
                                   tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS)
        spread_std = float(np.nanstd(pair_df["spread_prior"].values))
        fc = float(pairs_filtered[
            (pairs_filtered.ticker_a == ta) & (pairs_filtered.ticker_b == tb)
        ]["fcorr"].iloc[0]) if not pairs_filtered[
            (pairs_filtered.ticker_a == ta) & (pairs_filtered.ticker_b == tb)
        ].empty else np.nan

        for trade in result["trade_log"]:
            mech, det = classify_mechanism(pair_df, trade, spread_std)
            entry_ts = pd.Timestamp(trade["entry_ts"])
            exit_ts  = pd.Timestamp(trade["exit_ts"])
            ic = intra_trade_corr(ta, tb, t1min, entry_ts, exit_ts)
            # Would a 3d max hold have changed the outcome?
            would_exit_3d = det["n_bars"] > MAX_HOLD_BARS_3D
            all_trades.append({
                "fold":          fold_n,
                "month":         trading_month,
                "ta":            ta,
                "tb":            tb,
                "pair":          f"{ta}/{tb}",
                "direction":     trade["direction"],
                "net_pnl":       trade["net_pnl"],
                "gross_pnl":     trade["gross_pnl"],
                "n_bars":        det["n_bars"],
                "n_days":        round(det["n_bars"] / 390, 1),
                "mechanism":     mech,
                "max_adv_z":     det["max_adv_z"],
                "slope_hr":      det["slope_hr"],
                "formation_corr":fc,
                "intra_corr":    ic,
                "corr_drop":     (fc - ic) if not np.isnan(ic) else np.nan,
                "would_exit_3d": would_exit_3d,
            })

df = pd.DataFrame(all_trades)
losses = df[df["net_pnl"] < 0].copy()
wins   = df[df["net_pnl"] >= 0].copy()

# ---------------------------------------------------------------------------
# Print full trade table per fold
# ---------------------------------------------------------------------------

print()
for fold_n in [38, 39, 40]:
    fdf = df[df["fold"] == fold_n].sort_values("net_pnl")
    print(f"\n{'='*90}")
    print(f"FOLD {fold_n:02d} [{fdf['month'].iloc[0] if len(fdf)>0 else '?'}]  "
          f"{len(fdf)} trades  "
          f"total_PnL=${fdf['net_pnl'].sum():+,.0f}  "
          f"wins={len(fdf[fdf.net_pnl>=0])}  losses={len(fdf[fdf.net_pnl<0])}")
    print(f"{'='*90}")
    print(f"{'Pair':<16} {'Dir':>4} {'PnL':>8} {'N_days':>6} {'Mech':<7} "
          f"{'AdvZ':>5} {'Slope':>6} {'FCorr':>6} {'ICorr':>6} {'CorrDrop':>8}")
    print("-" * 90)
    for _, r in fdf.iterrows():
        dir_s = "+1(LA)" if r["direction"] == 1 else "-1(SA)"
        ic_s  = f"{r['intra_corr']:>6.3f}" if not np.isnan(r["intra_corr"]) else "   nan"
        cd_s  = f"{r['corr_drop']:>8.3f}" if not np.isnan(r["corr_drop"]) else "     nan"
        flag  = " *3d" if r["would_exit_3d"] and r["net_pnl"] < 0 else ""
        print(f"{r['pair']:<16} {dir_s:>6} {r['net_pnl']:>+8,.0f} {r['n_days']:>6.1f} "
              f"{r['mechanism']:<7} {r['max_adv_z']:>5.2f} {r['slope_hr']:>6.3f} "
              f"{r['formation_corr']:>6.3f} {ic_s} {cd_s}{flag}")

# ---------------------------------------------------------------------------
# H1 — Ticker concentration
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("H1 -- TICKER CONCENTRATION")
print(f"{'='*70}")
print(f"Total loss: ${losses['net_pnl'].sum():+,.0f} across {len(losses)} losing trades\n")

# Count each ticker's total loss exposure (sum of net_pnl of trades it appears in)
ticker_loss = {}
for _, r in losses.iterrows():
    for tk in [r["ta"], r["tb"]]:
        ticker_loss[tk] = ticker_loss.get(tk, 0.0) + r["net_pnl"]

ticker_loss = sorted(ticker_loss.items(), key=lambda x: x[1])
total_loss  = losses["net_pnl"].sum()

print(f"{'Ticker':<10} {'Loss ($)':>10} {'Share (%)':>10} {'N trades':>9}")
print("-" * 45)
for tk, loss in ticker_loss[:15]:
    n_trades = len(losses[(losses.ta == tk) | (losses.tb == tk)])
    print(f"{tk:<10} {loss:>+10,.0f} {100*loss/total_loss:>9.1f}% {n_trades:>9}")

top2_share = sum(loss for _, loss in ticker_loss[:2]) / total_loss if total_loss != 0 else 0
top5_share = sum(loss for _, loss in ticker_loss[:5]) / total_loss if total_loss != 0 else 0
print(f"\nTop-2 ticker loss share: {100*top2_share:.1f}%  (threshold: {100*TICKER_CONC_THRESH:.0f}%)")
print(f"Top-5 ticker loss share: {100*top5_share:.1f}%")

verdict_h1 = abs(top2_share) > TICKER_CONC_THRESH
print(f"\nH1 VERDICT: {'CONCENTRATED -- test ticker cap' if verdict_h1 else 'DIFFUSE -- losses spread across tickers, ticker cap unlikely to help'}")

# ---------------------------------------------------------------------------
# H2 — Correlation collapse during trade
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("H2 -- CORRELATION COLLAPSE DURING TRADE")
print(f"{'='*70}")

valid_corr = losses[losses["intra_corr"].notna() & losses["formation_corr"].notna()].copy()
print(f"Trades with valid corr data: {len(valid_corr)} / {len(losses)}\n")

if len(valid_corr) > 0:
    avg_fcorr  = valid_corr["formation_corr"].mean()
    avg_icorr  = valid_corr["intra_corr"].mean()
    avg_drop   = valid_corr["corr_drop"].mean()
    pct_below_threshold = (valid_corr["intra_corr"] < 0.15).mean()
    pct_negative = (valid_corr["intra_corr"] < 0.0).mean()

    print(f"Formation corr (mean):      {avg_fcorr:>7.3f}")
    print(f"Intra-trade corr (mean):    {avg_icorr:>7.3f}")
    print(f"Average corr drop:          {avg_drop:>7.3f}  (threshold: {CORR_DROP_THRESH:.2f})")
    print(f"% trades with intra < 0.15: {100*pct_below_threshold:>6.1f}%  (threshold: {100*CORR_COLLAPSE_PCT:.0f}%)")
    print(f"% trades with intra < 0.0:  {100*pct_negative:>6.1f}%")
    print()
    print(f"Corr distribution for losing trades:")
    for lo, hi in [(-.5,0),(0,.1),(.1,.2),(.2,.3),(.3,.5),(0.5,1.1)]:
        n = ((valid_corr["intra_corr"] >= lo) & (valid_corr["intra_corr"] < hi)).sum()
        bar = "#" * n
        print(f"  [{lo:+.1f}, {hi:+.1f}): {n:>3}  {bar}")

    verdict_h2 = avg_drop > CORR_DROP_THRESH or pct_below_threshold > CORR_COLLAPSE_PCT
    print(f"\nH2 VERDICT: {'COLLAPSE -- correlation drops materially during trade, live corr exit could help' if verdict_h2 else 'STABLE -- intra-trade corr close to formation corr, correlation exit unlikely to help'}")
else:
    verdict_h2 = False
    print("H2 VERDICT: INSUFFICIENT DATA")

# ---------------------------------------------------------------------------
# H3 — Direction bias and long drift
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("H3 -- DIRECTION BIAS + LONG DRIFT")
print(f"{'='*70}")

n_long_losses  = (losses["direction"] ==  1).sum()   # long A, short B
n_short_losses = (losses["direction"] == -1).sum()   # short A, long B
dir_bias = max(n_long_losses, n_short_losses) / max(len(losses), 1)

print(f"Losing trades direction breakdown:")
print(f"  +1 (Long A, Short B): {n_long_losses:>3} ({100*n_long_losses/max(len(losses),1):.0f}%)")
print(f"  -1 (Short A, Long B): {n_short_losses:>3} ({100*n_short_losses/max(len(losses),1):.0f}%)")
print(f"  Direction bias: {100*dir_bias:.0f}%  (threshold: {100*DIRECTION_BIAS:.0f}%)")

print()
print(f"Hold duration of losing trades:")
print(f"  Mean: {losses['n_days'].mean():.1f} days  ({losses['n_bars'].mean():.0f} bars)")
print(f"  Median: {losses['n_days'].median():.1f} days")
print(f"  Max:  {losses['n_days'].max():.1f} days")
print(f"  % longer than 3 days (1170 bars): {100*(losses['n_bars'] > MAX_HOLD_BARS_3D).mean():.0f}%")

# Simulate: if 3d max hold, which losing trades would have been cut?
drift_losses = losses[losses["mechanism"].isin(["DRIFT", "TREND"])]
drift_cut    = drift_losses[drift_losses["n_bars"] > MAX_HOLD_BARS_3D]
drift_saved  = drift_cut["net_pnl"].sum()
print()
print(f"DRIFT/TREND trades cut by 3-day max hold:")
print(f"  {len(drift_cut)} trades would be forcibly exited early")
print(f"  Max possible PnL saved: ${drift_saved:+,.0f}  (assumes exit at 3d = breakeven, actual depends on mid-trade price)")
print(f"  Note: actual savings < ${drift_saved:+,.0f} -- exit at 3d captures some loss already accrued")

verdict_h3 = (dir_bias > DIRECTION_BIAS) or (losses["n_bars"] > MAX_HOLD_BARS_3D).mean() > 0.5
print(f"\nH3 VERDICT: {'DIRECTION BIAS or LONG DRIFT -- test 3-day max hold' if verdict_h3 else 'BALANCED -- no strong direction bias, hold duration not extreme'}")

# ---------------------------------------------------------------------------
# Mechanism breakdown
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("MECHANISM BREAKDOWN")
print(f"{'='*70}")

for fold_n in [38, 39, 40]:
    flosses = losses[losses["fold"] == fold_n]
    if len(flosses) == 0:
        continue
    counts = flosses["mechanism"].value_counts()
    total_loss_fold = flosses["net_pnl"].sum()
    print(f"\nFold {fold_n}: {len(flosses)} losing trades  total=${total_loss_fold:+,.0f}")
    for mech, cnt in counts.items():
        mdf = flosses[flosses["mechanism"] == mech]
        print(f"  {mech:<7}: {cnt:>2} trades  avg_loss=${mdf['net_pnl'].mean():+,.0f}  avg_days={mdf['n_days'].mean():.1f}d  avg_adv_Z={mdf['max_adv_z'].mean():.2f}")

print()
print("Overall mechanism breakdown across all 3 folds (losing trades):")
for mech, grp in losses.groupby("mechanism"):
    print(f"  {mech:<7}: {len(grp):>2} ({100*len(grp)/len(losses):.0f}%)  avg_loss=${grp['net_pnl'].mean():+,.0f}  avg_days={grp['n_days'].mean():.1f}d")

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

verdicts = []
if verdict_h1:
    verdicts.append("H1 (ticker concentration)")
if verdict_h2:
    verdicts.append("H2 (correlation collapse)")
if verdict_h3:
    verdicts.append("H3 (direction bias / long drift)")

if verdicts:
    print(f"\nSupported hypotheses: {', '.join(verdicts)}")
    print("\nRecommended next tests:")
    if verdict_h1:
        print("  -> Ticker exposure cap: limit each ticker to appear in at most N losing pairs")
    if verdict_h2:
        print("  -> Live correlation exit: exit if rolling 60-bar 1-min corr drops below 0.10 during trade")
    if verdict_h3:
        print("  -> 3-day max holding cap: force exit after 1170 bars regardless of Z-score")
else:
    print("\nH1, H2, H3 all rejected.")
    print("CONCLUSION: Regime limitation.")
    print("  Losses in Folds 38/39/40 are broad-based, structurally diffuse, and")
    print("  not attributable to a single fixable mechanism. The late-2025 market")
    print("  environment caused widespread cointegration breakdown that cannot be")
    print("  screened with pair-level or trade-level filters.")
    print("  Recommendation: accept as regime risk, report in whitepaper,")
    print("  do not add further filters.")
