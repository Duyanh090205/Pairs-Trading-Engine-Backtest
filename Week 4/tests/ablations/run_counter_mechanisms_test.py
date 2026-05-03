"""
run_counter_mechanisms_test.py  --  All 3 counter-mechanisms vs baseline.

Mechanisms:
  MH1d  -- per-trade max hold 390 bars (1 trading day)
  MH3d  -- per-trade max hold 1170 bars (3 trading days)
  RC25  -- rolling 60-bar intra-trade return correlation < 0.25 -> exit
  RC30  -- rolling 60-bar intra-trade return correlation < 0.30 -> exit
  SF    -- sector filter: only trade pairs in same GICS sector
  COMBO -- MH3d + RC30 (two strongest independent signals)

Base config for all: Option A (gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7)

Test set: 10 standard diagnostic folds + 4 bad folds identified in deep audit.
"""

import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")

from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()

from src.phase2_execution.engine import (
    run_fold_execution, _apply_max_holding,
    _N_OPEN_PAIRS_MAX, _TOTAL_CAPITAL,
)
from src.phase3_backtest.metrics_runner import run_fold_pnl

PHASE1_DIR    = "results/metrics/phase1_folds"
DATA_1MIN     = "data/validated/1min_phase2"
DATA_5MIN     = "data/validated/5min_phase1"
SECTOR_CACHE  = "data/sector_cache.json"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
HL_MAX_DAYS   = 6.0
PER_PAIR_DOLLAR = _TOTAL_CAPITAL / _N_OPEN_PAIRS_MAX

MH_5D  = 1950   # 5 trading days (HL<=6d, expected reversion ~15d; cap at 5d cuts grinders)
MH_7D  = 2730   # 7 trading days (more conservative)
MH_10D = 3900   # 10 trading days (only cuts truly stuck trades)
RC_WIN = 60     # rolling window for correlation (1 hour)

TEST_FOLDS = [
    # Standard 10 diagnostic folds
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
    # 4 bad folds identified in deep audit
    (23, "2023-11-01", "2024-04-30", "2024-05", "Bull"),
    (34, "2024-10-01", "2025-03-31", "2025-04", "Bull"),
    (39, "2025-03-01", "2025-08-31", "2025-09", "Bull"),
    (40, "2025-04-01", "2025-09-30", "2025-10", "Bull"),
]

CONFIGS = [
    "A", "A+MH5d", "A+MH7d", "A+MH10d", "A+RC25", "A+RC30", "A+SF", "A+COMBO",
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
# Mechanism 1: Max holding cap (post-process exec_res)
# ---------------------------------------------------------------------------

def apply_mh_to_exec(exec_res, max_bars):
    return {key: _apply_max_holding(pdf, max_bars) for key, pdf in exec_res.items()}


# ---------------------------------------------------------------------------
# Mechanism 2: Rolling intra-trade correlation exit (post-process exec_res)
# ---------------------------------------------------------------------------

def apply_rolling_corr_exit(pair_df, close_a, close_b, window=60, threshold=0.30):
    """
    During an active position, monitor rolling return correlation.
    If correlation drops below threshold, force exit.
    """
    # Pre-compute full rolling correlation series
    ra = close_a.pct_change(fill_method=None)
    rb = close_b.pct_change(fill_method=None)
    combined     = ra.to_frame("ra").join(rb.to_frame("rb"), how="inner").dropna()
    rolling_corr = combined["ra"].rolling(window).corr(combined["rb"])
    # Align to pair_df index
    corr_aligned = rolling_corr.reindex(pair_df.index)
    corr_vals    = corr_aligned.values

    orig_signal = pair_df["signal"].values.copy()
    n           = len(orig_signal)
    new_signal  = np.zeros(n, dtype=np.int8)
    state       = np.int8(0)
    blocked     = False   # True after corr exit: suppress re-entry until orig returns to 0

    for i in range(n):
        orig = orig_signal[i]

        if blocked:
            # Wait for natural exit before allowing re-entry
            if orig == 0:
                blocked = False
            new_signal[i] = np.int8(0)
            continue

        if state == 0:
            if orig != 0:
                state = np.int8(orig)
        else:
            if orig == 0:
                state = np.int8(0)
            else:
                c = corr_vals[i]
                if not np.isnan(c) and c < threshold:
                    state   = np.int8(0)
                    blocked = True   # prevent immediate re-entry on next bar
        new_signal[i] = state

    new_pos    = np.zeros(n, dtype=np.int8)
    new_pos[1:] = new_signal[:-1]

    out             = pair_df.copy()
    out["signal"]   = new_signal
    out["position"] = new_pos
    return out


def apply_rc_to_exec(exec_res, t1min, window, threshold):
    result = {}
    for (ta, tb), pdf in exec_res.items():
        da = t1min.get(ta)
        db = t1min.get(tb)
        if da is None or db is None or "close" not in da.columns or "close" not in db.columns:
            result[(ta, tb)] = pdf
            continue
        try:
            result[(ta, tb)] = apply_rolling_corr_exit(
                pdf, da["close"], db["close"], window=window, threshold=threshold
            )
        except Exception:
            result[(ta, tb)] = pdf
    return result


# ---------------------------------------------------------------------------
# Mechanism 3: Sector filter (pre-engine, filters pairs_gated)
# ---------------------------------------------------------------------------

def fetch_and_cache_sectors(tickers, cache_path=SECTOR_CACHE):
    """
    Fetch GICS sector for each ticker via yfinance. Cached to disk.
    Unknown tickers get sector='Unknown'.
    """
    existing = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)

    missing = [tk for tk in tickers if tk not in existing]
    if missing:
        print(f"  Fetching sectors for {len(missing)} new tickers via yfinance...", flush=True)
        import yfinance as yf
        for i, tk in enumerate(missing):
            try:
                existing[tk] = yf.Ticker(tk).info.get("sector", "Unknown")
            except Exception:
                existing[tk] = "Unknown"
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(missing)} done", flush=True)
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Sector cache saved ({len(existing)} tickers total)")

    return existing


def apply_sector_filter(pairs_df, sector_map):
    """Keep only pairs where both tickers are in the same GICS sector."""
    if pairs_df.empty:
        return pairs_df
    keep = []
    for _, row in pairs_df.iterrows():
        sa = sector_map.get(row["ticker_a"], "Unknown")
        sb = sector_map.get(row["ticker_b"], "Unknown")
        if sa != "Unknown" and sb != "Unknown" and sa == sb:
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Score a given exec_res dict
# ---------------------------------------------------------------------------

def score(exec_res, t1min, pairs_df):
    if not exec_res:
        return float("nan"), 0
    m = run_fold_pnl(
        exec_res, t1min, pairs_df, FIXED_DELTA, {},
        total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
        tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
    )
    return float(m.get("sharpe", float("nan"))), int(m.get("n_trades", 0))


# ---------------------------------------------------------------------------
# Pre-fetch sector data for all tickers across all folds
# ---------------------------------------------------------------------------

print("Collecting all tickers across test folds for sector lookup...")
all_tickers = set()
for fold_n, *_ in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        df = pd.read_csv(csv)
        all_tickers.update(df["ticker_a"].tolist())
        all_tickers.update(df["ticker_b"].tolist())
    except Exception:
        pass
print(f"  {len(all_tickers)} unique tickers")
sector_map = fetch_and_cache_sectors(all_tickers)
unknown_ct = sum(1 for v in sector_map.values() if v == "Unknown")
print(f"  Sector coverage: {len(sector_map)-unknown_ct}/{len(sector_map)} known\n")

# ---------------------------------------------------------------------------
# Load data caches
# ---------------------------------------------------------------------------

print("Loading data caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")


# ---------------------------------------------------------------------------
# Main fold loop
# ---------------------------------------------------------------------------

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

    # Gate + HL filter (shared across all configs)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        continue
    if "half_life_days" in pairs_gated.columns:
        pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        continue

    # Sector-filtered pairs (for A+SF config — engine runs separately)
    pairs_sf = apply_sector_filter(pairs_gated, sector_map)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime}  gated={len(pairs_gated)}  sector_filtered={len(pairs_sf)}", flush=True)

    # --- Run engine: baseline (and base for MH/RC post-processing) ---
    exec_base = run_fold_execution(
        pairs_df=pairs_gated, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )

    # --- Run engine: sector-filtered (separate run for A+SF) ---
    if not pairs_sf.empty:
        exec_sf = run_fold_execution(
            pairs_df=pairs_sf, trading_1min=t1min, delta=FIXED_DELTA,
            eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
        )
    else:
        exec_sf = {}

    row = {"fold": fold_n, "month": trading_month, "regime": regime,
           "n_gated": len(pairs_gated), "n_sf": len(pairs_sf)}

    # Score each config
    configs_to_score = [
        ("A",        exec_base,                                                        pairs_gated),
        ("A+MH5d",   apply_mh_to_exec(exec_base, MH_5D),                              pairs_gated),
        ("A+MH7d",   apply_mh_to_exec(exec_base, MH_7D),                              pairs_gated),
        ("A+MH10d",  apply_mh_to_exec(exec_base, MH_10D),                             pairs_gated),
        ("A+RC25",   apply_rc_to_exec(exec_base, t1min, RC_WIN, 0.25),                pairs_gated),
        ("A+RC30",   apply_rc_to_exec(exec_base, t1min, RC_WIN, 0.30),                pairs_gated),
        ("A+SF",     exec_sf,                                 pairs_sf if not pairs_sf.empty else pairs_gated),
        ("A+COMBO",  apply_rc_to_exec(apply_mh_to_exec(exec_base, MH_7D), t1min, RC_WIN, 0.25), pairs_gated),
    ]

    for label, exec_res, p_df in configs_to_score:
        sr, nt = score(exec_res, t1min, p_df)
        row[f"sr_{label}"]     = sr
        row[f"trades_{label}"] = nt

    fold_results.append(row)

    # Per-fold print
    base_sr = row["sr_A"]
    print(f"  {'Config':<12} {'SR':>7}  {'Trades':>6}  {'Delta':>7}")
    for lbl in CONFIGS:
        sr  = row[f"sr_{lbl}"]
        nt  = row[f"trades_{lbl}"]
        dlt = (sr - base_sr) if lbl != "A" and not np.isnan(sr) and not np.isnan(base_sr) else float("nan")
        dlt_str = f"{dlt:+.2f}" if not np.isnan(dlt) else "   -"
        print(f"  {lbl:<12} {sr:>+7.2f}  {nt:>6}  {dlt_str:>7}")
    print()

df = pd.DataFrame(fold_results)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("=" * 95)
print("COUNTER-MECHANISMS SUMMARY  --  14 folds (10 diagnostic + 4 bad folds)")
print("=" * 95)

header = f"{'Fold':<6} {'Mo':<9} {'Reg':<5}"
for lbl in CONFIGS:
    header += f"  {lbl:>9}"
print(header)
print("-" * 95)

for _, r in df.iterrows():
    line = f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5}"
    for lbl in CONFIGS:
        sr = r[f"sr_{lbl}"]
        line += f"  {sr:>+9.2f}" if not np.isnan(sr) else f"  {'nan':>9}"
    print(line)

print("-" * 95)

for stat_name, fn in [("MEAN", lambda s: s.mean()), ("MED", lambda s: s.median())]:
    line = f"{stat_name:<6} {'':<9} {'':<5}"
    for lbl in CONFIGS:
        val = fn(df[f"sr_{lbl}"].dropna())
        line += f"  {val:>+9.2f}"
    print(line)

line = f"{'POS%':<6} {'':<9} {'':<5}"
for lbl in CONFIGS:
    pct = (df[f"sr_{lbl}"].dropna() > 0).mean() * 100
    line += f"  {pct:>8.0f}%"
print(line)

line = f"{'TRADES':<6} {'':<9} {'':<5}"
for lbl in CONFIGS:
    tot = df[f"trades_{lbl}"].sum()
    line += f"  {int(tot):>9}"
print(line)

print()
print("Delta vs baseline A (mean SR):")
base_mean = df["sr_A"].dropna().mean()
for lbl in CONFIGS[1:]:
    val  = df[f"sr_{lbl}"].dropna().mean()
    tr_a = df["trades_A"].sum()
    tr   = df[f"trades_{lbl}"].sum()
    print(f"  {lbl:<12}: delta={val-base_mean:+.2f}  trades={tr} ({100*tr/max(tr_a,1):.0f}% of baseline)")

print()
print("Per regime (14 folds):")
for reg in ["Bear", "Bull"]:
    sub  = df[df["regime"] == reg]
    line = f"  {reg} (n={len(sub)}): "
    for lbl in CONFIGS:
        val = sub[f"sr_{lbl}"].dropna().mean()
        line += f"{lbl}={val:+.2f}  "
    print(line)

print()
print("Bad folds only (23, 34, 39, 40):")
bad = df[df["fold"].isin([23, 34, 39, 40])]
if len(bad) > 0:
    line = "  "
    for lbl in CONFIGS:
        val = bad[f"sr_{lbl}"].dropna().mean()
        line += f"{lbl}={val:+.2f}  "
    print(line)

print()
print("Good folds only (1,2,5,6,8,10,11,12,13,17):")
good = df[df["fold"].isin([1,2,5,6,8,10,11,12,13,17])]
if len(good) > 0:
    line = "  "
    for lbl in CONFIGS:
        val = good[f"sr_{lbl}"].dropna().mean()
        line += f"{lbl}={val:+.2f}  "
    print(line)
