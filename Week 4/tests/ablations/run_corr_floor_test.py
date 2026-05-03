"""
run_corr_floor_test.py  --  Formation-window returns correlation floor.

Empirical finding: bad pairs have low formation-window returns correlation:
  D/EXPE   = 0.127   AWK/EXPE = 0.206   BAX/LEN = 0.225   ADP/LRCX = 0.052

Good pairs that actually traded:
  MLM/VMC (Fold17, SR+2.27) = 0.785   -- high corr, safe
  LDOS/UBER (Fold17, -0.107) -- never traded, so filtering it is free

Hypothesis: corr > 0.25 (full formation window, 5-min returns) screens the
economically incoherent pairs without touching the pairs that actually enter.

Thresholds tested: BASELINE (no floor), 0.20, 0.25, 0.30, 0.40.
Also tests: EXCL+CORR25 (combine exclusion list with corr floor).

Correlation computed on full formation window 5-min log-return series.
"""

import sys, os, json
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
SECTOR_FILE = "data/sector_cache.json"

FIXED_DELTA   = 1e-7
TC_BPS        = 30.0
BORROW_BPS    = 50.0
ENTRY_Z       = 3.0
JOHANSEN_PVAL = 0.05
HL_MAX_DAYS   = 6.0

CORR_THRESHOLDS = [None, 0.20, 0.25, 0.30, 0.40]  # None = baseline

# Worst cross-sector exclusion list (from bad-fold audit)
BLOCKED_PAIRS = {
    frozenset(["Utilities",  "Technology"]),
    frozenset(["Utilities",  "Consumer Cyclical"]),
    frozenset(["Utilities",  "Healthcare"]),
    frozenset(["Utilities",  "Communication Services"]),
    frozenset(["Utilities",  "Industrials"]),
    frozenset(["Healthcare", "Consumer Cyclical"]),
    frozenset(["Healthcare", "Energy"]),
    frozenset(["Healthcare", "Basic Materials"]),
    frozenset(["Industrials","Technology"]),  # added: ADP/LRCX pattern (Fold 34)
}

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
        pval = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pval) and pval < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def compute_formation_corr(pairs_df, form_5min, form_start, form_end):
    """
    Adds 'formation_corr' column: full-window 5-min log-return correlation.
    Pairs missing data get NaN (excluded by any positive threshold).
    """
    corrs = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = form_5min.get(ta)
        fb = form_5min.get(tb)
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


def apply_excl_filter(pairs_df, sector_map):
    """Remove specifically blocked cross-sector combos."""
    keep = []
    for _, row in pairs_df.iterrows():
        sa = sector_map.get(row["ticker_a"], "Unknown")
        sb = sector_map.get(row["ticker_b"], "Unknown")
        if frozenset([sa, sb]) not in BLOCKED_PAIRS:
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True)


def run_execution(pairs_df, t1min, f1min):
    if pairs_df is None or pairs_df.empty:
        return float("nan"), 0
    exec_res = run_fold_execution(
        pairs_df=pairs_df, trading_1min=t1min, delta=FIXED_DELTA,
        eos_flatten=False, formation_ref=f1min, entry_z=ENTRY_Z,
    )
    if not exec_res:
        return float("nan"), 0
    m = run_fold_pnl(
        exec_res, t1min, pairs_df, FIXED_DELTA, {},
        total_capital=_TOTAL_CAPITAL, n_open_pairs_max=_N_OPEN_PAIRS_MAX,
        tc_bps=TC_BPS, borrow_bps_yr=BORROW_BPS,
    )
    return float(m.get("sharpe", float("nan"))), int(m.get("n_trades", 0))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading sector cache...")
with open(SECTOR_FILE) as f:
    sector_map = json.load(f)
print(f"  {sum(1 for v in sector_map.values() if v != 'Unknown')} tickers with known sector")

print("Loading data caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

CONFIGS = (
    [f"CORR{int(t*100):02d}" if t is not None else "BASE" for t in CORR_THRESHOLDS]
    + ["EXCL", "EXCL+C25"]
)

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
    t1min = _slice_month(cache_1min, tickers, trading_month)
    f1min = _slice(cache_1min, tickers, form_start, form_end)
    f5min = _slice(cache_5min, tickers, form_start, form_end)

    # Gate + HL cap (shared)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        continue
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        continue

    # Precompute formation corr for all gated pairs (once per fold)
    pairs_with_corr = compute_formation_corr(pairs_gated, f5min, form_start, form_end)
    n_gated = len(pairs_gated)

    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_gated} gated pairs  "
          f"(corr stats: mean={pairs_with_corr.formation_corr.mean():.2f}  "
          f"min={pairs_with_corr.formation_corr.min():.2f}  "
          f"p25={pairs_with_corr.formation_corr.quantile(0.25):.2f})", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime}

    for threshold, cfg_name in zip(CORR_THRESHOLDS, CONFIGS[:len(CORR_THRESHOLDS)]):
        if threshold is None:
            pairs_cfg = pairs_with_corr
        else:
            pairs_cfg = pairs_with_corr[
                pairs_with_corr["formation_corr"].notna() &
                (pairs_with_corr["formation_corr"] >= threshold)
            ].reset_index(drop=True)
        n_cfg = len(pairs_cfg)
        sr, nt = run_execution(pairs_cfg, t1min, f1min)
        row[f"sr_{cfg_name}"]     = sr
        row[f"trades_{cfg_name}"] = nt
        row[f"n_{cfg_name}"]      = n_cfg
        base_sr = row.get("sr_BASE", float("nan"))
        delta_str = (f"  d={sr - base_sr:+.2f}" if cfg_name != "BASE" and not np.isnan(base_sr) else "")
        print(f"  {cfg_name:<10}: n={n_cfg:>3}  SR={sr:+.2f}  trades={nt}{delta_str}", flush=True)

    # EXCL filter
    pairs_excl = apply_excl_filter(pairs_with_corr, sector_map)
    n_excl = len(pairs_excl)
    sr_excl, nt_excl = run_execution(pairs_excl, t1min, f1min)
    row["sr_EXCL"] = sr_excl; row["trades_EXCL"] = nt_excl; row["n_EXCL"] = n_excl
    base_sr = row.get("sr_BASE", float("nan"))
    print(f"  {'EXCL':<10}: n={n_excl:>3}  SR={sr_excl:+.2f}  trades={nt_excl}  d={sr_excl - base_sr:+.2f}", flush=True)

    # EXCL + CORR25
    pairs_ec = pairs_excl[
        pairs_excl["formation_corr"].notna() &
        (pairs_excl["formation_corr"] >= 0.25)
    ].reset_index(drop=True)
    n_ec = len(pairs_ec)
    sr_ec, nt_ec = run_execution(pairs_ec, t1min, f1min)
    row["sr_EXCL+C25"] = sr_ec; row["trades_EXCL+C25"] = nt_ec; row["n_EXCL+C25"] = n_ec
    print(f"  {'EXCL+C25':<10}: n={n_ec:>3}  SR={sr_ec:+.2f}  trades={nt_ec}  d={sr_ec - base_sr:+.2f}", flush=True)

    fold_results.append(row)
    print()

df = pd.DataFrame(fold_results)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 110)
print("FORMATION CORRELATION FLOOR  --  gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7  (14 folds)")
print("=" * 110)

header = f"{'Fold':<6} {'Mo':<9} {'Reg':<5}"
for c in CONFIGS:
    header += f"  {c:>10}"
print(header)
print("-" * 110)

bad_folds = {23, 34, 39, 40}
for _, r in df.iterrows():
    fold_id = int(r["fold"])
    flag = " *" if fold_id in bad_folds else ""
    line = f"{fold_id:<6} {r['month']:<9} {r['regime']:<5}"
    for c in CONFIGS:
        sr = r.get(f"sr_{c}", float("nan"))
        line += f"  {sr:>+10.2f}" if not np.isnan(sr) else f"  {'nan':>10}"
    print(line + flag)

print("-" * 110)

for stat, fn in [("MEAN", lambda s: s.mean()), ("MED", lambda s: s.median())]:
    line = f"{stat:<6} {'':<9} {'':<5}"
    for c in CONFIGS:
        val = fn(df[f"sr_{c}"].dropna())
        line += f"  {val:>+10.2f}"
    print(line)

line = f"{'POS%':<6} {'':<9} {'':<5}"
for c in CONFIGS:
    pct = (df[f"sr_{c}"].dropna() > 0).mean() * 100
    line += f"  {pct:>9.0f}%"
print(line)

line = f"{'TRADES':<6} {'':<9} {'':<5}"
for c in CONFIGS:
    tot = df[f"trades_{c}"].sum()
    line += f"  {int(tot):>10}"
print(line)

line = f"{'AVG_N':<6} {'':<9} {'':<5}"
for c in CONFIGS:
    avg = df[f"n_{c}"].mean()
    line += f"  {avg:>10.1f}"
print(line)

print()
print("Bad-fold SR (folds 23, 34, 39, 40):")
bad = df[df["fold"].isin(bad_folds)]
for c in CONFIGS:
    mean_bad = bad[f"sr_{c}"].dropna().mean()
    print(f"  {c:<12}: {mean_bad:+.2f}")

print()
print("Delta vs BASELINE:")
base_mean = df["sr_BASE"].dropna().mean()
base_trades = df["trades_BASE"].sum()
base_n = df["n_BASE"].mean()
for c in CONFIGS[1:]:
    val = df[f"sr_{c}"].dropna().mean()
    tr  = df[f"trades_{c}"].sum()
    n   = df[f"n_{c}"].mean()
    print(f"  {c:<12}: delta={val - base_mean:+.2f}  trades={tr} ({100*tr/base_trades:.0f}%)  "
          f"avg_pairs={n:.1f} ({100*n/base_n:.0f}%)")

print()
print("Per regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    line = f"  {reg}: "
    for c in CONFIGS:
        val = sub[f"sr_{c}"].dropna().mean()
        line += f"{c}={val:+.2f}  "
    print(line)
