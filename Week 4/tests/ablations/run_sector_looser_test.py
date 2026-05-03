"""
run_sector_looser_test.py  --  Looser sector matching rules for pairs filter.

Strict sector filter (same-sector only) gave Delta=+0.76 but only 22 trades
vs 125 baseline (82% pair elimination -- too sparse). Test three looser rules:

  STRICT  -- same sector only (benchmark)
  SUPER   -- same super-sector (4 meta-groups based on economic linkage)
             Defensive = {Utilities, Consumer Defensive, Healthcare}
             Cyclical   = {Consumer Cyclical, Industrials, Basic Materials, Energy}
             Growth     = {Technology, Communication Services}
             Financial  = {Financial Services, Real Estate}

  ADJ     -- same sector OR defined adjacent neighbors:
             Technology <-> Communication Services
             Financial Services <-> Real Estate
             Energy <-> Basic Materials
             Consumer Cyclical <-> Consumer Defensive
             Industrials <-> Basic Materials
             Industrials <-> Energy

  EXCL    -- allow everything EXCEPT specifically blocked combos:
             Utilities  x {Technology, Consumer Cyclical, Healthcare, Communication Services}
             Healthcare x {Consumer Cyclical, Energy, Basic Materials}
             (motivated by bad-fold pairs: D/EXPE, AWK/EXPE, BAX/LEN)

Reports pair count, trade count, SR, and delta vs strict baseline.
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
    (23, "2023-11-01", "2024-04-30", "2024-05", "Bull"),  # Bad fold
    (34, "2024-10-01", "2025-03-31", "2025-04", "Bull"),  # Bad fold (tariff shock)
    (39, "2025-03-01", "2025-08-31", "2025-09", "Bull"),  # Bad fold
    (40, "2025-04-01", "2025-09-30", "2025-10", "Bull"),  # Bad fold
]

# ---------------------------------------------------------------------------
# Sector matching rules
# ---------------------------------------------------------------------------

SUPER_SECTOR = {
    "Utilities":              "Defensive",
    "Consumer Defensive":     "Defensive",
    "Healthcare":             "Defensive",
    "Consumer Cyclical":      "Cyclical",
    "Industrials":            "Cyclical",
    "Basic Materials":        "Cyclical",
    "Energy":                 "Cyclical",
    "Technology":             "Growth",
    "Communication Services": "Growth",
    "Financial Services":     "Financial",
    "Real Estate":            "Financial",
}

ADJACENT_PAIRS = {
    frozenset(["Technology",        "Communication Services"]),
    frozenset(["Financial Services","Real Estate"]),
    frozenset(["Energy",            "Basic Materials"]),
    frozenset(["Consumer Cyclical", "Consumer Defensive"]),
    frozenset(["Industrials",       "Basic Materials"]),
    frozenset(["Industrials",       "Energy"]),
}

# Specifically blocked cross-sector combinations (motivated by bad-fold audit)
BLOCKED_PAIRS = {
    frozenset(["Utilities",  "Technology"]),
    frozenset(["Utilities",  "Consumer Cyclical"]),
    frozenset(["Utilities",  "Healthcare"]),
    frozenset(["Utilities",  "Communication Services"]),
    frozenset(["Utilities",  "Industrials"]),
    frozenset(["Healthcare", "Consumer Cyclical"]),
    frozenset(["Healthcare", "Energy"]),
    frozenset(["Healthcare", "Basic Materials"]),
}


def sector_match_strict(sa, sb):
    return sa == sb


def sector_match_super(sa, sb):
    ga = SUPER_SECTOR.get(sa)
    gb = SUPER_SECTOR.get(sb)
    if ga is None or gb is None:
        return False
    return ga == gb


def sector_match_adj(sa, sb):
    if sa == sb:
        return True
    return frozenset([sa, sb]) in ADJACENT_PAIRS


def sector_match_excl(sa, sb):
    return frozenset([sa, sb]) not in BLOCKED_PAIRS


FILTER_FUNS = {
    "BASELINE": None,          # no sector filter
    "STRICT":   sector_match_strict,
    "SUPER":    sector_match_super,
    "ADJ":      sector_match_adj,
    "EXCL":     sector_match_excl,
}

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


def apply_sector_filter(pairs_df, sector_map, match_fn):
    """Keep pairs where match_fn(sector_a, sector_b) is True."""
    keep = []
    skipped_unknown = 0
    for _, row in pairs_df.iterrows():
        sa = sector_map.get(row["ticker_a"], "Unknown")
        sb = sector_map.get(row["ticker_b"], "Unknown")
        if sa == "Unknown" or sb == "Unknown":
            skipped_unknown += 1
            continue
        if match_fn(sa, sb):
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True), skipped_unknown


def run_config(pairs_df, t1min, f1min):
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
    n_trades = int(m.get("n_trades", 0))
    return float(m.get("sharpe", float("nan"))), n_trades


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading sector cache...")
with open(SECTOR_FILE) as f:
    sector_map = json.load(f)
print(f"  {sum(1 for v in sector_map.values() if v != 'Unknown')} tickers with known sector\n")

print("Loading data caches...")
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} (1-min), {len(cache_5min)} (5-min)\n")

# ---------------------------------------------------------------------------
# Main loop
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
    t1min = _slice_month(cache_1min, tickers, trading_month)
    f1min = _slice(cache_1min, tickers, form_start, form_end)
    f5min = _slice(cache_5min, tickers, form_start, form_end)

    # Apply gate + HL cap once (shared across all sector configs)
    pairs_gated = apply_persistence_gate(pairs_df, f5min, form_end)
    if pairs_gated.empty:
        continue
    pairs_gated = pairs_gated[pairs_gated["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_gated.empty:
        continue

    n_gated = len(pairs_gated)
    print(f"Fold {fold_n:02d} [{trading_month}] {regime} - {n_gated} gated pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime}

    for config_name, match_fn in FILTER_FUNS.items():
        if match_fn is None:
            pairs_cfg = pairs_gated
            n_cfg = n_gated
        else:
            pairs_cfg, n_unk = apply_sector_filter(pairs_gated, sector_map, match_fn)
            n_cfg = len(pairs_cfg)

        sr, nt = run_config(pairs_cfg, t1min, f1min)
        row[f"sr_{config_name}"]     = sr
        row[f"trades_{config_name}"] = nt
        row[f"n_{config_name}"]      = n_cfg

        baseline_sr = row.get("sr_BASELINE", float("nan"))
        delta_str = (f"  delta_vs_base={sr - baseline_sr:+.2f}"
                     if config_name != "BASELINE" and not np.isnan(baseline_sr) else "")
        print(f"  {config_name:<9}: n={n_cfg:>3}  SR={sr:+.2f}  trades={nt}{delta_str}", flush=True)

    fold_results.append(row)
    print()

df = pd.DataFrame(fold_results)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

LABELS = list(FILTER_FUNS.keys())

print("=" * 100)
print("SECTOR FILTER COMPARISON  --  gate + no-EOS + Z=3.0 + HL<=6d + delta=1e-7")
print("(BASELINE=no filter | STRICT=same sector | SUPER=super-sector | ADJ=adjacent | EXCL=excl-worst)")
print("=" * 100)

header = f"{'Fold':<6} {'Mo':<9} {'Reg':<5}"
for lbl in LABELS:
    header += f"  {lbl:>9}"
print(header)
print("-" * 100)

bad_folds = {23, 34, 39, 40}
for _, r in df.iterrows():
    fold_id = int(r["fold"])
    flag = " *BAD" if fold_id in bad_folds else ""
    line = f"{fold_id:<6} {r['month']:<9} {r['regime']:<5}"
    for lbl in LABELS:
        sr = r[f"sr_{lbl}"]
        line += f"  {sr:>+9.2f}" if not np.isnan(sr) else f"  {'nan':>9}"
    print(line + flag)

print("-" * 100)

for stat_name, fn in [("MEAN", lambda s: s.mean()), ("MED", lambda s: s.median())]:
    line = f"{stat_name:<6} {'':<9} {'':<5}"
    for lbl in LABELS:
        val = fn(df[f"sr_{lbl}"].dropna())
        line += f"  {val:>+9.2f}"
    print(line)

line = f"{'POS%':<6} {'':<9} {'':<5}"
for lbl in LABELS:
    pct = (df[f"sr_{lbl}"].dropna() > 0).mean() * 100
    line += f"  {pct:>8.0f}%"
print(line)

line = f"{'TRADES':<6} {'':<9} {'':<5}"
for lbl in LABELS:
    tot = df[f"trades_{lbl}"].sum()
    line += f"  {int(tot):>9}"
print(line)

line = f"{'AVG_N':<6} {'':<9} {'':<5}"
for lbl in LABELS:
    avg_n = df[f"n_{lbl}"].mean()
    line += f"  {avg_n:>9.1f}"
print(line)

print()
print("Bad-fold SR comparison (folds 23, 34, 39, 40):")
bad = df[df["fold"].isin(bad_folds)]
for lbl in LABELS:
    mean_bad = bad[f"sr_{lbl}"].dropna().mean()
    print(f"  {lbl:<9}: mean SR={mean_bad:+.2f}  n={len(bad[f'sr_{lbl}'].dropna())}")

print()
print("Delta vs BASELINE (mean SR):")
base_mean = df["sr_BASELINE"].dropna().mean()
for lbl in LABELS[1:]:
    val     = df[f"sr_{lbl}"].dropna().mean()
    tr_base = df["trades_BASELINE"].sum()
    tr      = df[f"trades_{lbl}"].sum()
    n_base  = df["n_BASELINE"].mean()
    n_avg   = df[f"n_{lbl}"].mean()
    print(f"  {lbl:<9}: delta={val - base_mean:+.2f}  trades={tr} ({100*tr/tr_base:.0f}% of base)  "
          f"avg_pairs={n_avg:.1f} ({100*n_avg/n_base:.0f}% of base)")

print()
print("Per regime:")
for reg in ["Bear", "Bull"]:
    sub = df[df["regime"] == reg]
    line = f"  {reg}: "
    for lbl in LABELS:
        val = sub[f"sr_{lbl}"].dropna().mean()
        line += f"{lbl}={val:+.2f}  "
    print(line)
