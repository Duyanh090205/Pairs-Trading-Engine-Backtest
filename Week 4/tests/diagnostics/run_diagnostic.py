"""
run_diagnostic.py  —  Root-cause isolation for strategy failure.

Test A (Oracle Z)   : Z-score normalised with trading-window's OWN spread stats.
                      If Sharpe turns positive → pairs CAN mean-revert; problem
                      is formation stats not matching trading window (non-persistence).

Test B (Reverse)    : Flip signal direction — enter on momentum, not mean-reversion.
                      If Sharpe turns positive → we have been trading backwards.

Test C (Same-Sector): Restrict pairs to same GICS sector only.
                      If Sharpe improves significantly → cross-sector spurious
                      cointegration is the primary driver of losses.

Folds: 1,2,5,6 (Bear 2022)  +  8,10,11,12,13,17 (Bull 2023)
Fixed delta = 1e-7 across all tests (isolate signal, not delta selection).

Usage:
    python run_diagnostic.py              # all three tests
    python run_diagnostic.py --skip-c    # skip same-sector (no yfinance needed)
"""

import sys, os, json, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ── Numba warmup ──────────────────────────────────────────────────────────────
print("Warming up Numba JIT...")
from src.phase2_execution.kalman import run_kalman, warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import (
    run_pair,
    apply_ticker_concentration_cap,
    _session_warmup_mask,
    _apply_eos_flatten,
    _state_machine,
    _SESSION_WARMUP_BARS,
    _N_OPEN_PAIRS_MAX,
    _TOTAL_CAPITAL,
)
from src.phase3_backtest.metrics_runner import run_fold_pnl

# ── Constants ─────────────────────────────────────────────────────────────────
PHASE1_DIR   = "results/metrics/phase1_folds"
DATA_1MIN    = "data/validated/1min_phase2"
DATA_5MIN    = "data/validated/5min_phase1"
OUT_DIR      = "results/metrics/diagnostic"
SECTOR_CACHE = "results/metrics/sector_cache.json"

FIXED_DELTA  = 1e-7
TC_BPS       = 30.0
BORROW_BPS   = 50.0
ENTRY_Z      = 2.0
SKIP_C       = "--skip-c" in sys.argv

os.makedirs(OUT_DIR, exist_ok=True)

TEST_FOLDS = [
    # (fold_n, form_start, form_end, trading_month, regime)
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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_cache(data_dir: str, compute_log_close: bool) -> dict:
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


def _slice(cache: dict, tickers: set, start: str, end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_month(cache: dict, tickers: set, month: str) -> dict:
    year, mon = int(month[:4]), int(month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == mon)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ── Sector map (Test C) ───────────────────────────────────────────────────────

def _get_sector_map(tickers: list) -> dict:
    """Fetch GICS sector for each ticker via yfinance. Caches results to JSON."""
    cache: dict = {}
    if os.path.exists(SECTOR_CACHE):
        with open(SECTOR_CACHE) as f:
            cache = json.load(f)

    missing = [t for t in tickers if t not in cache]
    if missing:
        try:
            import yfinance as yf
            print(f"  Fetching sector info for {len(missing)} tickers via yfinance...")
            for i, t in enumerate(missing):
                try:
                    info = yf.Ticker(t).info
                    cache[t] = info.get("sector", "Unknown")
                except Exception:
                    cache[t] = "Unknown"
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(missing)} done")
                time.sleep(0.15)
            with open(SECTOR_CACHE, "w") as f:
                json.dump(cache, f)
        except ImportError:
            print("  yfinance not installed — Test C will be skipped.")
            for t in missing:
                cache[t] = "Unknown"

    return cache


# ── Per-pair execution helpers ────────────────────────────────────────────────

def _prep_pair(row, trading_1min):
    """Extract aligned arrays for a single pair from trading cache."""
    ta, tb = row["ticker_a"], row["ticker_b"]
    if ta not in trading_1min or tb not in trading_1min:
        return None
    df_a = trading_1min[ta]
    df_b = trading_1min[tb]
    df_ab = (
        df_a[["log_close", "close"]]
        .join(df_b[["log_close", "close"]], how="inner", lsuffix="_a", rsuffix="_b")
        .dropna()
    )
    if len(df_ab) < 20:
        return None
    return df_ab


# ── Test A: Oracle Z-score ────────────────────────────────────────────────────

def run_oracle_fold(pairs_df: pd.DataFrame, trading_1min: dict) -> dict:
    """
    For each pair: compute Kalman prior spread on TRADING window,
    use its own mean/std as Z-score anchor (look-ahead oracle).
    """
    pairs_capped = apply_ticker_concentration_cap(pairs_df)
    results = {}

    for _, row in pairs_capped.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        df_ab = _prep_pair(row, trading_1min)
        if df_ab is None:
            continue

        log_a = df_ab["log_close_a"].values
        log_b = df_ab["log_close_b"].values

        # Compute prior spread on trading window to get oracle stats
        sp_prior, *_ = run_kalman(
            log_a, log_b,
            float(row["alpha_pca"]), float(row["beta_pca"]),
            float(row["R_measurement_noise"]), FIXED_DELTA,
        )
        sp_clean = sp_prior[~np.isnan(sp_prior)]
        if len(sp_clean) < 20:
            continue

        oracle_mean = float(np.mean(sp_clean))
        oracle_std  = max(float(np.std(sp_clean)), 1e-10)

        pair_df = run_pair(
            log_a_1min   = log_a,
            log_b_1min   = log_b,
            close_a_1min = df_ab["close_a"].values,
            close_b_1min = df_ab["close_b"].values,
            index_1min   = df_ab.index,
            alpha_pca    = float(row["alpha_pca"]),
            beta_pca     = float(row["beta_pca"]),
            R            = float(row["R_measurement_noise"]),
            delta        = FIXED_DELTA,
            half_life_days = float(row["half_life_days"]),
            entry_z      = ENTRY_Z,
            spread_mean_form = oracle_mean,
            spread_std_form  = oracle_std,
        )
        if not pair_df.empty:
            results[(ta, tb)] = pair_df

    return results


# ── Test B: Reverse signal ────────────────────────────────────────────────────

def reverse_exec_results(exec_results: dict) -> dict:
    """
    Flip position and signal sign in every pair_df.
    Converts mean-reversion entries into momentum entries.
    Zero (flat) bars remain zero. EOS-flattened zeros stay zero.
    The lookahead invariant position[i] == signal[i-1] is preserved
    because both arrays are negated together.
    """
    flipped = {}
    for key, df in exec_results.items():
        df2 = df.copy()
        df2["position"] = -df2["position"].values  # int8 negation
        df2["signal"]   = -df2["signal"].values
        flipped[key] = df2
    return flipped


# ── Test C: Same-sector filter ────────────────────────────────────────────────

def filter_same_sector(pairs_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Keep only pairs where both tickers share the same known GICS sector."""
    keep = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        sa = sector_map.get(ta, "Unknown")
        sb = sector_map.get(tb, "Unknown")
        if sa == sb and sa != "Unknown":
            keep.append(True)
        else:
            keep.append(False)
    filtered = pairs_df[keep].reset_index(drop=True)
    return filtered


# ── Sharpe from exec results ──────────────────────────────────────────────────

def _sharpe_from_exec(exec_results, trading_1min, pairs_df) -> float:
    """Run PnL and return fold Sharpe. Returns NaN on failure."""
    if not exec_results:
        return float("nan")
    try:
        metrics = run_fold_pnl(
            fold_results     = exec_results,
            trading_1min     = trading_1min,
            pairs_df         = pairs_df,
            delta            = FIXED_DELTA,
            delta_metrics    = {},
            total_capital    = _TOTAL_CAPITAL,
            n_open_pairs_max = _N_OPEN_PAIRS_MAX,
            tc_bps           = TC_BPS,
            borrow_bps_yr    = BORROW_BPS,
        )
        return float(metrics.get("sharpe", float("nan")))
    except Exception as e:
        print(f"      PnL error: {e}")
        return float("nan")


# ── Baseline fold runner ──────────────────────────────────────────────────────

def run_baseline_fold(pairs_df, trading_1min, formation_1min):
    """Baseline: formation-locked Z using 1-min formation data."""
    from src.phase2_execution.engine import run_fold_execution
    return run_fold_execution(
        pairs_df      = pairs_df,
        trading_1min  = trading_1min,
        delta         = FIXED_DELTA,
        eos_flatten   = True,
        formation_ref = formation_1min,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading data caches (1-min + 5-min)...")
t0 = time.time()
cache_1min = _load_cache(DATA_1MIN, compute_log_close=True)
cache_5min = _load_cache(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} tickers (1-min), {len(cache_5min)} tickers (5-min) — {time.time()-t0:.1f}s\n")

# Collect all tickers across test folds for sector fetch
all_tickers_needed = set()
fold_pairs = {}
for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        df = pd.read_csv(csv)
        # Apply pair-count gate consistent with full pipeline
        if len(df) > 500:
            df = df.nsmallest(500, "johansen_pval").reset_index(drop=True)
        fold_pairs[fold_n] = df
        all_tickers_needed.update(df["ticker_a"].tolist())
        all_tickers_needed.update(df["ticker_b"].tolist())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        fold_pairs[fold_n] = pd.DataFrame()

# Test C: fetch sector map
sector_map = {}
if not SKIP_C:
    sector_map = _get_sector_map(sorted(all_tickers_needed))
    known = sum(1 for v in sector_map.values() if v != "Unknown")
    print(f"  Sector map: {known}/{len(sector_map)} tickers with known sector\n")

# ── Per-fold loop ─────────────────────────────────────────────────────────────

rows = []

for fold_n, form_start, form_end, trading_month, regime in TEST_FOLDS:
    pairs_df = fold_pairs.get(fold_n, pd.DataFrame())
    if pairs_df.empty:
        print(f"Fold {fold_n:02d} [{trading_month}]: no pairs — skipped")
        continue

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_month(cache_1min, tickers, trading_month)
    formation_1min = _slice(cache_1min, tickers, form_start, form_end)

    n_pairs = len(pairs_df)
    print(f"Fold {fold_n:02d} [{trading_month}] {regime} — {n_pairs} pairs", flush=True)

    row = {"fold": fold_n, "month": trading_month, "regime": regime, "n_pairs": n_pairs}

    # ── Baseline ──────────────────────────────────────────────────────────────
    try:
        t1 = time.time()
        exec_base = run_baseline_fold(pairs_df, trading_1min, formation_1min)
        row["baseline_pairs"] = len(exec_base)
        row["baseline_sharpe"] = _sharpe_from_exec(exec_base, trading_1min, pairs_df)
        print(f"  Baseline : {row['baseline_sharpe']:+.2f}  ({len(exec_base)} pairs, {time.time()-t1:.0f}s)")
    except Exception as e:
        print(f"  Baseline ERROR: {e}")
        row["baseline_sharpe"] = float("nan")
        exec_base = {}

    # ── Test A: Oracle Z ──────────────────────────────────────────────────────
    try:
        t1 = time.time()
        exec_oracle = run_oracle_fold(pairs_df, trading_1min)
        row["oracle_pairs"] = len(exec_oracle)
        row["oracle_sharpe"] = _sharpe_from_exec(exec_oracle, trading_1min, pairs_df)
        print(f"  Oracle Z : {row['oracle_sharpe']:+.2f}  ({len(exec_oracle)} pairs, {time.time()-t1:.0f}s)")
    except Exception as e:
        print(f"  Oracle ERROR: {e}")
        row["oracle_sharpe"] = float("nan")

    # ── Test B: Reverse signal ────────────────────────────────────────────────
    try:
        t1 = time.time()
        if exec_base:
            exec_rev = reverse_exec_results(exec_base)
            row["reverse_sharpe"] = _sharpe_from_exec(exec_rev, trading_1min, pairs_df)
        else:
            row["reverse_sharpe"] = float("nan")
        print(f"  Reverse  : {row['reverse_sharpe']:+.2f}  ({time.time()-t1:.0f}s)")
    except Exception as e:
        print(f"  Reverse ERROR: {e}")
        row["reverse_sharpe"] = float("nan")

    # ── Test C: Same-sector ───────────────────────────────────────────────────
    if SKIP_C:
        row["sector_n_pairs"] = "skipped"
        row["sector_sharpe"]  = float("nan")
    else:
        try:
            t1 = time.time()
            pairs_sector = filter_same_sector(pairs_df, sector_map)
            row["sector_n_pairs"] = len(pairs_sector)
            if len(pairs_sector) >= 2:
                exec_sec = run_baseline_fold(pairs_sector, trading_1min, formation_1min)
                row["sector_sharpe"] = _sharpe_from_exec(exec_sec, trading_1min, pairs_sector)
                print(f"  Sector   : {row['sector_sharpe']:+.2f}  ({len(pairs_sector)} same-sector pairs, {time.time()-t1:.0f}s)")
            else:
                row["sector_sharpe"] = float("nan")
                print(f"  Sector   : skipped (only {len(pairs_sector)} same-sector pairs)")
        except Exception as e:
            print(f"  Sector ERROR: {e}")
            row["sector_sharpe"] = float("nan")

    rows.append(row)
    print()

# ── Summary table ─────────────────────────────────────────────────────────────

results = pd.DataFrame(rows)

def fmt(v):
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:+.2f}"
    return "  n/a"

print("=" * 75)
print(f"{'Fold':<6} {'Month':<9} {'Regime':<5} {'Pairs':>5} "
      f"{'Baseline':>9} {'OracleZ':>8} {'Reverse':>8} {'Sector':>7}")
print("-" * 75)
for _, r in results.iterrows():
    sec = fmt(r.get("sector_sharpe", float("nan"))) if not SKIP_C else "  skip"
    print(f"{int(r['fold']):<6} {r['month']:<9} {r['regime']:<5} {int(r['n_pairs']):>5} "
          f"{fmt(r['baseline_sharpe']):>9} {fmt(r['oracle_sharpe']):>8} "
          f"{fmt(r['reverse_sharpe']):>8} {sec:>7}")
print("-" * 75)

def col_mean(col):
    vals = results[col].dropna()
    return float(vals.mean()) if len(vals) else float("nan")

print(f"{'MEAN':<6} {'':9} {'':5} {'':>5} "
      f"{fmt(col_mean('baseline_sharpe')):>9} {fmt(col_mean('oracle_sharpe')):>8} "
      f"{fmt(col_mean('reverse_sharpe')):>8} "
      f"{fmt(col_mean('sector_sharpe')) if not SKIP_C else '  skip':>7}")
print("=" * 75)

print("\nDiagnostic verdict:")
o = col_mean("oracle_sharpe")
r = col_mean("reverse_sharpe")
b = col_mean("baseline_sharpe")
s = col_mean("sector_sharpe") if not SKIP_C else float("nan")

if not np.isnan(o) and o > b + 2:
    print("  Test A POSITIVE: pairs CAN mean-revert within trading window.")
    print("  Problem = formation stats don't match trading distribution (non-persistence).")
    print("  Fix candidate: rolling formation stats or shorter fold gap embargo.")
elif not np.isnan(o) and o <= b + 1:
    print("  Test A NEGATIVE: pairs do NOT mean-revert even in-sample.")
    print("  The spread has no stationary tendency within the trading window.")

if not np.isnan(r) and r > b + 2:
    print("  Test B POSITIVE: reversing signal gives better Sharpe.")
    print("  These pairs exhibit momentum, not mean-reversion.")
    print("  Fix candidate: momentum entry strategy, or exclude these pairs.")
else:
    print("  Test B NEGATIVE: momentum direction also loses. No signal either way.")

if not np.isnan(s) and s > b + 2:
    print("  Test C POSITIVE: same-sector pairs significantly better.")
    print("  Fix candidate: add sector constraint to Phase 1 pair formation.")
else:
    print("  Test C NEGATIVE: sector constraint does not rescue the strategy.")

# Save
out_csv = f"{OUT_DIR}/diagnostic_results.csv"
results.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")
