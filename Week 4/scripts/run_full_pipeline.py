"""
Full 45-Fold Pipeline Runner — Phase 2 + Phase 3

Optimizations vs naive implementation:
  1. All ticker parquets loaded once into memory (dict cache), sliced per fold.
     Avoids 45 x N_tickers redundant parquet reads.
  2. Numba Kalman JIT compiled once at startup via warmup_kalman().
  3. select_delta capped at 200 representative pairs (fold 23 has 6,008 — without
     cap, delta selection alone takes ~150s per fold).
  4. Session warmup + EOS flatten vectorized (numpy int ops, no Python datetime loop).
  5. log_close precomputed at cache load time, not recomputed per pair.

Outputs:
  results/metrics/fold_metrics.csv        per-fold summary table
  results/metrics/equity_full.parquet     concatenated bar-level equity
  results/metrics/foldNN_equity.parquet   per-fold equity
  results/logs/fold_NN_audit.txt          per-fold audit logs
"""

import sys, os, logging, traceback, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RUN_PHASE1 = "--skip-phase1" not in sys.argv

# Numba warmup — compile JIT once before any fold runs
print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase2_execution.delta_selector import select_delta
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.phase3_backtest.neg_control import run_neg_control
from src.phase3_backtest.latency import run_latency_sweep
from src.phase3_backtest.audit_log import write_audit_log

# ---- Constants ----
PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"
RESULTS_DIR = "results/metrics"
LOG_DIR     = "results/logs"

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
ENTRY_Z          = 2.0
TC_BPS           = 30.0
BORROW_BPS_YR    = 50.0

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Fold schedule ----
FOLD_SCHEDULE = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (4,  "2022-04-01", "2022-09-30", "2022-10"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (7,  "2022-07-01", "2022-12-31", "2023-01"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (9,  "2022-09-01", "2023-02-28", "2023-03"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (14, "2023-02-01", "2023-07-31", "2023-08"),
    (15, "2023-03-01", "2023-08-31", "2023-09"),
    (16, "2023-04-01", "2023-09-30", "2023-10"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (18, "2023-06-01", "2023-11-30", "2023-12"),
    (19, "2023-07-01", "2023-12-31", "2024-01"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (21, "2023-09-01", "2024-02-29", "2024-03"),
    (22, "2023-10-01", "2024-03-31", "2024-04"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (24, "2023-12-01", "2024-05-31", "2024-06"),
    (25, "2024-01-01", "2024-06-30", "2024-07"),
    (26, "2024-02-01", "2024-07-31", "2024-08"),
    (27, "2024-03-01", "2024-08-31", "2024-09"),
    (28, "2024-04-01", "2024-09-30", "2024-10"),
    (29, "2024-05-01", "2024-10-31", "2024-11"),
    (30, "2024-06-01", "2024-11-30", "2024-12"),
    (31, "2024-07-01", "2024-12-31", "2025-01"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (33, "2024-09-01", "2025-02-28", "2025-03"),
    (34, "2024-10-01", "2025-03-31", "2025-04"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (42, "2025-06-01", "2025-11-30", "2025-12"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
    (45, "2025-09-01", "2026-02-28", "2026-03"),
]

# ============================================================
# Data cache — load each ticker once, slice per fold
# ============================================================

def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict[str, pd.DataFrame]:
    """
    Load all per-ticker parquets from data_dir into memory.
    Each ticker is read exactly once regardless of how many folds use it.
    log_close is precomputed here so callers never recompute per-pair.
    """
    cache = {}
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    for fname in files:
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_1min(cache: dict, tickers: set, trading_month: str) -> dict:
    # Parse once outside the loop; compare year/month ints to avoid tz-drop warning
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == month)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_5min(cache: dict, tickers: set, formation_start: str, formation_end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= formation_start) & (df.index <= formation_end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ============================================================
# Per-fold runner
# ============================================================

def run_fold(
    fold_n: int,
    formation_start: str,
    formation_end: str,
    trading_month: str,
    prev_delta,
    cache_1min: dict,
    cache_5min: dict,
) -> tuple[dict | None, pd.Series | None]:

    t0 = time.time()
    pairs_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except pd.errors.EmptyDataError:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs (empty CSV) — skipped")
        return None, None

    if len(pairs_df) == 0:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs — skipped")
        return None, None

    # Pair-count gate: cap spike folds (BH-FDR misfires during common-factor shocks)
    _MAX_PAIRS = 500
    if len(pairs_df) > _MAX_PAIRS:
        n_before = len(pairs_df)
        pairs_df = pairs_df.nsmallest(_MAX_PAIRS, "johansen_pval").reset_index(drop=True)
        print(f"  Fold {fold_n:02d}: pair-count gate applied → {_MAX_PAIRS} pairs (was {n_before})")

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())

    # Slice from in-memory cache — no parquet I/O
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_5min(cache_5min, tickers, formation_start, formation_end)
    # 1-min formation slice: same frequency as trading window, used for Z-score reference
    formation_1min = _slice_5min(cache_1min, tickers, formation_start, formation_end)

    if not trading_1min:
        print(f"  Fold {fold_n:02d} [{trading_month}]: no 1-min data — skipped")
        return None, None

    # Phase 2: delta selection (capped at 200 pairs inside select_delta)
    try:
        optimal_delta, delta_metrics = select_delta(pairs_df, formation_5min)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: select_delta FAILED: {e}")
        traceback.print_exc()
        return None, None

    if optimal_delta is None:
        # Spec: universal constraint fail → kick fold (no execution)
        print(f"  Fold {fold_n:02d} [{trading_month}]: universal delta fail — fold skipped")
        return None, None
    effective_delta = optimal_delta

    # Phase 2: execution engine
    try:
        fold_engine_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=effective_delta,
            entry_z=ENTRY_Z,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_execution FAILED: {e}")
        traceback.print_exc()
        return None, None

    if not fold_engine_results:
        print(f"  Fold {fold_n:02d} [{trading_month}]: engine returned 0 pairs")
        return None, None

    fold_config = {
        "fold": fold_n,
        "formation_start": formation_start,
        "formation_end":   formation_end,
        "trading_month":   trading_month,
        "delta":           optimal_delta,
        "entry_z":         ENTRY_Z,
        "tc_bps":          int(TC_BPS),
        "borrow_bps_yr":   int(BORROW_BPS_YR),
        "n_open_pairs_max": N_OPEN_PAIRS_MAX,
    }

    # Phase 3: PnL metrics
    try:
        fold_metrics = run_fold_pnl(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=effective_delta,
            delta_metrics=delta_metrics,
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_pnl FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Save per-pair trade metrics for volume stratification
    try:
        ppm = fold_metrics.get("per_pair_metrics", {})
        if ppm:
            rows = [
                {"fold": fold_n, "ticker_a": ta, "ticker_b": tb, **v}
                for (ta, tb), v in ppm.items()
            ]
            ppm_df = pd.DataFrame(rows)
            out_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
            write_header = not os.path.exists(out_path)
            ppm_df.to_csv(out_path, mode="a", header=write_header, index=False)
    except Exception:
        pass

    # Phase 3: negative control
    nc = None
    try:
        nc = run_neg_control(
            trading_1min=trading_1min,
            delta=effective_delta,
            primary_sharpe=fold_metrics["sharpe"],
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_neg_control WARNING: {e}")

    # Phase 3: latency sweep
    lat = None
    try:
        lat = run_latency_sweep(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_latency_sweep WARNING: {e}")

    # Phase 3: audit log
    try:
        write_audit_log(
            fold_n=fold_n,
            fold_metrics=fold_metrics,
            nc_metrics=nc,
            latency_results=lat,
            delta=optimal_delta,
            delta_metrics=delta_metrics,
            config=fold_config,
            prev_delta=prev_delta,
            output_dir=LOG_DIR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: write_audit_log WARNING: {e}")

    # Export per-fold equity
    eq = fold_metrics.get("bar_equity", pd.Series(dtype=float))
    if not eq.empty:
        eq.to_frame("equity").to_parquet(f"{RESULTS_DIR}/fold{fold_n:02d}_equity.parquet")

    elapsed = time.time() - t0
    sharpe   = fold_metrics["sharpe"]
    n_trades = fold_metrics["n_trades"]
    nc_pass  = nc["nc_pass"] if nc else None
    t5       = lat["sharpe_by_lag"].get("t+5") if lat else None
    delta_str = f"{optimal_delta:.0e}" if optimal_delta is not None else "None"

    nc_str = "PASS" if nc_pass else ("FAIL" if nc_pass is not None else "N/A")
    t5_str = f"{t5:+.3f}" if t5 is not None else "N/A"

    print(
        f"  Fold {fold_n:02d} [{trading_month}]  delta={delta_str}"
        f"  pairs={len(fold_engine_results)}  trades={n_trades}"
        f"  Sharpe={sharpe:+.3f}  MaxDD={fold_metrics['max_dd']:.3f}"
        f"  nc={nc_str}  t+5={t5_str}  [{elapsed:.0f}s]"
    )

    summary_row = {
        "fold":            fold_n,
        "trading_month":   trading_month,
        "n_pairs":         len(fold_engine_results),
        "n_trades":        n_trades,
        "delta":           optimal_delta,
        "sharpe":          sharpe,
        "max_dd":          fold_metrics["max_dd"],
        "cagr":            fold_metrics["cagr"],
        "calmar":          fold_metrics["calmar"],
        "win_rate":        fold_metrics["win_rate"],
        "avg_hold_bars":   fold_metrics["avg_holding_bars"],
        "avg_net_bps":     fold_metrics["avg_net_bps"],
        "cost_commission": fold_metrics["cost_decomp"]["commission"],
        "cost_borrow":     fold_metrics["cost_decomp"]["borrow"],
        "cost_rebalance":  fold_metrics["cost_decomp"]["rebalance"],
        "nc_threshold":    nc["bootstrap_threshold"] if nc else None,
        "nc_pass":         nc["nc_pass"] if nc else None,
        "t1_sharpe":       lat["sharpe_by_lag"].get("t+1") if lat else None,
        "t5_sharpe":       lat["sharpe_by_lag"].get("t+5") if lat else None,
        "t10_sharpe":      lat["sharpe_by_lag"].get("t+10") if lat else None,
        "latency_pass":    lat["latency_pass"] if lat else None,
        "lookahead_ok":    fold_metrics["lookahead_ok"],
        "kalman_degen":    fold_metrics["kalman_degenerate"],
        "elapsed_s":       elapsed,
    }
    return summary_row, eq


# ============================================================
# Main
# ============================================================

print("=== Full 45-Fold Pipeline Run — Phase 1 + Phase 2 + Phase 3 ===\n" if RUN_PHASE1
      else "=== Full 45-Fold Pipeline Run — Phase 2 + Phase 3 (Phase 1 skipped) ===\n")

# ============================================================
# Phase 1 — Cointegration Discovery (optional, --run-phase1)
# ============================================================

if RUN_PHASE1:
    from src.phase1_cointegration.discovery import run as run_phase1
    from src.utils.io import VALIDATED_DIR

    os.makedirs(PHASE1_DIR, exist_ok=True)
    print("=== Phase 1: Cointegration Discovery ===")
    t_p1 = time.time()
    for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
        out_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
        print(f"  Fold {fold_n:02d} [{trading_month}]: formation {formation_start} → {formation_end} ...", end=" ", flush=True)
        try:
            pairs_df = run_phase1(formation_start, formation_end, VALIDATED_DIR)
            pairs_df.to_csv(out_csv, index=False)
            print(f"{len(pairs_df)} pairs")
        except Exception as e:
            print(f"ERROR: {e}")
            pd.DataFrame().to_csv(out_csv, index=False)
    print(f"Phase 1 complete: {time.time()-t_p1:.0f}s\n")

# Load all ticker data once
print(f"Loading 1-min data cache from {DATA_1MIN}/ ...")
t_load = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
print(f"  {len(cache_1min)} tickers loaded in {time.time()-t_load:.1f}s")

print(f"Loading 5-min data cache from {DATA_5MIN}/ ...")
t_load = time.time()
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} tickers loaded in {time.time()-t_load:.1f}s\n")

t_total  = time.time()
all_rows = []
all_eq   = []
prev_delta = None

# Clear per-pair metrics file before run to prevent duplicate rows on re-run
_ppm_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
if os.path.exists(_ppm_path):
    os.remove(_ppm_path)

for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
    try:
        row, eq = run_fold(
            fold_n, formation_start, formation_end, trading_month,
            prev_delta, cache_1min, cache_5min,
        )
        if row is not None:
            all_rows.append(row)
            prev_delta = row["delta"]
            if eq is not None and not eq.empty:
                all_eq.append(eq)
        else:
            prev_delta = None
    except Exception as e:
        print(f"  Fold {fold_n:02d}: UNHANDLED ERROR: {e}")
        traceback.print_exc()
        prev_delta = None

# ---- Save outputs ----
summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(f"{RESULTS_DIR}/fold_metrics.csv", index=False)

if all_eq:
    pd.concat(all_eq).sort_index().to_frame("equity").to_parquet(
        f"{RESULTS_DIR}/equity_full.parquet"
    )

total_min = (time.time() - t_total) / 60
print(f"\n=== Done in {total_min:.1f} min ===")
print(f"  Folds completed : {len(all_rows)} / 45")
print(f"  Results CSV     : {RESULTS_DIR}/fold_metrics.csv")

# ---- Aggregate report ----
if len(all_rows) > 0:
    df = summary_df
    print(f"\n{'='*55}")
    print(f"  AGGREGATE STATISTICS ({len(df)} completed folds)")
    print(f"{'='*55}")
    print(f"  Sharpe  : mean={df['sharpe'].mean():+.3f}  median={df['sharpe'].median():+.3f}"
          f"  std={df['sharpe'].std():.3f}  min={df['sharpe'].min():+.3f}  max={df['sharpe'].max():+.3f}")
    print(f"  % pos Sharpe folds : {(df['sharpe'] > 0).mean():.1%}")
    print(f"  MaxDD   : mean={df['max_dd'].mean():.4f}  worst={df['max_dd'].min():.4f}")
    print(f"  CAGR    : mean={df['cagr'].mean():+.4f}")
    print(f"  Calmar  : mean={df['calmar'].mean():+.3f}")
    print(f"  Win rate: mean={df['win_rate'].mean():.1%}")
    print(f"  Trades  : total={df['n_trades'].sum():.0f}  mean/fold={df['n_trades'].mean():.0f}")
    print(f"  Commission total : ${df['cost_commission'].sum():,.0f}")
    print(f"  Borrow total     : ${df['cost_borrow'].sum():,.0f}")
    print(f"  Rebalance total  : ${df['cost_rebalance'].sum():,.0f}")
    print(f"  NC pass rate     : {df['nc_pass'].mean():.1%}")
    print(f"  Latency t+5 > 0  : {(df['t5_sharpe'].dropna() > 0).mean():.1%}")
    print(f"  Lookahead ok (all): {bool(df['lookahead_ok'].all())}")
    print(f"  Kalman degen flags: {int(df['kalman_degen'].sum())}")
    print(f"  Delta=1e-7 count  : {int((df['delta'] == 1e-7).sum())} / {len(df)}")

    print(f"\n  --- By Regime ---")
    regimes = [
        ("Bear 2022        ", df["fold"].between(1, 6)),
        ("Early Bull 2023  ", df["fold"].between(7, 18)),
        ("Mid Bull 2024    ", df["fold"].between(19, 30)),
        ("Late Bull 2025-26", df["fold"].between(31, 45)),
    ]
    for name, mask in regimes:
        sub = df[mask]
        if len(sub) == 0:
            print(f"  {name}: no completed folds")
            continue
        print(
            f"  {name}  N={len(sub):2d}  "
            f"Sharpe mean={sub['sharpe'].mean():+.3f}  "
            f"pct_pos={( sub['sharpe'] > 0).mean():.0%}  "
            f"trades={int(sub['n_trades'].sum()):5d}  "
            f"nc_pass={sub['nc_pass'].mean():.0%}"
        )

    print(f"\n  --- Per-Fold Sharpe ---")
    for _, r in df.iterrows():
        bar = "+" * max(0, int(r["sharpe"])) if r["sharpe"] > 0 else "-" * min(20, int(abs(r["sharpe"])))
        print(f"  Fold {int(r['fold']):02d} [{r['trading_month']}]  "
              f"Sharpe={r['sharpe']:+6.3f}  trades={int(r['n_trades']):4d}  {bar}")
