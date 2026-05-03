# ======================================================================
# Week 4 Quant Finance — Diagnostics & Analytics Runners
# ======================================================================


# ===== FILE: scripts/run_phase1_all_folds.py =====
"""
Run Phase 1 for all 45 folds — parallel fold execution.

Root cause of prior OOM: each worker cached _log_a/_log_b for every pair
(~7 GB/worker with 44k pairs). Fix: discovery.py now stores only scalars;
FDR survivors are re-aligned cheaply. Each worker now uses ~100 MB.

4 workers is safe on 8 GB RAM. 6 workers fine on 16 GB+.

Usage:
    python run_phase1_all_folds.py             # 4 parallel workers (default)
    python run_phase1_all_folds.py --workers 6
    python run_phase1_all_folds.py --resume    # skip folds already saved
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys, logging, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Path("results/metrics/phase1_folds").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/phase1_all_folds.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("phase1_all")

# ---------------------------------------------------------------------------
# Fold schedule
# ---------------------------------------------------------------------------
DATA_START = "2022-01-03"
trading_months = pd.date_range("2022-07-01", "2026-03-01", freq="MS")  # 45 months

FOLDS = []
for i, trade_start in enumerate(trading_months, 1):
    form_end       = trade_start - pd.Timedelta(days=1)
    form_start_raw = trade_start - pd.DateOffset(months=6)
    form_start     = max(form_start_raw, pd.Timestamp(DATA_START))
    FOLDS.append({
        "fold":             i,
        "formation_start":  form_start.strftime("%Y-%m-%d"),
        "formation_end":    form_end.strftime("%Y-%m-%d"),
        "trading_month":    trade_start.strftime("%Y-%m"),
    })


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------
def _run_fold(fold: dict) -> dict:
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from src.phase1_cointegration.discovery import run as run_phase1

    n          = fold["fold"]
    form_start = fold["formation_start"]
    form_end   = fold["formation_end"]
    trade_mon  = fold["trading_month"]
    out_path   = f"results/metrics/phase1_folds/fold_{n:02d}.csv"

    t0 = time.time()
    try:
        pairs = run_phase1(form_start, form_end)
        error = None
    except Exception as exc:
        pairs = None
        error = str(exc)

    elapsed = round(time.time() - t0, 1)
    n_pairs = len(pairs) if pairs is not None and len(pairs) > 0 else 0

    if pairs is not None and n_pairs > 0:
        pairs["fold"]            = n
        pairs["formation_start"] = form_start
        pairs["formation_end"]   = form_end
        pairs["trading_month"]   = trade_mon
        pairs.to_csv(out_path, index=False)
    else:
        import pandas as pd
        pd.DataFrame().to_csv(out_path, index=False)

    return {
        "fold":              n,
        "formation_start":   form_start,
        "formation_end":     form_end,
        "trading_month":     trade_mon,
        "n_surviving_pairs": n_pairs,
        "elapsed_seconds":   elapsed,
        "error":             error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4; safe on 8 GB RAM)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip folds whose fold_XX.csv already exists and has data")
    args = parser.parse_args()
    n_workers = args.workers

    # Resume: skip folds already successfully completed
    folds_to_run = []
    skipped = 0
    for fold in FOLDS:
        out = Path(f"results/metrics/phase1_folds/fold_{fold['fold']:02d}.csv")
        if args.resume and out.exists() and out.stat().st_size > 10:
            skipped += 1
        else:
            folds_to_run.append(fold)

    log.info("=" * 60)
    log.info("PHASE 1 -- ALL 45 FOLDS | %d parallel workers", n_workers)
    log.info("Folds to run: %d  |  Skipped (resume): %d", len(folds_to_run), skipped)
    log.info("Fold 1:  Formation %s to %s", FOLDS[0]["formation_start"], FOLDS[0]["formation_end"])
    log.info("Fold 45: Formation %s to %s", FOLDS[-1]["formation_start"], FOLDS[-1]["formation_end"])
    log.info("Estimated: ~%.0f hours (~12 min/fold / %d workers)",
             len(folds_to_run) * 12 / 60 / n_workers, n_workers)
    log.info("=" * 60)

    t_total = time.time()
    results = {}
    # Pre-fill skipped folds from existing CSVs
    for fold in FOLDS:
        out = Path(f"results/metrics/phase1_folds/fold_{fold['fold']:02d}.csv")
        if fold not in folds_to_run and out.exists():
            try:
                df = pd.read_csv(out)
                results[fold["fold"]] = {
                    "fold": fold["fold"],
                    "formation_start": fold["formation_start"],
                    "formation_end": fold["formation_end"],
                    "trading_month": fold["trading_month"],
                    "n_surviving_pairs": len(df),
                    "elapsed_seconds": 0,
                    "error": None,
                }
            except Exception:
                pass

    completed = skipped
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_fold, fold): fold["fold"] for fold in folds_to_run}

        for future in as_completed(futures):
            fold_n = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "fold": fold_n, "n_surviving_pairs": 0,
                    "elapsed_seconds": 0, "error": str(exc),
                    "trading_month": FOLDS[fold_n - 1]["trading_month"],
                    "formation_start": FOLDS[fold_n - 1]["formation_start"],
                    "formation_end":   FOLDS[fold_n - 1]["formation_end"],
                }

            results[fold_n] = result
            completed += 1

            status = "ERROR" if result.get("error") else "OK"
            log.info(
                "[%2d/45] Fold %02d (%s): %3d pairs | %.0fs | %s",
                completed, result["fold"], result["trading_month"],
                result["n_surviving_pairs"], result["elapsed_seconds"], status,
            )
            if result.get("error"):
                log.error("  Fold %02d error: %s", result["fold"], result["error"])

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    all_results = [results[f["fold"]] for f in FOLDS if f["fold"] in results]
    summary_df = pd.DataFrame(all_results).sort_values("fold").reset_index(drop=True)
    summary_df.to_csv("results/metrics/phase1_all_folds.csv", index=False)

    total_min = (time.time() - t_total) / 60
    log.info("")
    log.info("=" * 60)
    log.info("ALL FOLDS COMPLETE | %.1f minutes total", total_min)
    log.info("=" * 60)
    log.info("Folds with pairs : %d / 45", (summary_df.n_surviving_pairs > 0).sum())
    log.info("Folds with 0 pairs: %d / 45", (summary_df.n_surviving_pairs == 0).sum())
    log.info("Total pairs (all folds): %d", summary_df.n_surviving_pairs.sum())
    if (summary_df.n_surviving_pairs > 0).any():
        log.info("Max pairs in one fold  : %d (Fold %d)",
                 summary_df.n_surviving_pairs.max(),
                 int(summary_df.loc[summary_df.n_surviving_pairs.idxmax(), "fold"]))
    log.info("")
    log.info("Per-fold results:")
    for _, row in summary_df.iterrows():
        bar = "#" * min(int(row.n_surviving_pairs), 60)
        log.info("  Fold %02d (%s): %3d pairs  %s",
                 int(row.fold), row.trading_month,
                 int(row.n_surviving_pairs), bar)
    log.info("")
    log.info("Saved: results/metrics/phase1_all_folds.csv")
    log.info("       results/metrics/phase1_folds/fold_XX.csv")



# ===== FILE: scripts/run_full_pipeline.py =====
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



# ===== FILE: run_phase4.py =====
"""
Phase 4 — Multi-Regime Defense Runner

Runs all analytical Phase 4 modules on existing pipeline results
and produces the 12 whitepaper output files.

Usage:
    python run_phase4.py                          # analytical only (fast ~2 min)
    python run_phase4.py --persistence            # + pair persistence (slow ~30 min)
    python run_phase4.py --volume-strat           # + volume stratification (~15 min)
    python run_phase4.py --structural-oat         # + structural OAT (~20 min)
    python run_phase4.py --all                    # everything (~60 min)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


def main(args: argparse.Namespace) -> None:
    from src.phase4_defense.orchestrator import (
        load_fold_metrics,
        print_fold_summary,
        _METRICS_DIR,
        _FIGURES_DIR,
    )

    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 4 — Multi-Regime Defense Analysis")
    log.info("=" * 60)

    # ── Load existing results ──────────────────────────────────────
    fold_metrics = load_fold_metrics()
    log.info("Loaded %d completed folds from fold_metrics.csv", len(fold_metrics))
    print_fold_summary(fold_metrics)

    # ── §1: 45-fold Sharpe distribution ───────────────────────────
    log.info("\n[§1] Sharpe distribution")
    _run_sharpe_distribution(fold_metrics)

    # ── §2: Regime partition ───────────────────────────────────────
    log.info("\n[§2] Regime partition")
    from src.phase4_defense.regime import run_regime_analysis, print_regime_report
    regime_df = run_regime_analysis(fold_metrics, save=True)
    print_regime_report(regime_df)

    # ── §5: Overfitting diagnostics ────────────────────────────────
    log.info("\n[§5] Overfitting diagnostics (DSR + PBO)")
    from src.phase4_defense.overfitting import run_overfitting_diagnostics, print_overfitting_report
    overfit_df = run_overfitting_diagnostics(fold_metrics, save=True)
    print_overfitting_report(overfit_df)

    # ── §6: OAT sensitivity (analytical) ──────────────────────────
    log.info("\n[§6] OAT sensitivity (analytical)")
    from src.phase4_defense.sensitivity import (
        run_oat_sensitivity, compute_exit_reasons, print_oat_report
    )
    oat_df = run_oat_sensitivity(
        fold_metrics,
        run_structural=args.structural_oat or args.all,
        structural_max_folds=10,
        save=True,
    )
    print_oat_report(oat_df)

    # ── §9: Exit reason breakdown ──────────────────────────────────
    log.info("\n[§9] Exit reason breakdown")
    exit_df = compute_exit_reasons()
    if not exit_df.empty:
        _print_exit_summary(exit_df)

    # ── §10: Cost decomposition ────────────────────────────────────
    log.info("\n[§10] Cost decomposition")
    _run_cost_decomposition(fold_metrics)

    # ── §11: Delta trajectory ──────────────────────────────────────
    log.info("\n[§11] Delta trajectory")
    _run_delta_trajectory(fold_metrics)

    # ── §12: Universe counts ───────────────────────────────────────
    log.info("\n[§12] Universe counts")
    _run_universe_counts()

    # ── §7 + §8: NC + latency (already in fold_metrics.csv) ───────
    log.info("\n[§7+8] NC bootstrap + latency — extracting from fold_metrics")
    _extract_nc_latency(fold_metrics)

    # ── §3: Pair persistence (optional — slow) ─────────────────────
    if args.persistence or args.all:
        log.info("\n[§3] Pair persistence (re-running Johansen, ~30 min)")
        from src.phase4_defense.persistence import run_persistence
        persist_df = run_persistence(save=True)
        log.info("Persistence complete: %d fold points", len(persist_df))
    else:
        log.info("\n[§3] Pair persistence skipped (pass --persistence to run)")

    # ── §4: Volume stratification (optional — slower) ─────────────
    if args.volume_strat or args.all:
        log.info("\n[§4] Volume stratification (~15 min)")
        from src.phase4_defense.volume_strat import run_volume_strat, print_volume_report
        vol_fold, vol_pairs = run_volume_strat(save=True)
        print_volume_report(vol_fold, vol_pairs)
    else:
        log.info("\n[§4] Volume stratification skipped (pass --volume-strat to run)")

    # ── Final checklist ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Phase 4 complete. Checking whitepaper output files:")
    _check_outputs()


# ---------------------------------------------------------------------------
# Helper runners for §1, §10-12, §7-8
# ---------------------------------------------------------------------------

def _run_sharpe_distribution(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    dist = fold_metrics[["fold", "trading_month", "sharpe", "max_dd", "cagr", "calmar"]].copy()
    dist.to_csv(_METRICS_DIR / "sharpe_distribution.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.hist(fold_metrics["sharpe"], bins=15, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(fold_metrics["sharpe"].median(), color="red",
                   linewidth=1.5, linestyle="--", label=f"Median={fold_metrics['sharpe'].median():.2f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Fold Count")
        ax.set_title(f"45-Fold Sharpe Distribution\n"
                     f"(N={len(fold_metrics)}, mean={fold_metrics['sharpe'].mean():.2f}, "
                     f"% pos={( fold_metrics['sharpe']>0).mean():.0%})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        months = fold_metrics["trading_month"].tolist()
        x = range(len(months))
        ax2.bar(x, fold_metrics["sharpe"], color=[
            "#2ca02c" if s > 0 else "#d62728" for s in fold_metrics["sharpe"]
        ], alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(list(x)[::6])
        ax2.set_xticklabels(months[::6], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe per Fold (chronological)")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "sharpe_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _run_cost_decomposition(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR

    cols = ["fold", "trading_month", "sharpe",
            "cost_commission", "cost_borrow", "cost_rebalance", "n_trades"]
    cost = fold_metrics[[c for c in cols if c in fold_metrics.columns]].copy()
    cost["cost_total"] = (
        cost.get("cost_commission", 0)
        + cost.get("cost_borrow", 0)
        + cost.get("cost_rebalance", 0)
    )
    cost.to_csv(_METRICS_DIR / "cost_decomp.csv", index=False)

    # Print summary
    print("\n  Cost Decomposition ($ totals across all folds):")
    total_capital = 1_000_000.0
    for col in ["cost_commission", "cost_borrow", "cost_rebalance", "cost_total"]:
        if col in cost:
            tot = cost[col].sum()
            print(f"    {col:<20}: ${tot:>15,.0f}  ({tot/total_capital/len(cost)*100:.1f} bps/fold/capital)")


def _run_delta_trajectory(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    delta_df = fold_metrics[["fold", "trading_month", "delta"]].copy()
    delta_df.to_csv(_METRICS_DIR / "delta_trajectory.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.semilogy(range(len(delta_df)), delta_df["delta"], "o-",
                    markersize=4, linewidth=1.5, color="#1f77b4")
        ax.set_xticks(range(0, len(delta_df), max(1, len(delta_df) // 8)))
        ax.set_xticklabels(
            delta_df["trading_month"].iloc[::max(1, len(delta_df) // 8)],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("Selected δ (log scale)")
        ax.set_title("Kalman δ Trajectory Across Folds")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "delta_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _run_universe_counts() -> None:
    from src.phase4_defense.orchestrator import (
        FOLD_SCHEDULE, load_phase1_pairs, _METRICS_DIR
    )

    rows = []
    for spec in FOLD_SCHEDULE:
        pairs_df = load_phase1_pairs(spec.fold_n)
        n_pairs = len(pairs_df) if pairs_df is not None else 0
        if pairs_df is not None and not pairs_df.empty:
            n_tickers = len(set(
                pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
            ))
        else:
            n_tickers = 0

        rows.append({
            "fold":          spec.fold_n,
            "trading_month": spec.trading_month,
            "form_start":    spec.form_start,
            "form_end":      spec.form_end,
            "n_surviving_pairs": n_pairs,
            "n_unique_tickers":  n_tickers,
        })

    result = pd.DataFrame(rows)
    result.to_csv(_METRICS_DIR / "universe_counts.csv", index=False)
    log.info("Universe counts saved (%d folds)", len(result))
    print(f"\n  Universe counts: median pairs/fold = {result['n_surviving_pairs'].median():.0f}, "
          f"max = {result['n_surviving_pairs'].max()}")


def _extract_nc_latency(fold_metrics: pd.DataFrame) -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    # NC bootstrap (§7)
    nc_cols = ["fold", "trading_month", "nc_threshold", "nc_pass", "sharpe"]
    nc_df = fold_metrics[[c for c in nc_cols if c in fold_metrics.columns]].copy()
    nc_df.to_csv(_METRICS_DIR / "nc_bootstrap.csv", index=False)
    n_pass = nc_df["nc_pass"].sum() if "nc_pass" in nc_df.columns else 0
    print(f"\n  NC bootstrap: {n_pass}/{len(nc_df)} folds pass (Primary > NC threshold)")

    # Latency (§8)
    lat_cols = ["fold", "trading_month", "t1_sharpe", "t5_sharpe", "t10_sharpe", "latency_pass"]
    lat_df = fold_metrics[[c for c in lat_cols if c in fold_metrics.columns]].copy()
    lat_df.to_csv(_METRICS_DIR / "latency_decay.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for col, lag, color in [
            ("t1_sharpe", "t+1", "#2ca02c"),
            ("t5_sharpe", "t+5", "#ff7f0e"),
            ("t10_sharpe","t+10","#d62728"),
        ]:
            if col in lat_df.columns:
                ax.plot(range(len(lat_df)), lat_df[col], "o-",
                        markersize=3, linewidth=1.2, alpha=0.8,
                        label=f"Lag {lag} (mean={lat_df[col].mean():.2f})",
                        color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Sharpe")
        ax.set_title("Latency Sweep: Alpha Decay vs Execution Lag")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FIGURES_DIR / "latency_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _print_exit_summary(exit_df: pd.DataFrame) -> None:
    tot = exit_df["n_trades_parsed"].sum()
    eos = exit_df["n_eos"].sum()
    zc  = exit_df["n_zero_cross"].sum()
    sl  = exit_df.get("n_sl", pd.Series(0)).sum()

    print(f"\n  Exit reasons (aggregate across all folds):")
    print(f"    EOS exits:       {int(eos):6d}  ({eos/tot:.1%})")
    print(f"    Zero-cross:      {int(zc):6d}  ({zc/tot:.1%})")
    if sl > 0:
        print(f"    Stop-loss:       {int(sl):6d}  ({sl/tot:.1%})")
    print(f"    Avg net EOS:     {exit_df['avg_net_eos'].mean():.1f} bps")
    print(f"    Avg net ZC:      {exit_df['avg_net_zc'].mean():.1f} bps")


def _check_outputs() -> None:
    from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR

    required_metrics = [
        "sharpe_distribution.csv",   # §1
        "regime_sharpes.csv",         # §2
        "pair_persistence.csv",       # §3 (optional)
        "volume_strat.csv",           # §4 (optional)
        "overfitting_diagnostics.csv",# §5
        "oat_sensitivity.csv",        # §6
        "nc_bootstrap.csv",           # §7
        "latency_decay.csv",          # §8
        "exit_reasons.csv",           # §9
        "cost_decomp.csv",            # §10
        "delta_trajectory.csv",       # §11
        "universe_counts.csv",        # §12
    ]
    required_figures = [
        "sharpe_hist.png",
        "regime_bar.png",
        "delta_traj.png",
        "latency_curve.png",
        "oat_grid.png",
    ]

    print()
    all_ok = True
    for fname in required_metrics:
        path = _METRICS_DIR / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/metrics/{fname:<35} {status}")

    for fname in required_figures:
        path = _FIGURES_DIR / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/figures/{fname:<35} {status}")

    print()
    if all_ok:
        log.info("All whitepaper outputs present.")
    else:
        log.warning("Some outputs missing. Run with --persistence --volume-strat for §3/§4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Defense Analysis")
    parser.add_argument("--persistence",     action="store_true",
                        help="Run P_2022 pair persistence (~30 min)")
    parser.add_argument("--volume-strat",    action="store_true",
                        help="Run volume stratification (~15 min)")
    parser.add_argument("--structural-oat",  action="store_true",
                        help="Run structural OAT variations (~20 min)")
    parser.add_argument("--all",             action="store_true",
                        help="Run all analyses including slow ones")
    args = parser.parse_args()
    main(args)



# ===== FILE: run_phase4_final.py =====
"""
run_phase4_final.py — Phase 4 Analytics on Option A + CORR25 Final Pipeline Results

Reads:  results/metrics/final/fold_metrics.csv  (23 completed folds)
        results/logs/final/fold_NN_audit.txt

Writes: results/metrics/final/phase4/  — all §1/§2/§5/§6/§7/§8/§9/§10/§11 CSVs
        results/figures/final/          — all figures

Does NOT overwrite:
  results/metrics/   (baseline Phase 3 outputs)
  results/figures/   (baseline Phase 4 figures)

Skipped (config-independent, unchanged from baseline run):
  §3  pair persistence   — re-run with --persistence if needed
  §4  volume strat       — re-run with --volume-strat if needed
  §12 universe counts    — same Phase 1 pairs regardless of downstream filters

Usage:
    python run_phase4_final.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── 1. Add src to path (must happen before any src import) ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4_final")

# ── 2. Monkey-patch orchestrator BEFORE importing sub-modules ───────────────
#
# Phase 4 sub-modules (regime.py, overfitting.py, sensitivity.py) do:
#   from src.phase4_defense.orchestrator import _METRICS_DIR, _FIGURES_DIR, ...
# at module-level on import.  If we patch orchestrator FIRST and import
# sub-modules AFTER, their module-level bindings pick up the patched paths.

import src.phase4_defense.orchestrator as _orch  # noqa: E402

_FINAL_METRICS = Path("results/metrics/final/phase4")
_FINAL_FIGURES = Path("results/figures/final")
_FINAL_LOGS    = Path("results/logs/final")
_FINAL_SRC_CSV = Path("results/metrics/final/fold_metrics.csv")

# Override module-level path attributes
_orch._METRICS_DIR = _FINAL_METRICS
_orch._FIGURES_DIR = _FINAL_FIGURES
_orch._LOGS_DIR    = _FINAL_LOGS

# Create output directories before sub-module imports (they call mkdir at import time)
_FINAL_METRICS.mkdir(parents=True, exist_ok=True)
_FINAL_FIGURES.mkdir(parents=True, exist_ok=True)


def _load_final_fold_metrics() -> pd.DataFrame:
    """Load fold_metrics from final pipeline run and attach regime labels."""
    df = pd.read_csv(_FINAL_SRC_CSV)
    df["regime"] = df["fold"].map(_orch.FOLD_TO_REGIME)
    return df


# Replace the orchestrator's loader so sub-modules calling load_fold_metrics()
# with no arguments get the final-pipeline data automatically.
_orch.load_fold_metrics = _load_final_fold_metrics

# Patch load_equity to read from results/metrics/final/ (not the phase4/ subdir).
# DSR in overfitting.py calls get_daily_returns -> load_equity, which uses
# _METRICS_DIR / "fold{N:02d}_equity.parquet".  Since _METRICS_DIR is now the
# phase4/ subdir, we override load_equity to point at the actual equity location.
_FINAL_EQUITY_DIR = Path("results/metrics/final")

def _load_final_equity(fold_n: int):
    path = _FINAL_EQUITY_DIR / f"fold{fold_n:02d}_equity.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

_orch.load_equity = _load_final_equity

# ── 3. NOW import sub-modules (they see patched _METRICS_DIR / _FIGURES_DIR) ─
from src.phase4_defense.regime import run_regime_analysis, print_regime_report  # noqa: E402
from src.phase4_defense.overfitting import (  # noqa: E402
    run_overfitting_diagnostics,
    print_overfitting_report,
)
from src.phase4_defense.sensitivity import (  # noqa: E402
    run_oat_sensitivity,
    compute_exit_reasons,
    print_oat_report,
)


# ── Helper runners (same logic as run_phase4.py, paths from patched orchestrator)

def _run_sharpe_distribution(fold_metrics: pd.DataFrame) -> None:
    dist = fold_metrics[["fold", "trading_month", "sharpe", "max_dd", "cagr", "calmar"]].copy()
    dist.to_csv(_FINAL_METRICS / "sharpe_distribution.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.hist(fold_metrics["sharpe"], bins=15, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(fold_metrics["sharpe"].median(), color="red",
                   linewidth=1.5, linestyle="--",
                   label=f"Median={fold_metrics['sharpe'].median():.2f}")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Fold Count")
        ax.set_title(f"45-Fold Sharpe Distribution (Option A+CORR25)\n"
                     f"N={len(fold_metrics)}, mean={fold_metrics['sharpe'].mean():.2f}, "
                     f"% pos={(fold_metrics['sharpe'] > 0).mean():.0%}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        months = fold_metrics["trading_month"].tolist()
        x = range(len(months))
        ax2.bar(x, fold_metrics["sharpe"],
                color=["#2ca02c" if s > 0 else "#d62728" for s in fold_metrics["sharpe"]],
                alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(list(x)[::4])
        ax2.set_xticklabels(months[::4], rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe per Fold (chronological) — Option A+CORR25")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "sharpe_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Sharpe distribution figure saved → %s", _FINAL_FIGURES / "sharpe_hist.png")
    except ImportError:
        log.warning("matplotlib not available — skipping figure")


def _run_cost_decomposition(fold_metrics: pd.DataFrame) -> None:
    cols = ["fold", "trading_month", "sharpe",
            "cost_commission", "cost_borrow", "cost_rebalance", "n_trades"]
    cost = fold_metrics[[c for c in cols if c in fold_metrics.columns]].copy()
    cost["cost_total"] = (
        cost.get("cost_commission", 0)
        + cost.get("cost_borrow", 0)
        + cost.get("cost_rebalance", 0)
    )
    cost.to_csv(_FINAL_METRICS / "cost_decomp.csv", index=False)

    total_capital = 1_000_000.0
    print("\n  Cost Decomposition ($ totals, 23 completed folds):")
    for col in ["cost_commission", "cost_borrow", "cost_rebalance", "cost_total"]:
        if col in cost:
            tot = cost[col].sum()
            print(f"    {col:<20}: ${tot:>12,.0f}  "
                  f"({tot / total_capital / len(cost) * 100:.1f} bps/fold/capital)")


def _run_delta_trajectory(fold_metrics: pd.DataFrame) -> None:
    delta_df = fold_metrics[["fold", "trading_month", "delta"]].copy()
    delta_df.to_csv(_FINAL_METRICS / "delta_trajectory.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.semilogy(range(len(delta_df)), delta_df["delta"], "o-",
                    markersize=4, linewidth=1.5, color="#1f77b4")
        ax.set_xticks(range(0, len(delta_df), max(1, len(delta_df) // 8)))
        ax.set_xticklabels(
            delta_df["trading_month"].iloc[::max(1, len(delta_df) // 8)].tolist(),
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("Selected δ (log scale)")
        ax.set_title("Kalman δ Trajectory — Option A+CORR25 (fixed δ=1e-7)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "delta_traj.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _extract_nc_latency(fold_metrics: pd.DataFrame) -> None:
    # NC bootstrap (§7)
    nc_cols = ["fold", "trading_month", "nc_threshold", "nc_pass", "sharpe"]
    nc_df = fold_metrics[[c for c in nc_cols if c in fold_metrics.columns]].copy()
    nc_df.to_csv(_FINAL_METRICS / "nc_bootstrap.csv", index=False)
    n_pass = int(nc_df["nc_pass"].sum()) if "nc_pass" in nc_df.columns else 0
    n_total = int(nc_df["nc_pass"].notna().sum()) if "nc_pass" in nc_df.columns else len(nc_df)
    print(f"\n  NC bootstrap: {n_pass}/{n_total} folds pass  "
          f"(Primary > NC threshold at same eos_flatten=False)")

    # Latency (§8)
    lat_cols = ["fold", "trading_month", "t1_sharpe", "t5_sharpe", "t10_sharpe", "latency_pass"]
    lat_df = fold_metrics[[c for c in lat_cols if c in fold_metrics.columns]].copy()
    lat_df.to_csv(_FINAL_METRICS / "latency_decay.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for col, lag, color in [
            ("t1_sharpe",  "t+1",  "#2ca02c"),
            ("t5_sharpe",  "t+5",  "#ff7f0e"),
            ("t10_sharpe", "t+10", "#d62728"),
        ]:
            if col in lat_df.columns:
                valid = lat_df[col].dropna()
                ax.plot(range(len(valid)), valid.values, "o-",
                        markersize=3, linewidth=1.2, alpha=0.8,
                        label=f"Lag {lag} (mean={valid.mean():.2f})",
                        color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fold (completed only)")
        ax.set_ylabel("Sharpe")
        ax.set_title("Latency Sweep — Option A+CORR25")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(_FINAL_FIGURES / "latency_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass


def _print_exit_summary(exit_df: pd.DataFrame) -> None:
    tot = exit_df["n_trades_parsed"].sum()
    if tot == 0:
        print("  No trade records parsed from audit logs.")
        return
    eos = exit_df["n_eos"].sum()
    zc  = exit_df["n_zero_cross"].sum()
    sl  = exit_df.get("n_sl", pd.Series(0, index=exit_df.index)).sum()

    print("\n  Exit reasons (aggregate, Option A+CORR25):")
    print(f"    EOS exits:       {int(eos):6d}  ({eos / tot:.1%})")
    print(f"    Zero-cross:      {int(zc):6d}  ({zc / tot:.1%})")
    if sl > 0:
        print(f"    Stop-loss:       {int(sl):6d}  ({sl / tot:.1%})")
    if "avg_net_eos" in exit_df.columns:
        print(f"    Avg net EOS:     {exit_df['avg_net_eos'].mean():.1f} bps")
    if "avg_net_zc" in exit_df.columns:
        print(f"    Avg net ZC:      {exit_df['avg_net_zc'].mean():.1f} bps")


def _check_outputs() -> None:
    required_metrics = [
        "sharpe_distribution.csv",    # §1
        "regime_sharpes.csv",          # §2
        "overfitting_diagnostics.csv", # §5
        "oat_sensitivity.csv",         # §6
        "nc_bootstrap.csv",            # §7
        "latency_decay.csv",           # §8
        "exit_reasons.csv",            # §9
        "cost_decomp.csv",             # §10
        "delta_trajectory.csv",        # §11
    ]
    required_figures = [
        "sharpe_hist.png",
        "regime_bar.png",
        "delta_traj.png",
        "latency_curve.png",
        "oat_grid.png",
    ]

    print()
    all_ok = True
    for fname in required_metrics:
        path = _FINAL_METRICS / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/metrics/final/phase4/{fname:<35} {status}")

    for fname in required_figures:
        path = _FINAL_FIGURES / fname
        status = "[OK]" if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  results/figures/final/{fname:<40} {status}")

    print()
    skipped = ["§3 pair_persistence.csv", "§4 volume_strat.csv", "§12 universe_counts.csv"]
    print(f"  Skipped (config-independent): {', '.join(skipped)}")

    if all_ok:
        log.info("All final-pipeline Phase 4 outputs present.")
    else:
        log.warning("Some outputs missing — check warnings above.")


# ── Main ───────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not _FINAL_SRC_CSV.exists():
        log.error("Source not found: %s  — run run_final_pipeline.py first.", _FINAL_SRC_CSV)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 4 — Option A+CORR25 Final Pipeline Analytics")
    log.info("  Input:  %s", _FINAL_SRC_CSV)
    log.info("  Output: %s", _FINAL_METRICS)
    log.info("=" * 60)

    fold_metrics = _load_final_fold_metrics()
    log.info("Loaded %d completed folds", len(fold_metrics))
    _orch.print_fold_summary(fold_metrics)

    # ── §1: Sharpe distribution ────────────────────────────────────────────
    log.info("\n[§1] Sharpe distribution")
    _run_sharpe_distribution(fold_metrics)

    # ── §2: Regime partition ───────────────────────────────────────────────
    log.info("\n[§2] Regime partition")
    regime_df = run_regime_analysis(fold_metrics, save=True)
    print_regime_report(regime_df)

    # ── §5: Overfitting diagnostics ────────────────────────────────────────
    log.info("\n[§5] Overfitting diagnostics (DSR + PBO)")
    overfit_df = run_overfitting_diagnostics(fold_metrics, save=True)
    print_overfitting_report(overfit_df)

    # ── §6: OAT sensitivity (analytical only) ─────────────────────────────
    log.info("\n[§6] OAT sensitivity (analytical)")
    oat_df = run_oat_sensitivity(
        fold_metrics,
        run_structural=args.structural_oat if hasattr(args, "structural_oat") else False,
        structural_max_folds=5,
        save=True,
    )
    print_oat_report(oat_df)

    # ── §9: Exit reason breakdown from final audit logs ────────────────────
    log.info("\n[§9] Exit reason breakdown")
    exit_df = compute_exit_reasons(logs_dir=_FINAL_LOGS)
    if not exit_df.empty:
        _print_exit_summary(exit_df)
    else:
        log.warning("[§9] No exit records parsed from %s", _FINAL_LOGS)

    # ── §10: Cost decomposition ────────────────────────────────────────────
    log.info("\n[§10] Cost decomposition")
    _run_cost_decomposition(fold_metrics)

    # ── §11: Delta trajectory ──────────────────────────────────────────────
    log.info("\n[§11] Delta trajectory")
    _run_delta_trajectory(fold_metrics)

    # ── §7 + §8: NC bootstrap + latency ───────────────────────────────────
    log.info("\n[§7+8] NC bootstrap + latency — extracting from fold_metrics")
    _extract_nc_latency(fold_metrics)

    # ── §3: Pair persistence (optional) ───────────────────────────────────
    if getattr(args, "persistence", False):
        log.info("\n[§3] Pair persistence (re-running Johansen, ~30 min)")
        from src.phase4_defense.persistence import run_persistence
        persist_df = run_persistence(save=True)
        log.info("Persistence complete: %d fold points", len(persist_df))
    else:
        log.info("\n[§3] Pair persistence skipped (pass --persistence to run)")

    # ── §4: Volume stratification (optional) ──────────────────────────────
    if getattr(args, "volume_strat", False):
        log.info("\n[§4] Volume stratification (~15 min)")
        from src.phase4_defense.volume_strat import run_volume_strat, print_volume_report
        vol_fold, vol_pairs = run_volume_strat(save=True)
        print_volume_report(vol_fold, vol_pairs)
    else:
        log.info("\n[§4] Volume stratification skipped (pass --volume-strat to run)")

    # ── Final checklist ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Phase 4 final analytics complete. Output files:")
    _check_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 analytics on Option A+CORR25 final pipeline results"
    )
    parser.add_argument("--structural-oat", action="store_true",
                        help="Run structural OAT variations (slow, ~20 min)")
    parser.add_argument("--persistence",    action="store_true",
                        help="Run P_2022 pair persistence (~30 min)")
    parser.add_argument("--volume-strat",   action="store_true",
                        help="Run volume stratification (~15 min)")
    args = parser.parse_args()
    main(args)



# ===== FILE: run_oat_structural_final.py =====
"""
run_oat_structural_final.py  —  §6 Structural OAT on Option A + CORR25 Final Config

Varies structural parameters one-at-a-time around the final config defaults:
  Default: Z_entry=3.0, eos_flatten=False, stop_loss=None, CORR25>=0.25,
           persistence gate, HL<=6d, delta=1e-7, TC=30bps/side

Parameters swept:
  Z_entry     : [2.5, 2.75, 3.0*, 3.25, 3.5]
  stop_loss   : [None*, -2.5%, -5.0%]  (post-hoc on Z=3.0 default run)

NOTE: max_holding variations omitted — the engine has no separate max-hold cap;
"EOS" vs "no-EOS" is already baked into the final config as eos_flatten=False,
and adding a day-cap requires engine changes not in scope.

Runs on all 23 completed folds (same fold set as run_final_pipeline.py).

Outputs:
  results/metrics/final/phase4/oat_structural.csv
  Appends structural rows to results/metrics/final/phase4/oat_sensitivity.csv
"""

import sys, os, logging, time, traceback
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pathlib import Path

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("oat_structural")

from src.phase2_execution.kalman import warmup_kalman
print("Warming up Numba JIT...")
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.utils.metrics import compute_sharpe, compute_max_dd, compute_cagr

# ── Constants (must match run_final_pipeline.py exactly) ───────────────────
PHASE1_DIR   = "results/metrics/phase1_folds"
DATA_1MIN    = "data/validated/1min_phase2"
DATA_5MIN    = "data/validated/5min_phase1"
OUT_DIR      = Path("results/metrics/final/phase4")

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
DEFAULT_Z        = 3.0
TC_BPS           = 30.0
BORROW_BPS_YR    = 50.0
FIXED_DELTA      = 1e-7
HL_MAX_DAYS      = 6.0
CORR25_THRESH    = 0.25
JOHANSEN_PVAL    = 0.05
MAX_PAIRS        = 500
PER_PAIR_DOLLAR  = TOTAL_CAPITAL / N_OPEN_PAIRS_MAX

# OAT sweep values
# Z=3.75 and Z=4.0 only — extending past Z=3.5 to confirm where the post-fix optimum lies.
# (Z in {2.5, 2.75, 3.0, 3.25, 3.5} already measured in the prior run.)
Z_SWEEP  = [3.75, 4.0]
SL_SWEEP = []                       # no SL re-test; SL is only meaningful at default Z=3.0
SL_K     = 5                        # rolling window for stop-loss check

# ── Fold schedule — completed folds only ──────────────────────────────────
COMPLETED_FOLDS = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
]


# ── Data helpers ───────────────────────────────────────────────────────────

def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict:
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


def _slice_1min(cache, tickers, trading_month):
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    return {
        tk: df[(df.index.year == year) & (df.index.month == month)]
        for tk in tickers if tk in cache
        if len(df := cache[tk][(cache[tk].index.year == year) &
                               (cache[tk].index.month == month)]) > 0
    }


def _slice_range(cache, tickers, start, end):
    return {
        tk: sl for tk in tickers if tk in cache
        for sl in [cache[tk][(cache[tk].index >= start) &
                              (cache[tk].index <= end)]]
        if len(sl) > 0
    }


# ── Filter chain (identical to run_final_pipeline.py) ─────────────────────

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
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_hl_cap(pairs_df):
    return pairs_df[pairs_df["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)


def apply_corr25(pairs_df, form_5min, form_start, form_end):
    if pairs_df.empty:
        return pd.DataFrame()
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        ra = aln["log_close_a"].diff()
        rb = aln["log_close_b"].diff()
        valid = ra.notna() & rb.notna()
        if valid.sum() < 20:
            continue
        if float(ra[valid].corr(rb[valid])) >= CORR25_THRESH:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ── Post-hoc stop-loss ─────────────────────────────────────────────────────

def apply_stop_loss(metrics, fold_results, sl_pct):
    """Re-compute Sharpe after applying rolling K-bar stop-loss post-hoc."""
    pair_pnls = metrics.get("pair_pnls", {})
    if not pair_pnls:
        return metrics["sharpe"]

    adj_pnls = []
    for (ta, tb), bar_pnl in pair_pnls.items():
        if (ta, tb) not in fold_results:
            adj_pnls.append(bar_pnl)
            continue
        pos = fold_results[(ta, tb)]["position"].reindex(bar_pnl.index).fillna(0).astype(int)
        rolling_pct = bar_pnl.rolling(SL_K, min_periods=1).sum() / PER_PAIR_DOLLAR

        adj = bar_pnl.copy()
        in_sl = False
        prev_pos = 0
        for i in range(len(adj)):
            cur_pos = int(pos.iloc[i])
            if prev_pos != 0 and cur_pos == 0:
                in_sl = False
            if cur_pos != 0 and not in_sl and float(rolling_pct.iloc[i]) < sl_pct:
                in_sl = True
            if in_sl and cur_pos != 0:
                adj.iloc[i] = 0.0
            prev_pos = cur_pos
        adj_pnls.append(adj)

    if not adj_pnls:
        return metrics["sharpe"]

    combined = pd.concat(adj_pnls).sort_index().groupby(level=0).sum()
    daily = combined.resample("D").sum().dropna()
    return float(compute_sharpe(daily))


# ── Per-fold runner ────────────────────────────────────────────────────────

def run_fold_filtered(fold_n, form_start, form_end, trading_month,
                      cache_1min, cache_5min, entry_z):
    """Apply full filter chain + run engine. Returns (metrics, fold_results) or (None, None)."""
    try:
        pairs_df = pd.read_csv(f"{PHASE1_DIR}/fold_{fold_n:02d}.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None, None
    if len(pairs_df) == 0:
        return None, None
    if len(pairs_df) > MAX_PAIRS:
        pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_range(cache_5min, tickers, form_start, form_end)
    formation_1min = _slice_range(cache_1min, tickers, form_start, form_end)

    if not trading_1min:
        return None, None

    pairs_df = apply_persistence_gate(pairs_df, formation_5min, form_end)
    if pairs_df.empty:
        return None, None
    pairs_df = apply_hl_cap(pairs_df)
    if pairs_df.empty:
        return None, None
    pairs_df = apply_corr25(pairs_df, formation_5min, form_start, form_end)
    if pairs_df.empty:
        return None, None

    try:
        fold_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            entry_z=entry_z,
            eos_flatten=False,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        log.warning("Fold %d Z=%.2f engine FAILED: %s", fold_n, entry_z, e)
        return None, None

    if not fold_results:
        return None, None

    try:
        metrics = run_fold_pnl(
            fold_results=fold_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        log.warning("Fold %d Z=%.2f pnl FAILED: %s", fold_n, entry_z, e)
        return None, None

    return metrics, fold_results


# ── Main ───────────────────────────────────────────────────────────────────

print("Loading data caches...")
t0 = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_1min)} 1-min, {len(cache_5min)} 5-min tickers  [{time.time()-t0:.1f}s]\n")

# Store results: {z_value: {fold_n: sharpe}}
z_sharpes   = {z: {} for z in Z_SWEEP}
sl_sharpes  = {sl: {} for sl in SL_SWEEP if sl is not None}
default_metrics = {}  # fold_n -> (metrics, fold_results) for SL post-hoc

t_total = time.time()
n_folds = len(COMPLETED_FOLDS)

for i, (fold_n, form_start, form_end, trading_month) in enumerate(COMPLETED_FOLDS, 1):
    print(f"Fold {fold_n:02d} [{trading_month}]  ({i}/{n_folds})")

    for z in Z_SWEEP:
        t1 = time.time()
        metrics, fold_results = run_fold_filtered(
            fold_n, form_start, form_end, trading_month,
            cache_1min, cache_5min, entry_z=z
        )
        if metrics is not None:
            sr = metrics["sharpe"]
            z_sharpes[z][fold_n] = sr
            # Cache default-Z run for stop-loss post-hoc
            if abs(z - DEFAULT_Z) < 1e-9:
                default_metrics[fold_n] = (metrics, fold_results)
        else:
            # Use NaN so fold is excluded from aggregate (same as skip)
            z_sharpes[z][fold_n] = np.nan
        print(f"  Z={z:.2f}  SR={sr:+.3f}  [{time.time()-t1:.0f}s]"
              if metrics else f"  Z={z:.2f}  skip")

    # Apply stop-loss post-hoc to the Z=3.0 run
    if fold_n in default_metrics:
        m, fr = default_metrics[fold_n]
        for sl in SL_SWEEP:
            if sl is None:
                continue
            sl_sr = apply_stop_loss(m, fr, sl)
            sl_sharpes[sl][fold_n] = sl_sr
            print(f"  SL={sl:.1%}  SR={sl_sr:+.3f}")

print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

# ── Build output DataFrame (extension run: Z in [3.75, 4.0] only) ──────────

rows = []

# Use the prior Z=3.0 mean SR from the existing oat_structural.csv as the delta reference
default_mean = np.nan
prior_path = OUT_DIR / "oat_structural.csv"
if prior_path.exists():
    prior = pd.read_csv(prior_path)
    prior_z3 = prior[(prior["param"] == "Z_entry") & (prior["value"].astype(str) == "3.0")]
    if len(prior_z3):
        default_mean = float(prior_z3["mean_sharpe"].iloc[0])

# Z_entry variations (extension only)
for z in Z_SWEEP:
    arr = np.array([v for v in z_sharpes[z].values() if not np.isnan(v)])
    if len(arr) == 0:
        continue
    label = f"Z_entry={z:.2f}"
    rows.append({
        "param":          "Z_entry",
        "value":          str(z),
        "label":          label,
        "mean_sharpe":    float(np.mean(arr)),
        "median_sharpe":  float(np.median(arr)),
        "pct_positive":   float(np.mean(arr > 0)),
        "n_folds":        len(arr),
        "mode":           "structural",
    })

structural_df = pd.DataFrame(rows)

# Write to a separate extension file, do NOT overwrite the prior results
out_structural = OUT_DIR / "oat_structural_extended.csv"
structural_df.to_csv(out_structural, index=False)
print(f"\nSaved: {out_structural}")

# Append (do not replace) to oat_sensitivity.csv as new structural rows
oat_combined_path = OUT_DIR / "oat_sensitivity.csv"
if oat_combined_path.exists():
    existing = pd.read_csv(oat_combined_path)
    # Drop any prior rows for the new Z values only (3.75, 4.0) to avoid duplication
    drop_mask = (existing["param"] == "Z_entry") & (existing["value"].astype(str).isin([str(z) for z in Z_SWEEP]))
    existing_clean = existing[~drop_mask]
    combined = pd.concat([existing_clean, structural_df], ignore_index=True)
else:
    combined = structural_df
combined.to_csv(oat_combined_path, index=False)
print(f"Updated: {oat_combined_path}")

print(f"\n=== §6 Structural OAT EXTENSION — Option A+CORR25 (Z in {Z_SWEEP}) ===")
if not np.isnan(default_mean):
    print(f"  Reference: prior Z=3.0 mean SR = {default_mean:+.3f}\n")
else:
    print(f"  No prior Z=3.0 reference found in {prior_path}; deltas will be NaN\n")

print(f"  {'Label':<25} {'Mean SR':>8}  {'Delta':>7}  {'%Pos':>5}  {'N':>4}")
print("  " + "-" * 60)
for _, r in structural_df.iterrows():
    delta = r["mean_sharpe"] - default_mean
    print(f"  {r['label']:<25} {r['mean_sharpe']:>+8.3f}  {delta:>+7.3f}  "
          f"{r['pct_positive']:>5.0%}  {int(r['n_folds']):>4}")

