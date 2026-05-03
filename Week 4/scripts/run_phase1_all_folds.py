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
