"""
run_v3_pipeline.py — V3.0 engine entry point
=============================================

V3.0 vs V2.0:
  R1-R3 CORR25 removed; R4 Kalman replaced by static spread; R5 β<0 dropped;
  R6 redundant HL cap removed; F1 factor-residual orthogonalization; F2 dynamic
  Z entry (fixed/vol/hl); F3 HL=[1,6d] single source; F4 static spread;
  A1 time-of-day mask; A2 hard SL |Z|>=5.5; A3 Z-velocity gate; A4 vol-target.

CLI examples:
    python run_v3_pipeline.py --folds 1 --smoke
    python run_v3_pipeline.py --folds all --z-strategy fixed --z-base 3.5
    python run_v3_pipeline.py --folds all --z-strategy vol   --z-base 3.5
    python run_v3_pipeline.py --folds all --z-strategy hl    --z-base 3.5

Outputs (under --out-dir, default results/v3/):
    fold_metrics.csv, equity_full.parquet, trade_log.csv, rebalance_log.csv,
    factor_loadings_per_fold.parquet, delta_log.csv
"""

from __future__ import annotations

import os as _os

# Force single-threaded BLAS BEFORE any scientific imports. Without this,
# LAPACK calls inside statsmodels.coint_johansen produce non-deterministic
# floating-point sums depending on BLAS thread scheduling, causing pairs near
# the top-500 p-value cap boundary to shuffle in/out between runs. Each Johansen
# test is on a tiny (2-variable) system so single-thread BLAS isn't slower.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_var] = "1"

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stdout/stderr so Greek letters (Δ, β, σ, κ) in log/print strings
# don't crash on Windows cp1252. No-op on already-UTF-8 systems.
try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

WEEK6_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6_ROOT))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from engine.phase1_cointegration import discovery as v3_discovery
from engine.phase1_cointegration.factor_residual import project_residual
from engine.phase2_execution.engine import run_fold_execution
from engine.phase3_backtest.metrics_runner import run_fold_pnl
from engine.delta_logger import log_delta, reset_log

# ---- Constants ----
VALIDATED_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated")
DATA_1MIN = VALIDATED_DIR / "1min_phase2"
DATA_5MIN = VALIDATED_DIR / "5min_phase1"

TOTAL_CAPITAL = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
TC_BPS = 30.0
BORROW_BPS_YR = 50.0
JOHANSEN_PVAL = 0.05
MAX_PAIRS = 500
FIXED_DELTA = 1e-7  # retained for run_fold_execution API compatibility (ignored under V3.0 static)

# 45-fold walk-forward schedule
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
# Data cache helpers
# ============================================================

def _load_all_tickers(data_dir: Path, compute_log_close: bool) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    # sorted() — file system listing order is not guaranteed; sort for reproducibility
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".parquet"):
            continue
        tk = fname[:-len(".parquet")]
        try:
            df = pd.read_parquet(data_dir / fname)
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_1min(cache: dict, tickers, trading_month: str) -> dict:
    """Deterministic iteration: sort ticker names so output dict's insertion
    order is reproducible across runs (PYTHONHASHSEED randomizes set order)."""
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    out = {}
    for tk in sorted(tickers):
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == month)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_range(cache: dict, tickers, start: str, end: str) -> dict:
    """Deterministic iteration: sort ticker names so output dict's insertion
    order is reproducible across runs (PYTHONHASHSEED randomizes set order)."""
    out = {}
    for tk in sorted(tickers):
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ============================================================
# Persistence gate (V3.0 — operates on residual log-close)
# ============================================================

def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate_v3(
    pairs_df: pd.DataFrame,
    formation_5min_residual: dict[str, pd.DataFrame],
    form_end: str,
) -> pd.DataFrame:
    """V3.0 persistence gate — Johansen retest on RESIDUAL log-close (not raw)."""
    if pairs_df.empty:
        return pd.DataFrame()
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa = formation_5min_residual.get(ta)
        fb = formation_5min_residual.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 200:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ============================================================
# Build residual data dicts from raw cache + factor state
# ============================================================

def _build_residual_data(
    raw_data: dict[str, pd.DataFrame],
    loadings_W: np.ndarray,
    formation_tickers: list[str],
    anchor_from: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Project a raw log-close dict onto factor residuals via project_residual.

    `anchor_from`, when provided, is the formation-window residual log-price
    DataFrame (from `fit_factor_model`). Each ticker's trading residual series
    is shifted additively so its first value equals that ticker's last value in
    `anchor_from`, making the trading residual a continuation of the formation
    residual. Without this, `project_residual` restarts its cumsum from the
    trading-window first bar, producing a per-ticker level break that pushes the
    formation-calibrated spread mean (alpha_pca) ~5 formation-sigma off-center
    at trading time. See audit_residual_spread_drift.py for the diagnostic.
    """
    if not raw_data:
        return {}
    residual_series = project_residual(raw_data, loadings_W, formation_tickers)

    # Path A: per-ticker re-anchor to continue from formation residual end.
    if anchor_from is not None:
        anchored: dict[str, pd.Series] = {}
        for tk, series in residual_series.items():
            if tk not in anchor_from.columns or len(series) == 0:
                anchored[tk] = series
                continue
            form_tk = anchor_from[tk].dropna()
            if len(form_tk) == 0:
                anchored[tk] = series
                continue
            shift = float(form_tk.iloc[-1]) - float(series.iloc[0])
            anchored[tk] = series + shift
        residual_series = anchored

    out: dict[str, pd.DataFrame] = {}
    for tk, series in residual_series.items():
        raw_df = raw_data[tk]
        df = raw_df.loc[raw_df.index.intersection(series.index)].copy()
        df["log_close"] = series.reindex(df.index)
        if "close" not in df.columns and "close" in raw_df.columns:
            df["close"] = raw_df["close"].reindex(df.index)
        df = df.dropna(subset=["log_close"])
        if len(df) > 0:
            out[tk] = df
    return out


# ============================================================
# Per-fold runner
# ============================================================

def run_fold(
    fold_n: int,
    formation_start: str,
    formation_end: str,
    trading_month: str,
    cache_1min: dict,
    cache_5min: dict,
    out_dir: Path,
    z_strategy: str = "fixed",
    z_base: float = 3.5,
    z_norm: str = "rolling",
) -> dict | None:
    """One fold of V3.0 pipeline. Returns metrics dict on success, None on skip/fail."""
    t0 = time.time()
    log = lambda msg: print(f"  Fold {fold_n:02d} [{trading_month}]: {msg}")

    # ---- 1. Phase 1 V3.0 discovery: cointegration on factor-residual + β>0 filter ----
    # Slice the pre-loaded 5-min cache to the formation window so discovery doesn't
    # re-read 528 parquets from disk (~25-50s on a cold cache).
    log(f"slicing formation 5-min cache for {formation_start} -> {formation_end}")
    formation_data_full = _slice_range(
        cache_5min, set(cache_5min.keys()), formation_start, formation_end,
    )
    log(f"formation data: {len(formation_data_full)} tickers ready")
    try:
        pairs_df, factor_state = v3_discovery.run(
            formation_start=formation_start,
            formation_end=formation_end,
            validated_dir=VALIDATED_DIR,
            max_pairs=MAX_PAIRS,
            formation_data=formation_data_full,
        )
    except Exception as e:
        log(f"discovery FAILED — {e}")
        traceback.print_exc()
        return None

    if pairs_df.empty or not factor_state:
        log("0 pairs from discovery — skip")
        return None

    # Save factor loadings for this fold (audit trail for F1)
    loadings_W = factor_state["loadings_W"]
    factor_tickers = factor_state["tickers"]
    cum_var = factor_state["diagnostics"]["cumulative_variance_explained"]
    log(f"discovery: {len(pairs_df)} pairs (after beta>0 filter), factor cum_var={cum_var:.3f}")
    log_delta("phase1_discovery", "n_pairs", v2_value=None, v3_value=len(pairs_df),
              notes=f"fold={fold_n}", log_path=str(out_dir / "delta_log.csv"))

    # ---- 2. Build residual formation data for persistence gate ----
    # Consistency fix: use the SAME residual_log_prices DataFrame that discovery
    # used (from fit_factor_model, inner-joined). Previously we recomputed via
    # _build_residual_data which uses project_residual (outer-join + ffill),
    # giving slightly different residual values and causing persistence-gate
    # results to diverge from what discovery's Johansen test was based on.
    residual_log_prices_df = factor_state["residual_log_prices"]
    formation_5min_residual: dict[str, pd.DataFrame] = {}
    for tk in factor_tickers:
        if tk not in residual_log_prices_df.columns:
            continue
        series = residual_log_prices_df[tk].dropna()
        if len(series) < 200:
            continue
        formation_5min_residual[tk] = pd.DataFrame(
            {"log_close": series.values}, index=series.index,
        )

    # ---- 3. Persistence gate (V3.0 — on residuals) ----
    pairs_df = apply_persistence_gate_v3(pairs_df, formation_5min_residual, formation_end)
    if pairs_df.empty:
        log("0 after persistence gate — skip")
        return None
    log(f"persistence gate: {len(pairs_df)} pairs")
    log_delta("persistence_gate", "n_pairs", v2_value=None, v3_value=len(pairs_df),
              notes=f"fold={fold_n}", log_path=str(out_dir / "delta_log.csv"))

    # ---- 4. Load trading data (FULL universe for accurate factor projection) ----
    full_tickers_set = set(factor_tickers)
    trading_1min_raw = _slice_1min(cache_1min, full_tickers_set, trading_month)
    if not trading_1min_raw:
        log("no 1-min data — skip")
        return None

    # Project trading data through formation loadings.
    # Pass formation residuals so trading residuals are anchored as a continuation
    # of formation residuals (Path A fix — removes per-ticker level break that
    # was offsetting the formation-fit alpha_pca by ~5 formation-sigma).
    trading_1min_residual = _build_residual_data(
        trading_1min_raw, loadings_W, factor_tickers,
        anchor_from=factor_state["residual_log_prices"],
    )
    log(f"trading data: {len(trading_1min_residual)} residual ticker series")

    # ---- 5. Engine execution (V3.0: static spread, time-mask, hard SL, vol-target) ----
    # z_norm='rolling': use rolling-window Z (formation_5min=None).
    # z_norm='formation': use formation-locked Z params (mean/std from formation
    #                     residual spread). Previously catastrophic (-99.87% on F1,
    #                     19,435 trades) due to the residual-routing bug that put
    #                     trading spread ~5 formation-sigma off-center; Path A
    #                     fix in _build_residual_data now anchors trading residuals
    #                     as a continuation of formation residuals, so formation Z
    #                     params should be applicable again.
    _formation_5min_arg = formation_5min_residual if z_norm == "formation" else None
    try:
        fold_engine_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min_residual,
            delta=FIXED_DELTA,       # ignored under V3.0 static; kept for API compat
            entry_z=z_base,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            total_capital=TOTAL_CAPITAL,
            eos_flatten=False,
            formation_5min=_formation_5min_arg,
            formation_ref=None,
            z_strategy=z_strategy,
        )
    except Exception as e:
        log(f"run_fold_execution FAILED — {e}")
        traceback.print_exc()
        return None

    if not fold_engine_results:
        log("engine 0 pairs")
        return None

    # ---- 6. Phase 3 PnL metrics ----
    try:
        fold_metrics = run_fold_pnl(
            fold_results=fold_engine_results,
            trading_1min=trading_1min_residual,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        log(f"run_fold_pnl FAILED — {e}")
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    log(f"complete in {elapsed:.1f}s -- n_trades={fold_metrics.get('n_trades', 0)}, "
        f"sharpe={fold_metrics.get('sharpe', 0):.3f}, "
        f"total_return={fold_metrics.get('total_return', 0.0):.4f}")

    return {
        "fold_metrics": fold_metrics,
        "fold_engine_results": fold_engine_results,
        "pairs_df": pairs_df,
        "factor_state": factor_state,
    }


# ============================================================
# Main aggregator
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="all",
                        help="'all' or range like '1-3' or comma-separated '1,2,3'")
    parser.add_argument("--z-strategy", default="fixed", choices=["fixed", "vol", "hl"])
    parser.add_argument("--z-base", type=float, default=3.5,
                        help="Baseline Z entry threshold")
    parser.add_argument("--z-norm", default="rolling", choices=["rolling", "formation"],
                        help="Z normalization: rolling window (default) or formation-locked "
                             "mean/std from formation residual spread.")
    parser.add_argument("--out-dir", default=str(WEEK6_ROOT / "results" / "v3"))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: write to results/v3/smoke/fold_NN/")
    args = parser.parse_args()

    # Parse fold selection
    if args.folds == "all":
        fold_nums = [s[0] for s in FOLD_SCHEDULE]
    elif "-" in args.folds:
        lo, hi = args.folds.split("-")
        fold_nums = list(range(int(lo), int(hi) + 1))
    else:
        fold_nums = [int(x) for x in args.folds.split(",")]
    selected = [s for s in FOLD_SCHEDULE if s[0] in fold_nums]

    out_dir = Path(args.out_dir)
    if args.smoke and len(selected) == 1:
        out_dir = out_dir / "smoke" / f"fold_{selected[0][0]:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fresh delta log for this run
    reset_log(str(out_dir / "delta_log.csv"))

    print(f"V3.0 pipeline | z_strategy={args.z_strategy}, z_base={args.z_base}, "
          f"z_norm={args.z_norm}")
    print(f"Folds: {len(selected)} | out_dir={out_dir}")
    print(f"Loading data caches...")
    t_load = time.time()
    cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
    cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
    print(f"  loaded {len(cache_1min)} 1-min + {len(cache_5min)} 5-min tickers "
          f"in {time.time()-t_load:.1f}s\n")

    fold_metrics_rows = []
    all_trade_logs = []
    all_rebalance_logs = []
    factor_loadings_rows = []

    for fold_n, fs, fe, tm in selected:
        result = run_fold(
            fold_n=fold_n,
            formation_start=fs,
            formation_end=fe,
            trading_month=tm,
            cache_1min=cache_1min,
            cache_5min=cache_5min,
            out_dir=out_dir,
            z_strategy=args.z_strategy,
            z_base=args.z_base,
            z_norm=args.z_norm,
        )
        if result is None:
            fold_metrics_rows.append({
                "fold": fold_n, "trading_month": tm,
                "pnl_usd": 0.0, "sharpe": 0.0, "n_pairs": 0, "n_trades": 0,
            })
            continue

        fm = result["fold_metrics"]
        fm["fold"] = fold_n
        fm["trading_month"] = tm
        fold_metrics_rows.append(fm)

        # Persist factor loadings for this fold (audit trail)
        W = result["factor_state"]["loadings_W"]
        tickers = result["factor_state"]["tickers"]
        for k in range(W.shape[0]):
            for j, tk in enumerate(tickers):
                factor_loadings_rows.append({
                    "fold": fold_n,
                    "component": k,
                    "ticker": tk,
                    "loading": float(W[k, j]),
                })

        # Trade log + rebalance log aggregation will be added if phase3 outputs them
        # (currently relying on metrics_runner internals — write_audit_log is V2 path)

    # ---- Persist outputs ----
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    print(f"\nWrote {out_dir / 'fold_metrics.csv'} ({len(fold_metrics_df)} rows)")

    if factor_loadings_rows:
        pd.DataFrame(factor_loadings_rows).to_parquet(
            out_dir / "factor_loadings_per_fold.parquet", index=False,
        )
        print(f"Wrote factor_loadings_per_fold.parquet ({len(factor_loadings_rows)} rows)")

    # phase3 metrics_runner returns columns: sharpe, total_return, n_trades, etc.
    # (no 'pnl_usd' column under V3 — use total_return as proxy for "folds traded")
    if "n_trades" in fold_metrics_df.columns:
        log_delta("final", "n_folds_traded",
                  v2_value=None,
                  v3_value=int((fold_metrics_df["n_trades"] > 0).sum()),
                  log_path=str(out_dir / "delta_log.csv"))
    if "total_return" in fold_metrics_df.columns:
        log_delta("final", "total_return_sum",
                  v2_value=None,
                  v3_value=float(fold_metrics_df["total_return"].sum()),
                  log_path=str(out_dir / "delta_log.csv"))
    if "sharpe" in fold_metrics_df.columns:
        log_delta("final", "mean_fold_sharpe",
                  v2_value=0.995,  # V2.0 mean Sharpe (from Week 4 reports)
                  v3_value=float(fold_metrics_df["sharpe"].mean()),
                  log_path=str(out_dir / "delta_log.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
