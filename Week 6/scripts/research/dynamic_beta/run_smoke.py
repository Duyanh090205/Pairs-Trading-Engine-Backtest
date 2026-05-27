"""
Dynamic-β smoke test — runner.

Same input (V4 discovery) → 3 arms (A0/A1/B1) → same downstream → per-pair results.

Usage:
    python -m scripts.research.dynamic_beta.run_smoke

Outputs to: results/dynamic_beta_smoke/<timestamp>/
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Single-threaded BLAS (V4 determinism convention)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd

WEEK6_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WEEK6_ROOT))

from engine_daily import discovery_daily
from engine.phase1_cointegration.factor_residual import project_residual

from functools import partial

from scripts.research.dynamic_beta.arms import (
    run_arm_a0, run_arm_a1, run_arm_b1, run_arm_b1_guarded,
)

# ---------------------------------------------------------------------------
# Config — matches V4 defaults (see scripts/run_v4_pipeline.py)
# ---------------------------------------------------------------------------

VALIDATED_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated")
DATA_DAILY = VALIDATED_DIR / "daily_phase3"

TOTAL_CAPITAL = 1_000_000.0
ENTRY_Z = 2.0
Z_WINDOW = 60
HARD_SL_Z = 4.0
HL_MIN = 5.0
HL_MAX = 30.0
USE_DYNAMIC_COST = True   # match V4 final_dynamic_cost run

# 6 selected folds (see README.md §"Folds"). Pre-registered.
SELECTED_FOLDS = [4, 5, 18, 22, 28, 32]

# Arm sets — selected via --arms CLI flag (default: 'default')
ARMS_DEFAULT = {
    "A0_static_v4": run_arm_a0,
    "A1_short_ols60": run_arm_a1,
    "B1_kalman_hl10": run_arm_b1,
    "B1c_clamp_gate": partial(
        run_arm_b1_guarded,
        half_life_bars=10.0,
        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=None,
    ),
    "B1g_all_guards": partial(
        run_arm_b1_guarded,
        half_life_bars=10.0,
        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=0.30,
    ),
}

# HL sweep: B1c (clamp+gate, no drift-close) at four half-life values.
# Tests whether HL=10 is optimal or whether slower/faster β adaptation wins.
ARMS_HL_SWEEP = {
    "A0_static_v4": run_arm_a0,
    "B1c_hl10": partial(run_arm_b1_guarded, half_life_bars=10.0,
                        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=None),
    "B1c_hl20": partial(run_arm_b1_guarded, half_life_bars=20.0,
                        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=None),
    "B1c_hl30": partial(run_arm_b1_guarded, half_life_bars=30.0,
                        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=None),
    "B1c_hl60": partial(run_arm_b1_guarded, half_life_bars=60.0,
                        clamp_factor=3.0, innov_k=4.0, drift_close_threshold=None),
}

ARMS = ARMS_DEFAULT


# ---------------------------------------------------------------------------
# Fold schedule helpers (copy of run_v4_pipeline._build_fold_schedule_v4)
# ---------------------------------------------------------------------------

def _build_fold_schedule_v4() -> list[tuple[int, str, str, str]]:
    schedule: list[tuple[int, str, str, str]] = []
    months = ([(y, m) for y in (2023, 2024, 2025) for m in range(1, 13)]
              + [(2026, 1), (2026, 2), (2026, 3)])
    for fold_n, (year, month) in enumerate(months, start=1):
        trading_month = f"{year:04d}-{month:02d}"
        trade_start = pd.Timestamp(f"{trading_month}-01")
        form_end = (trade_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        form_start = (trade_start - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        schedule.append((fold_n, form_start, form_end, trading_month))
    return schedule


FOLD_SCHEDULE = _build_fold_schedule_v4()


# ---------------------------------------------------------------------------
# Daily cache helpers (copy of run_v4_pipeline._load_all_daily, _slice_daily)
# ---------------------------------------------------------------------------

def _load_all_daily(data_dir: Path) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".parquet"):
            continue
        tk = fname[:-len(".parquet")]
        try:
            cache[tk] = pd.read_parquet(data_dir / fname)
        except Exception:
            pass
    return cache


def _slice_daily(cache_daily: dict, start: str, end: str) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = {}
    for tk in sorted(cache_daily.keys()):
        df = cache_daily[tk]
        sliced = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ---------------------------------------------------------------------------
# Per-fold runner — V4 discovery, then 3 arms on identical pair set
# ---------------------------------------------------------------------------

def _aggregate_pair_metrics(pair_result: pd.DataFrame, beta_info: dict,
                            fold_n: int, trading_month: str,
                            arm: str, ticker_a: str, ticker_b: str) -> dict:
    """Reduce per-pair bar-level DataFrame to row of summary metrics."""
    if pair_result.empty:
        return {
            "fold": fold_n, "trading_month": trading_month, "arm": arm,
            "ticker_a": ticker_a, "ticker_b": ticker_b,
            "n_bars": 0, "n_trades": 0, "total_pnl_net": 0.0, "total_pnl_gross": 0.0,
            "daily_pnl_std": 0.0,
            **beta_info,
        }
    pos = pair_result["position"].values
    pos_prev = np.concatenate([[0], pos[:-1]])
    n_trades = int(np.sum((pos != 0) & (pos_prev == 0)))   # entries
    total_pnl_net = float(pair_result["daily_pnl_net"].sum())
    total_pnl_gross = float(pair_result["daily_pnl_gross"].sum())
    daily_pnl_std = float(pair_result["daily_pnl_net"].std(ddof=1)) if len(pair_result) > 1 else 0.0

    return {
        "fold": fold_n, "trading_month": trading_month, "arm": arm,
        "ticker_a": ticker_a, "ticker_b": ticker_b,
        "n_bars": len(pair_result), "n_trades": n_trades,
        "total_pnl_net": total_pnl_net, "total_pnl_gross": total_pnl_gross,
        "daily_pnl_std": daily_pnl_std,
        **beta_info,
    }


def run_fold_smoke(fold_n: int, formation_start: str, formation_end: str,
                   trading_month: str, cache_daily: dict, cost_data,
                   log_fn) -> list[dict]:
    """One fold: discover pairs (V4), then run all 3 arms over the same pair list."""
    t0 = time.time()
    log_fn(f"Fold {fold_n:02d} [{trading_month}] — slicing formation")
    formation_daily = _slice_daily(cache_daily, formation_start, formation_end)
    if not formation_daily:
        log_fn(f"Fold {fold_n:02d}: no formation data; skipping")
        return []

    # ---- V4 discovery (single source of truth, all arms share this output) ----
    pairs_df, factor_state = discovery_daily.run(
        formation_data=formation_daily, hl_min=HL_MIN, hl_max=HL_MAX,
    )
    if pairs_df.empty or not factor_state:
        log_fn(f"Fold {fold_n:02d}: empty discovery; skipping")
        return []
    log_fn(f"Fold {fold_n:02d}: discovery={len(pairs_df)} pairs")

    # ---- Trading-window residual projection (V4 Path A re-anchor) ----
    trade_start = trading_month + "-01"
    trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    trading_daily_raw = _slice_daily(cache_daily, trade_start, trade_end)
    if not trading_daily_raw:
        log_fn(f"Fold {fold_n:02d}: no trading data; skipping")
        return []

    loadings_W = factor_state["loadings_W"]
    factor_tickers = factor_state["tickers"]
    resid_form_df = factor_state["residual_log_prices"]

    resid_trade_dict = project_residual(
        trading_daily_raw, loadings_W, factor_tickers, min_obs=10,
    )
    for tk in list(resid_trade_dict.keys()):
        if tk not in resid_form_df.columns:
            continue
        s_form_tk = resid_form_df[tk].dropna()
        if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
            continue
        shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
        resid_trade_dict[tk] = resid_trade_dict[tk] + shift
    resid_trade_df = pd.concat(resid_trade_dict, axis=1)

    # ---- Per-pair × per-arm ----
    rows: list[dict] = []
    skipped_no_data = 0
    for _, p in pairs_df.iterrows():
        ta, tb = p["ticker_a"], p["ticker_b"]
        if ta not in resid_form_df.columns or tb not in resid_form_df.columns:
            skipped_no_data += 1
            continue
        if ta not in resid_trade_df.columns or tb not in resid_trade_df.columns:
            skipped_no_data += 1
            continue

        resid_a_form = resid_form_df[ta].dropna()
        resid_b_form = resid_form_df[tb].dropna()
        resid_a_trade = resid_trade_df[ta].dropna()
        resid_b_trade = resid_trade_df[tb].dropna()

        beta_v4 = float(p["beta_pca"])
        alpha_v4 = float(p["alpha_pca"])
        R = float(p["R_measurement_noise"])

        for arm_name, runner in ARMS.items():
            try:
                pair_res, beta_info = runner(
                    resid_a_form, resid_b_form, resid_a_trade, resid_b_trade,
                    beta_v4=beta_v4, alpha_v4=alpha_v4, R=R,
                    entry_z=ENTRY_Z, z_window=Z_WINDOW, hard_sl_z=HARD_SL_Z,
                    cost_data=cost_data, ticker_a=ta, ticker_b=tb,
                )
            except Exception as e:
                log_fn(f"  Fold {fold_n:02d} {arm_name} {ta}/{tb} FAILED: {e}")
                pair_res = pd.DataFrame()
                beta_info = {"beta_used": float("nan"), "alpha_used": float("nan"),
                             "error": str(e)}

            rows.append(_aggregate_pair_metrics(
                pair_res, beta_info, fold_n, trading_month, arm_name, ta, tb,
            ))

    elapsed = time.time() - t0
    log_fn(f"Fold {fold_n:02d}: done | {len(pairs_df)} pairs × {len(ARMS)} arms "
           f"| skipped {skipped_no_data} no-data | {elapsed:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# Aggregation (per fold, per arm)
# ---------------------------------------------------------------------------

def _sharpe(daily_pnl: pd.Series) -> float:
    """Annualized Sharpe on daily P&L series. NaN if std == 0 or n < 2."""
    if len(daily_pnl) < 2:
        return 0.0
    mu = float(daily_pnl.mean())
    sd = float(daily_pnl.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * np.sqrt(252.0)


def aggregate_per_fold_per_arm(per_pair_df: pd.DataFrame, fold_pnl_daily: dict) -> pd.DataFrame:
    """
    fold_pnl_daily: {(fold, arm) -> pd.Series of daily-summed-over-pairs P&L}
    """
    out_rows = []
    grouped = per_pair_df.groupby(["fold", "trading_month", "arm"])
    for (fold, tm, arm), grp in grouped:
        key = (fold, arm)
        daily_series = fold_pnl_daily.get(key, pd.Series(dtype=float))
        sharpe = _sharpe(daily_series)
        total_return = float(grp["total_pnl_net"].sum()) / TOTAL_CAPITAL
        avg_beta_used = float(grp["beta_used"].mean())
        max_beta = float(grp["beta_max"].max()) if "beta_max" in grp.columns else float("nan")
        min_beta = float(grp["beta_min"].min()) if "beta_min" in grp.columns else float("nan")
        n_trades = int(grp["n_trades"].sum())
        out_rows.append({
            "fold": fold, "trading_month": tm, "arm": arm,
            "n_pairs": int(len(grp)),
            "n_trades": n_trades,
            "total_pnl_net": float(grp["total_pnl_net"].sum()),
            "total_return": total_return,
            "sharpe": sharpe,
            "avg_beta_used": avg_beta_used,
            "max_beta_observed": max_beta,
            "min_beta_observed": min_beta,
        })
    return pd.DataFrame(out_rows).sort_values(["fold", "arm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(WEEK6_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="smoke",
                        help="'smoke' (6 folds), 'all' (39 folds), or comma-separated list e.g. '1,2,3'")
    parser.add_argument("--arms", default="default",
                        choices=["default", "hl_sweep"],
                        help="'default' = A0/A1/B1/B1c/B1g; 'hl_sweep' = A0 + B1c at HL 10/20/30/60")
    parser.add_argument("--entry-z", type=float, default=2.0,
                        help="Override ENTRY_Z (default 2.0 = V4 baseline; ship config uses 3.0)")
    parser.add_argument("--hard-sl-z", type=float, default=4.0,
                        help="Override HARD_SL_Z (default 4.0; ship uses 5.0 with entry-z=3.0)")
    args = parser.parse_args()

    global SELECTED_FOLDS, ARMS, ENTRY_Z, HARD_SL_Z
    if args.folds == "smoke":
        pass  # use default 6 folds
    elif args.folds == "all":
        SELECTED_FOLDS = [s[0] for s in FOLD_SCHEDULE]
    else:
        SELECTED_FOLDS = [int(x) for x in args.folds.split(",")]

    if args.arms == "hl_sweep":
        ARMS = ARMS_HL_SWEEP

    # Override Z thresholds if provided
    ENTRY_Z = float(args.entry_z)
    HARD_SL_Z = float(args.hard_sl_z)
    print(f"  ENTRY_Z={ENTRY_Z}, HARD_SL_Z={HARD_SL_Z}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = WEEK6_ROOT / "results" / "dynamic_beta_smoke" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    def log_fn(msg: str) -> None:
        print(msg, flush=True)
        logging.info(msg)

    log_fn(f"=== Dynamic-β smoke test ===")
    log_fn(f"timestamp: {timestamp}")
    log_fn(f"git SHA: {_git_sha()}")
    log_fn(f"folds: {SELECTED_FOLDS}")
    log_fn(f"arms: {list(ARMS.keys())}")
    log_fn(f"cost mode: {'DYNAMIC (Week 5)' if USE_DYNAMIC_COST else 'FLAT 30bps'}")
    log_fn(f"out_dir: {out_dir}")

    # ---- Load data + cost ----
    t_load = time.time()
    log_fn(f"Loading daily cache...")
    cache_daily = _load_all_daily(DATA_DAILY)
    log_fn(f"  loaded {len(cache_daily)} tickers in {time.time()-t_load:.1f}s")

    cost_data = None
    if USE_DYNAMIC_COST:
        from engine_daily.cost_engine import load_cost_data
        t_cd = time.time()
        log_fn(f"Loading cost data (Week 5 spread + kappa)...")
        cost_data = load_cost_data()
        log_fn(f"  loaded {len(cost_data.daily_spread)} spread rows, "
               f"{len(cost_data.kappa_map)} kappa tickers in {time.time()-t_cd:.1f}s")

    # ---- Run folds ----
    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}
    all_rows: list[dict] = []
    fold_pnl_daily: dict = {}   # (fold, arm) -> pd.Series

    # We also need to collect per-bar P&L across pairs for Sharpe calculation. To keep
    # memory reasonable, accumulate per-fold sums on-the-fly.
    for fold_n in SELECTED_FOLDS:
        if fold_n not in sched_by_fold:
            log_fn(f"WARNING: fold {fold_n} not in schedule, skipping")
            continue
        _, fs, fe, tm = sched_by_fold[fold_n]

        # We need per-bar P&L per arm. Run differently: collect bar-level
        # DataFrames per arm and sum across pairs.
        t_fold = time.time()
        log_fn(f"Fold {fold_n:02d} [{tm}] — slicing formation {fs}..{fe}")
        formation_daily = _slice_daily(cache_daily, fs, fe)
        if not formation_daily:
            log_fn(f"  Fold {fold_n:02d}: empty formation, skipping")
            continue

        pairs_df, factor_state = discovery_daily.run(
            formation_data=formation_daily, hl_min=HL_MIN, hl_max=HL_MAX,
        )
        if pairs_df.empty or not factor_state:
            log_fn(f"  Fold {fold_n:02d}: empty discovery, skipping")
            continue
        log_fn(f"  Fold {fold_n:02d}: discovery={len(pairs_df)} pairs")

        trade_start = tm + "-01"
        trade_end = (pd.Timestamp(trade_start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        trading_daily_raw = _slice_daily(cache_daily, trade_start, trade_end)
        if not trading_daily_raw:
            log_fn(f"  Fold {fold_n:02d}: empty trading slice, skipping")
            continue

        loadings_W = factor_state["loadings_W"]
        factor_tickers = factor_state["tickers"]
        resid_form_df = factor_state["residual_log_prices"]

        resid_trade_dict = project_residual(
            trading_daily_raw, loadings_W, factor_tickers, min_obs=10,
        )
        for tk in list(resid_trade_dict.keys()):
            if tk not in resid_form_df.columns:
                continue
            s_form_tk = resid_form_df[tk].dropna()
            if len(s_form_tk) == 0 or len(resid_trade_dict[tk]) == 0:
                continue
            shift = float(s_form_tk.iloc[-1]) - float(resid_trade_dict[tk].iloc[0])
            resid_trade_dict[tk] = resid_trade_dict[tk] + shift
        resid_trade_df = pd.concat(resid_trade_dict, axis=1)

        # Per-bar P&L accumulators per arm
        arm_daily_pnl: dict[str, pd.Series] = {arm: None for arm in ARMS}
        skipped_no_data = 0

        for _, p in pairs_df.iterrows():
            ta, tb = p["ticker_a"], p["ticker_b"]
            if ta not in resid_form_df.columns or tb not in resid_form_df.columns:
                skipped_no_data += 1
                continue
            if ta not in resid_trade_df.columns or tb not in resid_trade_df.columns:
                skipped_no_data += 1
                continue

            resid_a_form = resid_form_df[ta].dropna()
            resid_b_form = resid_form_df[tb].dropna()
            resid_a_trade = resid_trade_df[ta].dropna()
            resid_b_trade = resid_trade_df[tb].dropna()

            beta_v4 = float(p["beta_pca"])
            alpha_v4 = float(p["alpha_pca"])
            R = float(p["R_measurement_noise"])

            for arm_name, runner in ARMS.items():
                try:
                    pair_res, beta_info = runner(
                        resid_a_form, resid_b_form, resid_a_trade, resid_b_trade,
                        beta_v4=beta_v4, alpha_v4=alpha_v4, R=R,
                        entry_z=ENTRY_Z, z_window=Z_WINDOW, hard_sl_z=HARD_SL_Z,
                        cost_data=cost_data, ticker_a=ta, ticker_b=tb,
                    )
                except Exception as e:
                    log_fn(f"    {arm_name} {ta}/{tb} FAILED: {e}")
                    pair_res = pd.DataFrame()
                    beta_info = {"beta_used": float("nan"), "alpha_used": float("nan"),
                                 "error": str(e)}

                all_rows.append(_aggregate_pair_metrics(
                    pair_res, beta_info, fold_n, tm, arm_name, ta, tb,
                ))

                # Accumulate daily P&L across pairs for this arm/fold
                if not pair_res.empty:
                    pnl_series = pair_res["daily_pnl_net"]
                    if arm_daily_pnl[arm_name] is None:
                        arm_daily_pnl[arm_name] = pnl_series.copy()
                    else:
                        arm_daily_pnl[arm_name] = arm_daily_pnl[arm_name].add(
                            pnl_series, fill_value=0.0,
                        )

        for arm_name, series in arm_daily_pnl.items():
            if series is not None:
                fold_pnl_daily[(fold_n, arm_name)] = series

        elapsed = time.time() - t_fold
        log_fn(f"  Fold {fold_n:02d}: done in {elapsed:.1f}s "
               f"| {len(pairs_df)} pairs × {len(ARMS)} arms "
               f"| skipped {skipped_no_data} no-data")

    # ---- Save outputs ----
    per_pair_df = pd.DataFrame(all_rows)
    per_pair_df.to_parquet(out_dir / "per_pair.parquet", index=False)
    log_fn(f"Wrote per_pair.parquet: {len(per_pair_df)} rows")

    per_fold_df = aggregate_per_fold_per_arm(per_pair_df, fold_pnl_daily)
    per_fold_df.to_csv(out_dir / "per_fold.csv", index=False)
    per_fold_df.to_parquet(out_dir / "per_fold.parquet", index=False)
    log_fn(f"Wrote per_fold: {len(per_fold_df)} rows")

    # Save daily P&L series too (for bootstrap CI in analyze.py)
    pnl_records = []
    for (fold_n, arm_name), series in fold_pnl_daily.items():
        for date, pnl in series.items():
            pnl_records.append({
                "fold": fold_n, "arm": arm_name,
                "date": pd.Timestamp(date), "daily_pnl_net": float(pnl),
            })
    daily_pnl_df = pd.DataFrame(pnl_records)
    daily_pnl_df.to_parquet(out_dir / "daily_pnl.parquet", index=False)
    log_fn(f"Wrote daily_pnl.parquet: {len(daily_pnl_df)} rows")

    # Save run metadata
    metadata = {
        "timestamp": timestamp,
        "git_sha": _git_sha(),
        "folds": SELECTED_FOLDS,
        "arms": list(ARMS.keys()),
        "use_dynamic_cost": USE_DYNAMIC_COST,
        "entry_z": ENTRY_Z, "z_window": Z_WINDOW, "hard_sl_z": HARD_SL_Z,
        "hl_min": HL_MIN, "hl_max": HL_MAX,
        "total_capital": TOTAL_CAPITAL,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Pointer file at top of results dir for convenience
    pointer = WEEK6_ROOT / "results" / "dynamic_beta_smoke" / "latest.txt"
    pointer.write_text(timestamp, encoding="utf-8")

    log_fn(f"=== Done. Run analyze.py {timestamp} for decision report. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
