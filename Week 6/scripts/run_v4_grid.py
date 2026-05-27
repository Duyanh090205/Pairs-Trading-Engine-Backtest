"""
V4 daily grid search — entry_z × hl_max sweep across all 39 folds.

Grid: entry_z ∈ {1.5, 2.0, 2.5} × hl_max ∈ {10, 15, 30} = 9 combos.
For each combo, runs all 39 walk-forward folds and writes per-combo
fold_metrics.csv. Final summary aggregates winning combo selection
on training subset (folds 1-30) with OOS verification (folds 31-39).

Output layout:
    results/v4/grid/ez{Z}_hl{HL}/fold_metrics.csv     (per combo)
    results/v4/grid/grid_summary.csv                  (aggregate)
    results/v4/grid/oos_split.csv                     (train vs OOS)

Total runtime estimate: ~2.5 hours (9 combos × ~16 min each).
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
import logging
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
except (AttributeError, Exception):
    pass

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))
sys.path.insert(0, str(WEEK6 / "scripts"))   # for `from run_v4_pipeline import ...`
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from run_v4_pipeline import (
    _load_all_daily, run_fold_v4, FOLD_SCHEDULE,
    DATA_DAILY, TOTAL_CAPITAL,
)

# ----- Grid definition -----
ENTRY_Z_VALUES = [1.5, 2.0, 2.5]
HL_MAX_VALUES  = [10.0, 15.0, 30.0]
HARD_SL_Z      = 4.0   # held constant for this grid
Z_WINDOW       = 60    # held constant
TRAIN_FOLDS    = set(range(1, 31))   # folds 1-30 = in-sample tuning
OOS_FOLDS      = set(range(31, 40))  # folds 31-39 = held out

OUT_BASE = WEEK6 / "results" / "v4" / "grid"


def _run_one_combo(
    entry_z: float, hl_max: float,
    cache: dict, out_dir: Path,
) -> pd.DataFrame:
    """Run all 39 folds for one (entry_z, hl_max) combo. Returns fold_metrics df."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for fold_n, fs, fe, tm in FOLD_SCHEDULE:
        res = run_fold_v4(
            fold_n=fold_n, formation_start=fs, formation_end=fe,
            trading_month=tm, cache_daily=cache, out_dir=out_dir,
            entry_z=entry_z, z_window=Z_WINDOW, hard_sl_z=HARD_SL_Z,
            hl_max=hl_max, hl_min=5.0,
        )
        if res is None:
            fold_rows.append({
                "fold": fold_n, "trading_month": tm,
                "n_pairs": 0, "n_trades": 0, "sharpe": 0.0,
                "total_return": 0.0, "max_dd": 0.0, "win_rate": 0.0,
                "avg_net_bps": 0.0, "n_zero_cross": 0, "n_hard_sl": 0,
                "n_open_at_eom": 0, "cum_var": 0.0, "elapsed_s": 0.0,
            })
            continue
        fm = res.get("fold_metrics", {})
        eb = fm.get("exit_breakdown", {})
        fold_rows.append({
            "fold": res["fold"],
            "trading_month": res["trading_month"],
            "n_pairs": res["n_pairs"],
            "n_trades": fm.get("n_trades", 0),
            "sharpe": fm.get("sharpe", 0.0),
            "total_return": fm.get("total_return", 0.0),
            "max_dd": fm.get("max_dd", 0.0),
            "win_rate": fm.get("win_rate", 0.0),
            "avg_net_bps": fm.get("avg_net_bps", 0.0),
            "n_zero_cross": eb.get("zero_cross", 0),
            "n_hard_sl": eb.get("hard_sl", 0),
            "n_open_at_eom": eb.get("open_at_eom", 0),
            "cum_var": res["cum_var"],
            "n_factor_tickers": res["n_factor_tickers"],
            "elapsed_s": res["elapsed_s"],
        })
    df = pd.DataFrame(fold_rows)
    df.to_csv(out_dir / "fold_metrics.csv", index=False)
    return df


def _combo_summary(df: pd.DataFrame, entry_z: float, hl_max: float) -> dict:
    """Aggregate one combo's per-fold results into a summary row."""
    traded = df[df["n_trades"] > 0]
    train = df[df["fold"].isin(TRAIN_FOLDS) & (df["n_trades"] > 0)]
    oos = df[df["fold"].isin(OOS_FOLDS) & (df["n_trades"] > 0)]
    eb_total = df["n_zero_cross"].sum() + df["n_hard_sl"].sum() + df["n_open_at_eom"].sum()
    return {
        "entry_z": entry_z,
        "hl_max": hl_max,
        "n_folds_traded": int(len(traded)),
        "n_trades_total": int(df["n_trades"].sum()),
        "mean_sharpe_all":   float(traded["sharpe"].mean()) if len(traded) else 0.0,
        "median_sharpe_all": float(traded["sharpe"].median()) if len(traded) else 0.0,
        "winners_all":       int((traded["sharpe"] > 0).sum()),
        "sum_return_all":    float(traded["total_return"].sum()) if len(traded) else 0.0,
        # Training subset (folds 1-30)
        "mean_sharpe_train":   float(train["sharpe"].mean()) if len(train) else 0.0,
        "median_sharpe_train": float(train["sharpe"].median()) if len(train) else 0.0,
        "winners_train":       int((train["sharpe"] > 0).sum()),
        "n_train":             int(len(train)),
        # OOS subset (folds 31-39)
        "mean_sharpe_oos":   float(oos["sharpe"].mean()) if len(oos) else 0.0,
        "median_sharpe_oos": float(oos["sharpe"].median()) if len(oos) else 0.0,
        "winners_oos":       int((oos["sharpe"] > 0).sum()),
        "n_oos":              int(len(oos)),
        # Exit breakdown
        "pct_zero_cross":  100.0 * df["n_zero_cross"].sum() / max(eb_total, 1),
        "pct_hard_sl":     100.0 * df["n_hard_sl"].sum() / max(eb_total, 1),
        "pct_open_at_eom": 100.0 * df["n_open_at_eom"].sum() / max(eb_total, 1),
    }


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"V4 GRID SWEEP — entry_z × hl_max = {len(ENTRY_Z_VALUES)*len(HL_MAX_VALUES)} combos × {len(FOLD_SCHEDULE)} folds")
    print(f"  hold-constant: hard_sl_z={HARD_SL_Z}, z_window={Z_WINDOW}, hl_min=5.0")
    print(f"  Train folds = 1..30 (in-sample), OOS folds = 31..39 (held out)")
    print(f"  Output dir: {OUT_BASE}\n")

    print("Loading daily cache (once)...")
    t_load = time.time()
    cache = _load_all_daily(DATA_DAILY)
    print(f"  loaded {len(cache)} tickers in {time.time()-t_load:.1f}s\n")

    grid_summaries: list[dict] = []
    t_grid = time.time()
    combo_idx = 0
    n_combos = len(ENTRY_Z_VALUES) * len(HL_MAX_VALUES)
    for entry_z in ENTRY_Z_VALUES:
        for hl_max in HL_MAX_VALUES:
            combo_idx += 1
            combo_name = f"ez{entry_z}_hl{int(hl_max)}"
            combo_dir = OUT_BASE / combo_name
            print(f"\n{'='*60}")
            print(f"[{combo_idx}/{n_combos}] Combo: entry_z={entry_z}, hl_max={hl_max} -> {combo_name}")
            print(f"{'='*60}")
            t_combo = time.time()
            df = _run_one_combo(entry_z, hl_max, cache, combo_dir)
            elapsed = time.time() - t_combo

            summary = _combo_summary(df, entry_z, hl_max)
            grid_summaries.append(summary)
            print(f"\nCombo {combo_name} done in {elapsed:.0f}s")
            print(f"  mean Sharpe (all 39)={summary['mean_sharpe_all']:+.3f}, "
                  f"median={summary['median_sharpe_all']:+.3f}, "
                  f"winners={summary['winners_all']}/{summary['n_folds_traded']}")
            print(f"  train (1-30): mean Sharpe={summary['mean_sharpe_train']:+.3f}, "
                  f"OOS (31-39): mean Sharpe={summary['mean_sharpe_oos']:+.3f}")

            # Persist incrementally so a crash doesn't lose prior work
            pd.DataFrame(grid_summaries).to_csv(OUT_BASE / "grid_summary.csv", index=False)

    elapsed_total = time.time() - t_grid
    print(f"\n{'='*60}")
    print(f"GRID COMPLETE in {elapsed_total/60:.1f} min")
    print(f"{'='*60}\n")

    # Final summary
    df_sum = pd.DataFrame(grid_summaries)
    print("Grid summary (sorted by mean_sharpe_train):")
    cols = ["entry_z", "hl_max", "n_trades_total",
            "mean_sharpe_train", "mean_sharpe_oos",
            "median_sharpe_train", "median_sharpe_oos",
            "winners_train", "winners_oos",
            "pct_open_at_eom"]
    df_sorted = df_sum.sort_values("mean_sharpe_train", ascending=False)
    print(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    df_sorted.to_csv(OUT_BASE / "grid_summary.csv", index=False)

    # Winning combo
    if len(df_sum) > 0:
        winner = df_sorted.iloc[0]
        print(f"\nWinner by in-sample mean Sharpe (folds 1-30):")
        print(f"  entry_z={winner['entry_z']}, hl_max={winner['hl_max']}")
        print(f"  in-sample mean Sharpe: {winner['mean_sharpe_train']:+.3f}")
        print(f"  OOS         mean Sharpe: {winner['mean_sharpe_oos']:+.3f}")
        print(f"  IS-OOS gap (data-dredging premium): "
              f"{winner['mean_sharpe_train'] - winner['mean_sharpe_oos']:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
