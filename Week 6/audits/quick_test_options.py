"""
quick_test_options.py — Test design hypotheses on representative folds
=======================================================================
Quick (~10 min total) test of design options on a subset of folds. Avoids
full 39-fold re-run while still giving directional signal.

Tests:
    B1a: Z entry = 2.5 (vs baseline 2.0)
    B1b: Z entry = 3.0
    B1c: Z entry = 3.5

Folds chosen for diversity:
    1  (2023-01) — start, V4 baseline Sharpe +2.17 (best 2023)
    8  (2023-08) — V4 +4.57 (winner) -> with carry was -4.16 (catastrophe)
    18 (2024-06) — V4 -5.93 (worst 2024)
    25 (2025-01) — V4 +0.95 (mid)
    33 (2025-09) — V4 -1.94 (bad)

Output: comparison table — same folds, different Z entry settings.
Runtime: ~12 min (5 folds × 3 settings × 30s each, sequential).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from scripts.run_v4_pipeline import (
    DATA_DAILY, FOLD_SCHEDULE, _load_all_daily, run_fold_v4,
)
from engine_daily.cost_engine import load_cost_data


# Representative folds chosen for regime diversity (no cherry-picking)
REPRESENTATIVE_FOLDS = [1, 8, 18, 25, 33]

# Settings to compare
SETTINGS = [
    # (label, entry_z, hard_sl_z)
    ("baseline_Z=2.0", 2.0, 4.0),
    ("Z=2.5",          2.5, 4.5),
    ("Z=3.0",          3.0, 5.0),
    ("Z=3.5",          3.5, 5.5),
]


def main() -> int:
    print(f"=== Quick test: Z entry sensitivity ===\n", flush=True)
    print(f"Folds: {REPRESENTATIVE_FOLDS}", flush=True)
    print(f"Settings: {[s[0] for s in SETTINGS]}\n", flush=True)

    print("Loading daily cache + cost data...", flush=True)
    t0 = time.time()
    cache_daily = _load_all_daily(DATA_DAILY)
    cost_data = load_cost_data()
    print(f"  Loaded in {time.time()-t0:.1f}s\n", flush=True)

    sched_by_fold = {s[0]: s for s in FOLD_SCHEDULE}
    results: list[dict] = []

    for label, ez, slz in SETTINGS:
        print(f"--- Setting: {label} (entry_z={ez}, hard_sl_z={slz}) ---", flush=True)
        for fold_n in REPRESENTATIVE_FOLDS:
            sched = sched_by_fold.get(fold_n)
            if sched is None:
                continue
            _, fs, fe, tm = sched
            t_fold = time.time()
            res = run_fold_v4(
                fold_n=fold_n, formation_start=fs, formation_end=fe,
                trading_month=tm, cache_daily=cache_daily,
                out_dir=Path("results/v4/quick_test"),
                entry_z=ez, hard_sl_z=slz,
                cost_data=cost_data,
                # no carry-forward in this test — isolating B1 effect
            )
            if res is None:
                continue
            fm = res["fold_metrics"]
            eb = fm.get("exit_breakdown", {})
            results.append({
                "setting": label, "entry_z": ez, "fold": fold_n,
                "trading_month": tm,
                "n_pairs": res["n_pairs"], "n_trades": fm["n_trades"],
                "sharpe": fm["sharpe"], "total_return": fm["total_return"],
                "avg_net_bps": fm["avg_net_bps"],
                "n_zero_cross": eb.get("zero_cross", 0),
                "n_hard_sl": eb.get("hard_sl", 0),
                "n_open_at_eom": eb.get("open_at_eom", 0),
                "elapsed_s": time.time() - t_fold,
            })
            print(f"  Fold {fold_n} ({tm}): n_trades={fm['n_trades']}, "
                  f"sharpe={fm['sharpe']:+.3f}, ret={fm['total_return']:+.4f}, "
                  f"zc/sl/eom={eb.get('zero_cross',0)}/{eb.get('hard_sl',0)}/{eb.get('open_at_eom',0)}",
                  flush=True)
        print()

    df = pd.DataFrame(results)
    df.to_csv(WEEK6 / "results" / "v4" / "quick_test_z_sweep.csv", index=False)
    print(f"\nWrote results/v4/quick_test_z_sweep.csv ({len(df)} rows)\n")

    # --- Summary by setting ---
    print("=== Summary by setting (averaged across 5 representative folds) ===")
    summary = df.groupby(["setting", "entry_z"]).agg(
        mean_sharpe=("sharpe", "mean"),
        median_sharpe=("sharpe", "median"),
        sum_return=("total_return", "sum"),
        sum_trades=("n_trades", "sum"),
        sum_zc=("n_zero_cross", "sum"),
        sum_sl=("n_hard_sl", "sum"),
        sum_eom=("n_open_at_eom", "sum"),
    ).round(3)
    print(summary.to_string())

    print()
    print("=== Pivot: Sharpe by fold and setting ===")
    pivot = df.pivot(index="fold", columns="setting", values="sharpe")
    pivot["fold_month"] = pivot.index.map(
        {r["fold"]: r["trading_month"] for _, r in df.iterrows()}
    )
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else x))

    return 0


if __name__ == "__main__":
    sys.exit(main())
