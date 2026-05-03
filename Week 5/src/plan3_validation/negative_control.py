"""
4.6 — Negative control under dynamic costs.

Decision: Option B (skip with documented reason).

Week 4 stores NC results as Sharpe values in fold_metrics.csv columns
nc_threshold/nc_pass — NOT as a separate trade log we can re-cost. Re-running
NC pairs (CVNA/ISRG, synthetic RW) would require re-running the entire
Week 4 backtester on those pairs, which is out of Week 5's scope.

This module reports Week 4's existing NC results and notes the gap.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


WEEK5_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLD_METRICS = WEEK5_ROOT / "data" / "week4_inputs" / "fold_metrics.csv"


def run_dynamic_negative_control(
    fold_metrics_path: Path | str = DEFAULT_FOLD_METRICS,
) -> pd.DataFrame:
    """
    Reports Week 4's existing NC discrimination results.

    Returns a DataFrame with the NC pass/fail count from Week 4
    and a note that dynamic-cost NC is deferred (Option B).
    """
    fm = pd.read_csv(fold_metrics_path)
    if "nc_pass" not in fm.columns:
        return pd.DataFrame([{
            "n_folds": len(fm),
            "nc_pass_count": 0,
            "nc_pass_rate": float("nan"),
            "note": "fold_metrics.csv has no nc_pass column.",
        }])

    pass_count = int(fm["nc_pass"].sum())
    n_folds = (fm["n_trades"] > 0).sum() if "n_trades" in fm.columns else len(fm)

    return pd.DataFrame([{
        "n_traded_folds": int(n_folds),
        "nc_pass_count_static": pass_count,
        "nc_pass_rate_static": pass_count / n_folds if n_folds else float("nan"),
        "nc_dynamic_status": "DEFERRED",
        "note": (
            "Dynamic-cost NC requires re-running Week 4 NC pairs through Plan 2 hooks "
            "with synthetic NC trade timestamps. Week 4 stores only aggregate NC Sharpe, "
            "not per-trade logs. Plan 3 reports Week 4 static-cost NC results above and "
            "flags this as a Week 6 follow-up. (Option B per workflow.)"
        ),
    }])
