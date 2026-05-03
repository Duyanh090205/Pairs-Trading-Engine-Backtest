"""
Phase 4 — Orchestrator

Defines the canonical fold schedule, regime map, and result-loading utilities.
The full pipeline has already been run (run_full_pipeline.py); this module
loads the persisted outputs for downstream analysis.

Re-run entrypoint: run_orchestrator() delegates to run_full_pipeline.py logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT         = Path(__file__).parents[2]
_BASELINE_DIR = _ROOT / "results" / "metrics" / "baseline"
_EQUITY_DIR   = _BASELINE_DIR / "equities"
_METRICS_DIR  = _BASELINE_DIR / "phase4"      # Phase 4 analytics output
_LOGS_DIR     = _ROOT / "results" / "logs"
_FIGURES_DIR  = _ROOT / "results" / "figures" / "baseline"
_PHASE1_DIR   = _ROOT / "results" / "metrics" / "phase1_folds"


# ---------------------------------------------------------------------------
# Fold Schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldSpec:
    fold_n:         int
    form_start:     str   # YYYY-MM-DD
    form_end:       str   # YYYY-MM-DD
    trade_year:     int
    trade_month:    int   # 1..12
    trading_month:  str   # "YYYY-MM"

    @property
    def trading_label(self) -> str:
        return self.trading_month


def _build_fold_schedule() -> list[FoldSpec]:
    """
    Generate 45-fold schedule.

    Fold N: formation starts at beginning of (January 2022 + N-1 months),
    spans 6 months, trading is the month immediately after formation end.

    Fold 1: formation 2022-01-03 → 2022-06-30, trading 2022-07
    Fold 2: formation 2022-02-01 → 2022-07-31, trading 2022-08
    ...
    Fold 45: formation 2025-09-01 → 2026-02-28, trading 2026-03
    """
    folds: list[FoldSpec] = []
    base = pd.Timestamp("2022-01-01")

    for i in range(45):
        fold_n = i + 1

        # Formation start: first of month (Fold 1 uses 2022-01-03)
        form_start_dt = base + pd.DateOffset(months=i)
        if fold_n == 1:
            form_start_dt = pd.Timestamp("2022-01-03")

        # Formation end: last day of the 6th calendar month from form_start.
        # Always anchor to month-start to handle Fold 1 (2022-01-03 start)
        # correctly — otherwise +6 months from Jan 3 = Jul 3, not Jun 30.
        month_anchor = pd.Timestamp(form_start_dt.year, form_start_dt.month, 1)
        form_end_dt  = month_anchor + pd.DateOffset(months=6) - pd.DateOffset(days=1)

        # Trading month: calendar month immediately after formation end
        trade_dt = pd.Timestamp(form_end_dt.year, form_end_dt.month, 1) + pd.DateOffset(months=1)

        folds.append(FoldSpec(
            fold_n       = fold_n,
            form_start   = form_start_dt.strftime("%Y-%m-%d"),
            form_end     = form_end_dt.strftime("%Y-%m-%d"),
            trade_year   = trade_dt.year,
            trade_month  = trade_dt.month,
            trading_month = trade_dt.strftime("%Y-%m"),
        ))

    return folds


FOLD_SCHEDULE: list[FoldSpec] = _build_fold_schedule()
FOLD_BY_N: dict[int, FoldSpec] = {f.fold_n: f for f in FOLD_SCHEDULE}


# ---------------------------------------------------------------------------
# Regime Map
# ---------------------------------------------------------------------------

REGIME_MAP: dict[str, list[int]] = {
    "Late Bear 2022":        list(range(1,  7)),   # Trading Jul–Dec 2022
    "Early Bull 2023":       list(range(7,  19)),  # Trading Jan–Dec 2023
    "Mid Bull 2024":         list(range(19, 31)),  # Trading Jan–Dec 2024
    "Late Bull 2025-Q12026": list(range(31, 46)),  # Trading Jan 2025–Mar 2026
}

FOLD_TO_REGIME: dict[int, str] = {
    fold_n: regime
    for regime, folds in REGIME_MAP.items()
    for fold_n in folds
}


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------

def load_fold_metrics() -> pd.DataFrame:
    """Load aggregate per-fold metrics from fold_metrics.csv."""
    path = _BASELINE_DIR / "fold_metrics.csv"
    df = pd.read_csv(path)
    df["regime"] = df["fold"].map(FOLD_TO_REGIME)
    return df


def load_equity(fold_n: int) -> Optional[pd.DataFrame]:
    """Load bar-level equity curve for a fold (None if not present)."""
    path = _EQUITY_DIR / f"fold{fold_n:02d}_equity.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_phase1_pairs(fold_n: int) -> Optional[pd.DataFrame]:
    """Load Phase 1 surviving pairs for a fold (None if not present)."""
    path = _PHASE1_DIR / f"fold_{fold_n:02d}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def load_all_fold_equities() -> dict[int, pd.DataFrame]:
    """Load all available equity curves keyed by fold number."""
    result: dict[int, pd.DataFrame] = {}
    for fold_n in range(1, 46):
        eq = load_equity(fold_n)
        if eq is not None:
            result[fold_n] = eq
    return result


def load_equity_concat() -> pd.DataFrame:
    """
    Return concatenated bar-level equity curve across all completed folds.
    Uses the pre-built equity_full.parquet if available, otherwise rebuilds.
    """
    full_path = _BASELINE_DIR / "equity_full.parquet"
    if full_path.exists():
        return pd.read_parquet(full_path)

    # Rebuild from per-fold files
    frames = []
    for fold_n in range(1, 46):
        eq = load_equity(fold_n)
        if eq is not None:
            eq = eq.copy()
            eq["fold"] = fold_n
            frames.append(eq)

    if not frames:
        return pd.DataFrame()

    concat = pd.concat(frames).sort_index()
    concat.to_parquet(full_path)
    return concat


def get_completed_folds(fold_metrics: Optional[pd.DataFrame] = None) -> list[int]:
    """Return list of fold numbers that completed (have metrics)."""
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()
    return sorted(fold_metrics["fold"].tolist())


def get_daily_returns(fold_n: int) -> Optional[pd.Series]:
    """Compute daily log-returns from bar-level equity curve."""
    eq = load_equity(fold_n)
    if eq is None or eq.empty:
        return None
    col = "equity" if "equity" in eq.columns else eq.columns[0]
    daily = eq[col].resample("B").last().dropna()
    if len(daily) < 2:
        return None
    return daily.pct_change().dropna()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_fold_summary(fold_metrics: Optional[pd.DataFrame] = None) -> None:
    """Print a quick summary of completed folds by regime."""
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    total = len(FOLD_SCHEDULE)
    done  = len(fold_metrics)
    skipped = total - done
    pct_pos = (fold_metrics["sharpe"] > 0).mean()

    print(f"\n=== Fold Summary ===")
    print(f"  Total folds:     {total}")
    print(f"  Completed:       {done}")
    print(f"  Skipped:         {skipped}")
    print(f"  Median Sharpe:   {fold_metrics['sharpe'].median():.3f}")
    print(f"  Mean Sharpe:     {fold_metrics['sharpe'].mean():.3f}")
    print(f"  % Positive:      {pct_pos:.1%}")

    print(f"\n  By regime:")
    for regime, folds_in in REGIME_MAP.items():
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)]
        if sub.empty:
            print(f"    {regime:30s}: no data")
            continue
        print(f"    {regime:30s}: N={len(sub):2d}  "
              f"mean={sub['sharpe'].mean():.2f}  "
              f"median={sub['sharpe'].median():.2f}  "
              f"pos={( sub['sharpe']>0).mean():.0%}")
