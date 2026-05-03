"""
Per-fold kappa calibration. Mirrors Week 4 FOLD_SCHEDULE.
"""
import pandas as pd
from typing import NamedTuple
from src.plan1_cost_model.impact_cost import assign_kappa_tier


class FoldSpec(NamedTuple):
    fold_id: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_label: str


def _build_fold_schedule() -> list[FoldSpec]:
    """
    Reproduces Week 4's 45-fold schedule.
    Source: Week 4/src/phase4_defense/orchestrator.py:_build_fold_schedule().
    """
    folds: list[FoldSpec] = []
    base = pd.Timestamp("2022-01-01")
    for i in range(45):
        fold_n = i + 1
        form_start = base + pd.DateOffset(months=i)
        if fold_n == 1:
            form_start = pd.Timestamp("2022-01-03")
        anchor = pd.Timestamp(form_start.year, form_start.month, 1)
        form_end = anchor + pd.DateOffset(months=6) - pd.DateOffset(days=1)
        trade_start = pd.Timestamp(form_end.year, form_end.month, 1) + pd.DateOffset(months=1)
        folds.append(FoldSpec(
            fold_id=fold_n,
            formation_start=form_start.tz_localize("US/Eastern"),
            formation_end=(form_end + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_localize("US/Eastern"),
            trading_start=trade_start.tz_localize("US/Eastern"),
            trading_label=trade_start.strftime("%Y-%m"),
        ))
    return folds


FOLD_SCHEDULE: list[FoldSpec] = _build_fold_schedule()
FOLD_BY_ID: dict[int, FoldSpec] = {f.fold_id: f for f in FOLD_SCHEDULE}


def calibrate_fold(
    fold_id: int,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    trading_start: pd.Timestamp,
    tickers: list[str],
    spreads_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Returns {ticker: kappa}. Uses ONLY [formation_start, formation_end] data.

    Parameters
    ----------
    spreads_df : DataFrame
        Must have columns timestamp_et, ticker, full_spread_l1_bps, is_valid.
        NOT MultiIndexed for this function (we filter by columns).

    Raises
    ------
    ValueError
        If formation_end >= trading_start (lookahead guard).
    """
    if formation_end >= trading_start:
        raise ValueError(
            f"Lookahead violation in fold {fold_id}: "
            f"formation_end {formation_end} >= trading_start {trading_start}"
        )

    formation_slice = spreads_df[
        (spreads_df["timestamp_et"] >= formation_start)
        & (spreads_df["timestamp_et"] <= formation_end)
        & (spreads_df["is_valid"])
        & (spreads_df["ticker"].isin(tickers))
    ]

    medians = formation_slice.groupby("ticker")["full_spread_l1_bps"].median()

    # Tickers without formation data: assign Tier 2 (kappa=0.5) as neutral default
    kappa_map: dict[str, float] = {}
    for t in tickers:
        med = medians.get(t)
        if pd.isna(med) or med is None:
            kappa_map[t] = 0.5
        else:
            kappa_map[t] = assign_kappa_tier(float(med))
    return kappa_map
