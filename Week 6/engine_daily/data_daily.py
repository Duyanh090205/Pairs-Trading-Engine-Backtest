"""
Daily bar construction from 5-min validated data.

Daily close-to-close bars are built by taking the LAST log_close of each
NYSE session per ticker (the 15:55-16:00 5-min bar in regular sessions,
or the last bar of the day on half-days). Volume is summed across the session.

Output index is timezone-naive `pd.Timestamp` at midnight, conforming to the
daily convention (e.g. `2022-01-03 00:00:00`). This matches how Avellaneda &
Lee (2010) and most stat-arb literature index daily series.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def resample_5min_to_daily(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample one ticker's 5-min OHLCV-style frame to daily close-to-close.

    Parameters
    ----------
    df_5min : DataFrame indexed by tz-aware 5-min timestamps. Must contain
              `log_close` column. `volume` is optional (summed if present).

    Returns
    -------
    DataFrame indexed by tz-naive daily Timestamp (midnight), columns:
        log_close : last log_close of each session
        volume    : sum of volume across the session (if input had volume)
    """
    # ---- HARD STOPS ----
    if not isinstance(df_5min.index, pd.DatetimeIndex):
        raise ValueError("resample_5min_to_daily: index must be DatetimeIndex")
    if "log_close" not in df_5min.columns:
        raise ValueError("resample_5min_to_daily: missing 'log_close' column")
    if len(df_5min) == 0:
        return pd.DataFrame(columns=["log_close", "volume"])

    # Group by session date (date in the index's tz). Using .date avoids tz
    # weirdness around DST transitions when summing within a single session.
    session_date = df_5min.index.date  # numpy array of python date objects
    grouped = df_5min.groupby(session_date)

    agg = {"log_close": "last"}
    if "volume" in df_5min.columns:
        agg["volume"] = "sum"
    daily = grouped.agg(agg)

    # Cast date index to tz-naive Timestamp at midnight (standard daily convention).
    daily.index = pd.to_datetime(list(daily.index))
    daily.index.name = "date"
    return daily


def load_daily_universe(
    cache_5min: dict[str, pd.DataFrame],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """
    Build a daily-bar universe from a pre-loaded 5-min cache, sliced to [start, end].

    Parameters
    ----------
    cache_5min : dict ticker -> 5-min DataFrame (from `_load_all_tickers`)
    start, end : YYYY-MM-DD strings or Timestamps. Inclusive on both ends.

    Returns
    -------
    dict ticker -> daily DataFrame (log_close, volume). Tickers with no bars
    in the window are dropped.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: dict[str, pd.DataFrame] = {}
    # sorted() — deterministic iteration order
    for tk in sorted(cache_5min.keys()):
        df_5 = cache_5min[tk]
        # Pre-slice the 5-min frame to roughly the window (tz-aware compare).
        # The .date() comparison absorbs tz offset.
        idx_dates = pd.Series(df_5.index.date, index=df_5.index)
        mask = (idx_dates >= start_ts.date()) & (idx_dates <= end_ts.date())
        sliced = df_5[mask.values]
        if len(sliced) == 0:
            continue
        daily = resample_5min_to_daily(sliced)
        if len(daily) > 0:
            out[tk] = daily
    return out
