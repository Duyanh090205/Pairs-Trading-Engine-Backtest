"""NYSE trading calendar helpers.

Backtest implicitly uses NYSE trading days (Polygon data already excludes
weekends/holidays). Live engine must use the SAME calendar — otherwise we'd
treat half-days / holidays as "stale bar gap" and unnecessarily halt.

Half-days handled correctly: NYSE early close at 13:00 ET on certain dates
(Black Friday, day before July 4, Christmas Eve). The calendar lib knows.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


@lru_cache(maxsize=1024)
def is_trading_day(d: date) -> bool:
    sched = _NYSE.schedule(start_date=d, end_date=d)
    return len(sched) > 0


@lru_cache(maxsize=8)
def trading_days_in_month(year: int, month: int) -> list[date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    sched = _NYSE.schedule(start_date=start, end_date=end)
    return [ts.date() for ts in sched.index if ts.date().month == month]


def last_trading_day_of_month(year: int, month: int) -> date:
    days = trading_days_in_month(year, month)
    if not days:
        raise ValueError(f"No trading days in {year}-{month:02d}")
    return days[-1]


def is_last_trading_day_of_month(d: date) -> bool:
    if not is_trading_day(d):
        return False
    return d == last_trading_day_of_month(d.year, d.month)


def session_close_utc(d: date) -> datetime:
    """Return market-close timestamp (UTC) for date d. Handles half-days."""
    sched = _NYSE.schedule(start_date=d, end_date=d)
    if len(sched) == 0:
        raise ValueError(f"{d} is not a trading day")
    close = sched.iloc[0]["market_close"]
    if close.tzinfo is None:
        close = close.tz_localize("UTC")
    else:
        close = close.tz_convert("UTC")
    return close.to_pydatetime()


def is_half_day(d: date) -> bool:
    """True if NYSE closes early (13:00 ET = 18:00 UTC standard, 17:00 UTC DST)."""
    if not is_trading_day(d):
        return False
    close = session_close_utc(d)
    # Normal close = 21:00 UTC (16:00 ET DST) or 20:00 UTC (winter). Half-day = 18:00/17:00.
    close_hour_utc = close.hour
    return close_hour_utc < 19   # rough cutoff — half-days close before 19:00 UTC
