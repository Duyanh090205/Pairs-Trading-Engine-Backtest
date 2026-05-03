"""
Date and Time manipulation helpers
"""
import pandas as pd


def split_periods(
    df: pd.DataFrame,
    formation_start: str,
    formation_end: str,
    trading_start: str,
    trading_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an aligned pair DataFrame into formation and trading period slices.

    The no-lookahead invariant is enforced: formation_end must be strictly
    before trading_start. If violated, a ValueError is raised immediately
    before any slice is returned.

    Args:
        df              : Merged pair DataFrame with a UTC DatetimeIndex.
        formation_start : ISO date string, e.g. "2022-01-01"
        formation_end   : ISO date string, e.g. "2022-06-30"
        trading_start   : ISO date string, e.g. "2022-07-01"
        trading_end     : ISO date string, e.g. "2022-12-31"

    Returns:
        (formation, trading) — both are label-based slices of df retaining
        the same column structure and DatetimeIndex.
    """
    # Convert boundary strings to tz-aware Timestamps (accept pre-built Timestamps too)
    def _to_ts(val: str | pd.Timestamp) -> pd.Timestamp:
        if isinstance(val, pd.Timestamp):
            return val if val.tzinfo is not None else val.tz_localize("UTC")
        return pd.Timestamp(val, tz="UTC")

    fs = _to_ts(formation_start)
    fe = _to_ts(formation_end)
    ts = _to_ts(trading_start)
    te = _to_ts(trading_end)

    # Core no-lookahead guard
    if fe >= ts:
        raise ValueError(
            f"Lookahead violation: formation_end ({fe}) must be strictly before "
            f"trading_start ({ts}). Fix the date boundaries in your config."
        )

    formation = df.loc[fs:fe]
    trading = df.loc[ts:te]

    if formation.empty:
        raise ValueError(
            f"Formation slice is empty. Check formation dates [{fs}, {fe}] "
            f"against the DataFrame range [{df.index.min()}, {df.index.max()}]."
        )
    if trading.empty:
        raise ValueError(
            f"Trading slice is empty. Check trading dates [{ts}, {te}] "
            f"against the DataFrame range [{df.index.min()}, {df.index.max()}]."
        )

    return formation, trading
