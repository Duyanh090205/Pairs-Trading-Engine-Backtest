"""
Z-Score Computation

Vectorized rolling Z-score using pandas Cython accumulators (O(N) time).
"""
import numpy as np
import pandas as pd


def compute_zscore(
    spread: pd.Series,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """Rolling Z-score of a spread series.

    z(t) = (spread(t) - rolling_mean(t)) / max(rolling_std(t), eps)

    Uses ddof=1 (sample standard deviation) as specified in the research plan.
    The eps guard prevents division-by-zero when the spread is locally constant
    (e.g. a trading halt that slipped through the session filter).

    Args:
        spread      : Spread series produced by compute_spread().
        window      : Look-back window in bars. Derive from half-life via
                      window_from_half_life(); do NOT recalculate on the
                      trading period — that would be lookahead.
        min_periods : Minimum bars required before emitting a Z-score.
                      Defaults to window // 2, so the first valid Z-score
                      appears after half the window has filled.
        eps         : Floor applied to rolling_std before division.
                      Prevents ±Inf Z-scores during low-volatility patches.

    Returns:
        DataFrame with three columns:
            rolling_mean  — rolling window mean of spread
            rolling_std   — rolling window std (ddof=1)
            zscore        — standardised spread value

        The first (min_periods - 1) rows will have NaN in all columns.
        The pipeline's state machine treats NaN Z-scores as "hold current
        position" — see state_machine.py.

    !! FLAG — min_periods choice:
        window // 2 means the Z-score starts emitting halfway through the
        first window.  Those early values are estimated from fewer bars than
        the window implies, so their std is noisier.  For a strict burn-in
        (no signal until the window is fully filled) pass min_periods=window.
        The research plan uses window // 2 for more usable bars at period
        boundaries, which is a reasonable trade-off on 47k-bar datasets.
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = spread.rolling(window, min_periods=min_periods).mean()
    rolling_std  = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    zscore       = (spread - rolling_mean) / rolling_std.clip(lower=eps)

    return pd.DataFrame(
        {
            "rolling_mean": rolling_mean,
            "rolling_std":  rolling_std,
            "zscore":       zscore,
        },
        index=spread.index,
    )


def apply_session_warmup(zdf: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    """Suppress Z-score for the first `warmup_bars` of each calendar session.

    Overnight price gaps cause the first bars of each session to have a spread
    that jumps relative to the prior session's rolling statistics, producing
    artificial Z-score extremes.  Setting those bars to NaN causes the state
    machine to hold its current position (flat at open) rather than entering
    on gap-driven noise.

    Args:
        zdf          : DataFrame from compute_zscore() — must have a DatetimeIndex
                       and a 'zscore' column.
        warmup_bars  : Number of bars to suppress at each session open.
                       30 bars = first 30 minutes of each session.
                       0 = disabled (returns zdf unchanged).

    Returns:
        Copy of zdf with zscore set to NaN for the first `warmup_bars` of each
        unique calendar date.  rolling_mean and rolling_std are left intact so
        the suppression is visible only in the signal, not in the statistics.

    !! FLAG — session detection:
        Sessions are identified by calendar date via index.normalize() (midnight
        of each day in the index timezone).  This is correct for a DST-aware UTC
        index that has already been session-filtered to 09:30-16:00 ET — the
        first bar of each date is always the 09:30 bar.  On early-close days
        (e.g. day before Thanksgiving) the session is shorter, but the first-bar
        detection is unaffected.
    """
    if warmup_bars <= 0:
        return zdf

    df  = zdf.copy()
    z_col = df.columns.get_loc("zscore")

    # Find the integer position of the first bar of each calendar date
    dates      = df.index.normalize()
    unique_dates = np.unique(dates)
    for d in unique_dates:
        locs = np.where(dates == d)[0]
        end  = min(locs[0] + warmup_bars, len(df))
        df.iloc[locs[0]:end, z_col] = np.nan

    return df
