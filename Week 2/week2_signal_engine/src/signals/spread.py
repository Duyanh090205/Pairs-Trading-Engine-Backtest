"""
Spread Construction

OLS hedge ratio estimation (formation period ONLY) and spread computation.
"""
import numpy as np
import pandas as pd


def estimate_hedge_ratio(
    close_a: pd.Series | np.ndarray,
    close_b: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """OLS regression: log(A) = alpha + beta * log(B) + epsilon.

    Must be called on the FORMATION period only.  Calling this on the trading
    period or the full series would introduce lookahead bias.

    Uses np.linalg.lstsq (numerically stable, no scipy dependency).

    Args:
        close_a : Price series for leg A (levels, not log-transformed).
        close_b : Price series for leg B (levels, not log-transformed).

    Returns:
        (alpha, beta) — intercept and slope of the OLS fit.

    !! FLAG: lstsq silently handles near-singular design matrices, but if both
    series have near-zero variance (e.g. a trading halt day slipped through the
    session filter) the returned beta can be numerically garbage.  The caller
    should verify beta > 0 for economically plausible pairs.
    """
    y = np.log(np.asarray(close_a, dtype=np.float64))
    x = np.log(np.asarray(close_b, dtype=np.float64))
    X = np.column_stack([np.ones(len(x)), x])
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(params[0]), float(params[1])
    return alpha, beta


def compute_spread(
    close_a: pd.Series | np.ndarray,
    close_b: pd.Series | np.ndarray,
    alpha: float | pd.Series | np.ndarray,
    beta:  float | pd.Series | np.ndarray,
) -> pd.Series:
    """Spread = log(A) - alpha - beta * log(B).

    The hedge ratio (alpha, beta) must come from the formation period —
    either as a scalar (static OLS) or as a same-length Series / array
    (Kalman filter dynamic hedge ratio).

    When alpha/beta are pd.Series, their pandas index is stripped before
    the arithmetic so that numpy element-wise broadcasting is used rather
    than pandas label alignment (which would produce NaN wherever indices
    differ in type or value).

    Returns a pd.Series preserving the index of close_a if it is a Series,
    otherwise returns a plain Series with a default RangeIndex.
    """
    log_a = np.log(np.asarray(close_a, dtype=np.float64))
    log_b = np.log(np.asarray(close_b, dtype=np.float64))

    # Strip pandas index from alpha/beta if Series — prevents label alignment
    # issues when mixing pd.Series with raw numpy arrays in arithmetic.
    if isinstance(alpha, pd.Series):
        alpha = alpha.values
    if isinstance(beta, pd.Series):
        beta = beta.values

    spread_vals = log_a - alpha - beta * log_b

    if isinstance(close_a, pd.Series):
        return pd.Series(spread_vals, index=close_a.index, name="spread")
    return pd.Series(spread_vals, name="spread")
