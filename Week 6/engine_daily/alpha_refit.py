"""
Alpha refit at trading-window start (Q2 of V4 plan).

Discovery fits beta via PCA hedge ratio on full formation residuals; alpha is
defined there as `alpha_pca = mean(resid_a) - beta * mean(resid_b)`. By the start
of the trading window, the residual series anchor has shifted (project_residual
re-anchors at trading[0]); Path A in V3 fixed the level break, but the *long-run*
mean within the trading window can still differ from the full-formation mean.

Q2 fix: refit alpha (and only alpha — keep beta from formation) on the LAST N
daily bars of formation, which is the most recent estimate of the spread's
center going into trading. Cheap, no Johansen rerun, no risk of overfitting.

    alpha_new = mean(resid_a[-N:] - beta * resid_b[-N:])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def recompute_alpha(
    resid_a: pd.Series,
    resid_b: pd.Series,
    beta: float,
    n_lookback: int = 60,
) -> float:
    """Recompute alpha from the last `n_lookback` bars of residual log-prices.

    Parameters
    ----------
    resid_a, resid_b : pd.Series, daily residual log-prices (formation window).
                       Must be index-aligned.
    beta             : float, formation-fit beta_pca (>0 per V3 R5 filter).
    n_lookback       : int, days of recent formation history to use.

    Returns
    -------
    alpha : float, the recomputed centering term.
            spread = resid_a - alpha - beta * resid_b is centered ~0 on the lookback.
    """
    # ---- HARD STOPS ----
    if not isinstance(resid_a, pd.Series) or not isinstance(resid_b, pd.Series):
        raise ValueError("recompute_alpha: resid_a, resid_b must be pd.Series")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"recompute_alpha: invalid beta={beta} (must be finite, >0)")
    if n_lookback < 10:
        raise ValueError(f"recompute_alpha: n_lookback={n_lookback} too small")

    # Align (inner) and drop missing
    df = pd.concat([resid_a, resid_b], axis=1, join="inner").dropna()
    df.columns = ["a", "b"]
    if len(df) < n_lookback:
        # Use whatever's available; log a warning rather than fail. Daily formation
        # is finite (~252 bars) so we may have less than 60 in degenerate folds.
        log.warning("recompute_alpha: only %d aligned bars (< n_lookback=%d), "
                    "using all available", len(df), n_lookback)
        tail = df
    else:
        tail = df.tail(n_lookback)

    # alpha = mean of (resid_a - beta * resid_b) over the tail window
    spread_uncentered = tail["a"].values - beta * tail["b"].values
    alpha = float(spread_uncentered.mean())

    if not np.isfinite(alpha):
        raise ValueError(f"recompute_alpha: non-finite output alpha={alpha}")
    return alpha
