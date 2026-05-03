"""
Statistical utilities shared across phases:
  - OU half-life (AR(1) regression on spread differences)
  - BH-FDR multiple testing correction
  - Block bootstrap Sharpe distribution (for negative control threshold)
  - Rolling Z-score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------

def compute_ou_halflife(spread: pd.Series | np.ndarray, bars_per_day: int = 78) -> float:
    """
    Fit OU half-life via AR(1) regression on spread differences.

    Regression: ΔS_t = κ · S_{t-1} + c + ε
    Mean-reversion speed: θ = -κ  (κ must be < 0 for mean-reversion)
    Half-life in bars: ln(2) / θ
    Half-life in days: half_life_bars / bars_per_day

    Returns np.nan if spread is not mean-reverting (κ ≥ 0).

    bars_per_day:
        78  for 5-min bars (Phase 1 default, per pipeline spec §1.6)
        390 for 1-min bars (Phase 2)
    """
    s = np.asarray(spread, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return np.nan

    diff = np.diff(s)
    lag = s[:-1]

    X = sm.add_constant(lag)
    try:
        res = sm.OLS(diff, X).fit()
    except Exception:
        return np.nan

    kappa = float(res.params[1])  # coefficient on lagged level (index 0 = constant)
    if kappa >= 0:
        return np.nan  # not mean-reverting

    theta = -kappa
    half_life_bars = np.log(2) / theta
    return half_life_bars / bars_per_day


# ---------------------------------------------------------------------------
# BH-FDR multiple testing correction
# ---------------------------------------------------------------------------

def bh_fdr_correct(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns a boolean array: True = pair survives after correction.
    NaN p-values are treated as 1.0 (no rejection).
    """
    pvals = np.asarray(pvalues, dtype=float)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    reject, _, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return reject.astype(bool)


# ---------------------------------------------------------------------------
# Block bootstrap Sharpe (for NC threshold, Phase 3)
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe(
    daily_returns: pd.Series | np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap distribution of annualized Sharpe ratios.

    block_size = 1 trading day (default) preserves daily serial correlation
    of pairs strategy returns.

    Returns array of n_bootstrap Sharpe values.
    """
    rets = np.asarray(daily_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n < 2:
        return np.zeros(n_bootstrap)

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    sharpes = np.empty(n_bootstrap)
    offsets = np.arange(block_size)  # reused each iteration

    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Vectorized gather: shape (n_blocks, block_size) → flatten → trim to n
        idx = (starts[:, None] + offsets).ravel()[:n]
        resampled = rets[idx]
        std = resampled.std(ddof=1)
        sharpes[i] = 0.0 if std < 1e-10 else (resampled.mean() / std) * np.sqrt(252)

    return sharpes


# ---------------------------------------------------------------------------
# Rolling Z-score (vectorized)
# ---------------------------------------------------------------------------

def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling Z-score: (x - rolling_mean) / rolling_std.

    min_periods defaults to window // 2 (burn-in period per pipeline spec).
    """
    mp = min_periods if min_periods is not None else window // 2
    mean = series.rolling(window, min_periods=mp).mean()
    std = series.rolling(window, min_periods=mp).std(ddof=1).clip(lower=1e-10)
    return (series - mean) / std
