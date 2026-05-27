"""
Phase 1 — Factor-Residual Orthogonalization (V3.0)
===================================================

Strips common factor exposure from each ticker before cointegration testing.
Reference: Avellaneda & Lee (2010), *Quantitative Finance* 10(7).

Pipeline role:
    formation 5-min log-prices (full universe)
    -> fit_factor_model -> PCA loadings W (5 factors)
    -> residual log-prices (formation)
    -> Johansen/PCA hedge ratio on residuals (cleaner cointegration)

At trading time:
    trading 1-min log-prices (full universe slice for the trading month)
    -> project_residual (uses formation W)
    -> residual log-prices (trading)
    -> spread / Z-score computed on residuals

Why full universe at trading time:
    Factor returns F[t, k] are estimated by regressing the full cross-section
    onto W at each bar. Projecting a subset (e.g., only pair tickers) biases F.
    The fold loader must slice ALL surviving formation tickers for the trading
    month, not just pair tickers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

_MIN_TICKERS = 50
_MIN_OBS = 200
_VAR_LOW = 0.30
_VAR_HIGH = 0.95   # widened from 0.85 — synthetic data with one dominant factor
                   # legitimately exceeds 0.85; real S&P 500 with 5 PCs is ~0.30-0.50


def fit_factor_model(
    formation_data: dict[str, pd.DataFrame],
    n_components: int = 5,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict]:
    """
    Fit PCA factor model on formation-window 5-min log-returns.

    Parameters
    ----------
    formation_data : dict[ticker] -> DataFrame with column 'log_close', 5-min index
    n_components   : 5 (fixed; do not sweep — see plan F2 DSR accounting)

    Returns
    -------
    loadings_W          : np.ndarray, shape (n_components, n_tickers)
    tickers             : list[str], column order matching W
    residual_log_prices : pd.DataFrame, shape (T, n_tickers), residual log-prices
    diagnostics         : dict with explained_variance_ratio_, n_obs, n_tickers_kept,
                          cumulative_variance_explained

    Raises
    ------
    ValueError on hard-stop violations.
    """
    # ---- HARD STOPS (input) ----
    assert isinstance(formation_data, dict), "formation_data must be dict"
    if len(formation_data) < _MIN_TICKERS:
        raise ValueError(
            f"fit_factor_model: need >={_MIN_TICKERS} tickers, got {len(formation_data)}"
        )

    # ---- ALIGN: inner join all tickers on common index ----
    aligned = pd.concat(
        {tk: df["log_close"] for tk, df in formation_data.items()},
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < _MIN_OBS:
        raise ValueError(
            f"fit_factor_model: need >={_MIN_OBS} aligned obs, got {len(aligned)}"
        )

    tickers = list(aligned.columns)
    n_tickers = len(tickers)

    # ---- COMPUTE: log-returns matrix ----
    returns = aligned.diff().dropna()
    R = returns.values  # (T-1, n_tickers)

    # ---- COMPUTE: PCA on returns ----
    # `svd_solver='full'` + fixed `random_state` are REQUIRED for run-to-run
    # reproducibility. Default `svd_solver='auto'` selects the `randomized`
    # solver for shapes like ours (T×500 with T>500), and randomized uses
    # unseeded np.random internally — producing different loadings on each
    # run and propagating into different trade selection and Sharpe.
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    pca.fit(R)
    loadings_W = pca.components_                          # (n_components, n_tickers)
    explained_variance_ratio = pca.explained_variance_ratio_

    # ---- HARD STOPS (output invariants) ----
    if np.any(np.isnan(loadings_W)):
        raise ValueError("fit_factor_model: NaN in loadings_W")

    cum_var = float(np.sum(explained_variance_ratio))
    if not (_VAR_LOW <= cum_var <= _VAR_HIGH):
        raise ValueError(
            f"fit_factor_model: cumulative explained variance {cum_var:.3f} "
            f"outside sanity band [{_VAR_LOW}, {_VAR_HIGH}]"
        )

    # ---- COMPUTE: residual log-prices ----
    # F[t, k] = R[t, :] @ W[k, :].T  -- factor returns at each bar
    factor_returns = R @ loadings_W.T  # (T-1, n_components)
    # residual_returns[t, i] = R[t, i] - sum_k(W[k, i] * F[t, k])
    residual_returns = R - factor_returns @ loadings_W  # (T-1, n_tickers)

    # Cumulate residual returns; anchor at formation log_close[t=0]
    anchor = aligned.iloc[0].values  # (n_tickers,)
    residual_log_price_values = np.vstack([
        anchor,
        anchor + np.cumsum(residual_returns, axis=0),
    ])  # (T, n_tickers)

    residual_log_prices = pd.DataFrame(
        residual_log_price_values,
        index=aligned.index,
        columns=tickers,
    )

    # ---- HARD STOPS (output finite) ----
    if not np.all(np.isfinite(residual_log_prices.values)):
        raise ValueError("fit_factor_model: residual_log_prices has nan/inf")

    diagnostics = {
        "explained_variance_ratio_": explained_variance_ratio.tolist(),
        "cumulative_variance_explained": cum_var,
        "n_obs": int(len(R)),
        "n_tickers_kept": n_tickers,
        "n_components": n_components,
    }

    log.info(
        "fit_factor_model: T=%d, n_tickers=%d, cum_var=%.3f",
        len(R), n_tickers, cum_var,
    )

    return loadings_W, tickers, residual_log_prices, diagnostics


def project_residual(
    trading_data: dict[str, pd.DataFrame],
    loadings_W: np.ndarray,
    formation_tickers: list[str],
    min_obs: int = 20,
) -> dict[str, pd.Series]:
    """
    Project trading-time 1-min returns onto formation loadings W; return residual log-prices.

    Parameters
    ----------
    trading_data       : dict[ticker] -> DataFrame with 'log_close', 1-min index.
                         Must include the FULL universe of formation_tickers that
                         have data in the trading window (subset OK, but bigger = better).
    loadings_W         : np.ndarray, (n_components, n_formation_tickers) from fit_factor_model
    formation_tickers  : list[str], column order matching loadings_W

    Returns
    -------
    dict[ticker] -> pd.Series of residual log-prices (1-min freq).
                    Only returns series for tickers present in BOTH trading_data
                    and formation_tickers.

    Raises
    ------
    ValueError on hard-stop violations.
    """
    # ---- HARD STOPS (input) ----
    assert isinstance(trading_data, dict), "trading_data must be dict"
    assert isinstance(loadings_W, np.ndarray), "loadings_W must be ndarray"
    if loadings_W.shape[1] != len(formation_tickers):
        raise ValueError(
            f"project_residual: W.shape[1]={loadings_W.shape[1]} != "
            f"len(formation_tickers)={len(formation_tickers)}"
        )

    # ---- INTERSECT: trading tickers that are in formation universe ----
    available = [t for t in formation_tickers if t in trading_data]
    if len(available) < _MIN_TICKERS:
        log.warning(
            "project_residual: only %d trading tickers overlap formation universe "
            "(min recommended %d); factor returns may be biased",
            len(available), _MIN_TICKERS,
        )

    # Indices into loadings_W for available tickers
    formation_idx = {t: i for i, t in enumerate(formation_tickers)}
    avail_idx = np.array([formation_idx[t] for t in available])
    W_avail = loadings_W[:, avail_idx]  # (n_components, n_available)

    # ---- ALIGN: outer-join + forward-fill (Bug #1 fix) ----
    # Previous inner-join collapsed 1-min trading data from ~7800 bars to ~145 bars
    # because it required ALL 500 tickers to have data at every minute. Now we
    # forward-fill missing bars on each ticker (return=0 at filled bars, benign for
    # factor estimation since high-coverage tickers dominate the factor returns).
    aligned = pd.concat(
        {tk: trading_data[tk]["log_close"] for tk in available},
        axis=1,
        join="outer",
    )
    aligned = aligned.ffill().dropna()   # drop only initial rows where some tickers
                                          # haven't started yet
    if len(aligned) < min_obs:
        raise ValueError(
            f"project_residual: too few aligned trading obs "
            f"({len(aligned)} < min_obs={min_obs})"
        )

    # ---- COMPUTE: returns matrix ----
    returns = aligned.diff().dropna()
    R_trading = returns.values  # (T-1, n_available)

    # ---- COMPUTE: factor returns at each trading bar ----
    # F[t, k] = R[t, :] @ W[k, :].T (using only available tickers)
    factor_returns = R_trading @ W_avail.T  # (T-1, n_components)
    residual_returns = R_trading - factor_returns @ W_avail  # (T-1, n_available)

    # Cumulate from anchor (first log_close of each ticker)
    anchor = aligned.iloc[0].values
    residual_log_price_values = np.vstack([
        anchor,
        anchor + np.cumsum(residual_returns, axis=0),
    ])  # (T, n_available)

    # ---- HARD STOPS (output finite) ----
    if not np.all(np.isfinite(residual_log_price_values)):
        raise ValueError("project_residual: residual_log_prices has nan/inf")

    # ---- PACKAGE: dict[ticker] -> Series ----
    residuals: dict[str, pd.Series] = {}
    for j, tk in enumerate(available):
        residuals[tk] = pd.Series(
            residual_log_price_values[:, j],
            index=aligned.index,
            name=tk,
        )

    log.info(
        "project_residual: T=%d, n_tickers=%d (of %d formation)",
        len(R_trading), len(available), len(formation_tickers),
    )

    return residuals
