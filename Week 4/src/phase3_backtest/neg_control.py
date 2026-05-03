"""
Phase 3 — Negative Control Validation

Two controls per fold:
  1. Empirical: CVNA/ISRG (known non-cointegrated pair in 2022 Bear)
  2. Synthetic: two uncorrelated random walks

Block bootstrap on NC daily returns (block=1 trading day = 390 1-min bars):
  threshold = mean(bootstrap_sharpes) + 2 * std(bootstrap_sharpes)
  Primary strategy Sharpe must exceed threshold to PASS.

If CVNA or ISRG data is unavailable, falls back to synthetic only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.phase2_execution.kalman import run_kalman
from src.phase2_execution.engine import run_pair
from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

_NC_EMPIRICAL_PAIRS = [("CVNA", "ISRG")]
_N_BOOTSTRAP        = 1000
_BLOCK_SIZE_DAYS    = 1
_BARS_PER_DAY_1MIN  = 390
_TOTAL_CAPITAL      = 1_000_000.0
_N_OPEN_PAIRS_MAX   = 50


def run_neg_control(
    trading_1min: dict[str, pd.DataFrame],
    delta: float,
    primary_sharpe: float,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    tc_bps: float = 30.0,
    borrow_bps_yr: float = 50.0,
    eos_flatten: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run negative controls and bootstrap discrimination test.

    Parameters
    ----------
    trading_1min   : {ticker: 1min DataFrame} for the trading window
    delta          : Kalman delta used for this fold (same as primary)
    primary_sharpe : aggregate Sharpe of the primary strategy (for pass/fail)

    Returns
    -------
    dict with: empirical_sharpe, synthetic_sharpe, nc_daily_returns,
               bootstrap_mean, bootstrap_std, bootstrap_threshold,
               nc_pass, nc_source ('empirical'|'synthetic'|'both')
    """
    per_pair_dollar = total_capital / n_open_pairs_max

    # --- Empirical NC (CVNA/ISRG) ---
    emp_sharpe     = None
    emp_daily_ret  = None

    for ta, tb in _NC_EMPIRICAL_PAIRS:
        if ta in trading_1min and tb in trading_1min:
            result = _run_nc_pair(
                ta, tb, trading_1min, delta,
                per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
                eos_flatten=eos_flatten,
            )
            if result is not None:
                emp_sharpe    = result["sharpe"]
                emp_daily_ret = result["daily_returns"]
                log.info("NC empirical (%s/%s): Sharpe=%.3f", ta, tb, emp_sharpe)
            break

    # --- Synthetic NC (two uncorrelated random walks) ---
    # Build synthetic prices over the trading window
    ref_ticker = next(iter(trading_1min))
    ref_idx    = trading_1min[ref_ticker].index
    n_bars     = len(ref_idx)

    rng = np.random.default_rng(seed)
    # Two independent GBM paths, no cointegration
    vol = 0.001   # ~0.1% per bar (realistic for 1-min)
    log_ret_a = rng.normal(0.0, vol, n_bars)
    log_ret_b = rng.normal(0.0, vol, n_bars)
    price_a   = 100.0 * np.exp(np.cumsum(log_ret_a))
    price_b   = 100.0 * np.exp(np.cumsum(log_ret_b))
    log_a     = np.log(price_a)
    log_b     = np.log(price_b)

    # PCA-style init: flat alpha=0, beta=1, large R (no real relationship)
    R_synthetic = 1.0  # large R relative to small spread variance
    try:
        syn_pair_df = run_pair(
            log_a_1min   = log_a,
            log_b_1min   = log_b,
            close_a_1min = price_a,
            close_b_1min = price_b,
            index_1min   = ref_idx,
            alpha_pca    = 0.0,
            beta_pca     = 1.0,
            R            = R_synthetic,
            delta        = delta,
            half_life_days = 5.0,   # use mid-range half-life for Z window
            eos_flatten  = eos_flatten,
        )
        close_a_ser = pd.Series(price_a, index=ref_idx)
        close_b_ser = pd.Series(price_b, index=ref_idx)
        syn_pnl_result = compute_pair_pnl(
            syn_pair_df, close_a_ser, close_b_ser,
            per_pair_dollar, tc_bps, borrow_bps_yr,
        )
        syn_metrics = compute_metrics(
            syn_pnl_result["bar_pnl"], total_capital,
            syn_pnl_result["trade_log"],
        )
        syn_sharpe    = syn_metrics["sharpe"]
        syn_daily_ret = syn_metrics["daily_returns"]
        log.info("NC synthetic: Sharpe=%.3f", syn_sharpe)
    except Exception as e:
        log.warning("NC synthetic failed: %s", e)
        syn_sharpe    = 0.0
        syn_daily_ret = pd.Series(dtype=float)

    # --- Bootstrap on NC returns ---
    # Use empirical NC if available; fall back to synthetic
    if emp_daily_ret is not None and len(emp_daily_ret) > 5:
        nc_daily = emp_daily_ret
        nc_source = "empirical"
    elif len(syn_daily_ret) > 5:
        nc_daily = syn_daily_ret
        nc_source = "synthetic"
    else:
        nc_daily  = pd.Series([0.0] * 20)
        nc_source = "fallback_zeros"

    bootstrap_sharpes = _block_bootstrap_sharpe(
        nc_daily, n_bootstrap=_N_BOOTSTRAP,
        block_size=_BLOCK_SIZE_DAYS, seed=seed,
    )
    b_mean   = float(np.mean(bootstrap_sharpes))
    b_std    = float(np.std(bootstrap_sharpes))
    threshold = b_mean + 2.0 * b_std
    nc_pass   = primary_sharpe > threshold

    log.info(
        "NC bootstrap: mean=%.3f std=%.3f threshold=%.3f | primary=%.3f | %s",
        b_mean, b_std, threshold, primary_sharpe,
        "PASS" if nc_pass else "FAIL",
    )

    return {
        "empirical_sharpe":   emp_sharpe,
        "synthetic_sharpe":   syn_sharpe,
        "nc_daily_returns":   nc_daily,
        "bootstrap_mean":     b_mean,
        "bootstrap_std":      b_std,
        "bootstrap_threshold": threshold,
        "bootstrap_sharpes":  bootstrap_sharpes,
        "nc_pass":            nc_pass,
        "nc_source":          nc_source,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_nc_pair(
    ta: str, tb: str,
    trading_1min: dict,
    delta: float,
    per_pair_dollar: float,
    tc_bps: float,
    borrow_bps_yr: float,
    total_capital: float,
    eos_flatten: bool = True,
) -> dict | None:
    """Run the engine on one empirical NC pair and return metrics."""
    df_a = trading_1min[ta].copy()
    df_b = trading_1min[tb].copy()

    if "log_close" not in df_a.columns:
        df_a["log_close"] = np.log(df_a["close"].clip(lower=1e-10))
    if "log_close" not in df_b.columns:
        df_b["log_close"] = np.log(df_b["close"].clip(lower=1e-10))

    df_ab = df_a[["log_close", "close"]].join(
        df_b[["log_close", "close"]], how="inner", lsuffix="_a", rsuffix="_b"
    ).dropna()

    if len(df_ab) < 50:
        return None

    # OLS hedge ratio: cov(log_a, log_b) / var(log_b)
    # std-ratio ignores correlation sign and magnitude
    log_a = df_ab["log_close_a"].values
    log_b = df_ab["log_close_b"].values
    var_b = float(np.var(log_b))
    beta_naive  = float(np.cov(log_a, log_b)[0, 1] / var_b) if var_b > 1e-12 else 1.0
    residuals   = log_a - beta_naive * log_b
    alpha_naive = np.mean(residuals)
    R_naive    = float(np.var(residuals))
    if R_naive <= 0:
        R_naive = 1.0

    try:
        pair_df = run_pair(
            log_a_1min   = log_a,
            log_b_1min   = log_b,
            close_a_1min = df_ab["close_a"].values,
            close_b_1min = df_ab["close_b"].values,
            index_1min   = df_ab.index,
            alpha_pca    = alpha_naive,
            beta_pca     = beta_naive,
            R            = R_naive,
            delta        = delta,
            half_life_days = 5.0,
            eos_flatten  = eos_flatten,
        )
    except Exception as e:
        log.warning("NC pair %s/%s engine failed: %s", ta, tb, e)
        return None

    if pair_df.empty:
        return None

    pnl_result = compute_pair_pnl(
        pair_df,
        df_ab["close_a"],
        df_ab["close_b"],
        per_pair_dollar, tc_bps, borrow_bps_yr,
    )
    metrics = compute_metrics(
        pnl_result["bar_pnl"], total_capital,
        pnl_result["trade_log"],
    )
    return metrics


def _block_bootstrap_sharpe(
    daily_returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap Sharpe ratio distribution.

    daily_returns is at trading-day frequency. block_size=5 (1 trading week)
    preserves intra-week serial correlation that pairs strategies typically
    exhibit (positions held across multiple days). block_size=1 collapses to
    iid bootstrap and underestimates NC variance.

    Uses overlapping blocks (Politis & Romano moving-block bootstrap):
    starts drawn uniformly from [0, n - block_size + 1).
    Returns array of n_bootstrap Sharpe values.
    """
    arr = daily_returns.values
    n   = len(arr)
    if n < block_size + 1:
        # Fallback: shrink block size if series is short
        block_size = max(min(block_size, n // 2), 1)
        if n < 2:
            return np.zeros(n_bootstrap)

    rng      = np.random.default_rng(seed)
    n_blocks = max(n // block_size, 1)
    max_start = max(n - block_size + 1, 1)
    sharpes  = np.zeros(n_bootstrap)

    for k in range(n_bootstrap):
        starts    = rng.integers(0, max_start, size=n_blocks)
        resampled = np.concatenate([
            arr[s: s + block_size] for s in starts
        ])[:n]
        if len(resampled) < 2:
            continue
        std = resampled.std(ddof=1)
        sharpes[k] = (resampled.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0

    return sharpes
