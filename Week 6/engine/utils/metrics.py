"""
Performance metrics: Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate.
Adapted from Week 3/scripts/engine.py.

All metrics follow pipeline spec §3.3:
  - Sharpe: daily returns × √252
  - MaxDD:  bar-level equity curve (NOT daily MTM)
  - Calmar: CAGR / bar-level MaxDD
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def compute_sharpe(daily_returns: pd.Series | np.ndarray) -> float:
    """Annualized Sharpe on daily returns. Returns nan if std=0."""
    rets = _clean(daily_returns)
    if len(rets) < 2:
        return np.nan
    std = rets.std(ddof=1)
    if std < 1e-10:
        return np.nan
    return float((rets.mean() / std) * np.sqrt(252))


def compute_max_dd(bar_equity: pd.Series | np.ndarray) -> float:
    """
    Maximum drawdown on bar-level equity curve (most negative peak-to-trough ratio).
    Returns a negative number (e.g. -0.15 = 15% drawdown).
    """
    eq = _clean(bar_equity)
    if len(eq) == 0:
        return np.nan
    running_max = np.maximum.accumulate(eq)
    return float((eq / running_max - 1.0).min())


def compute_cagr(bar_equity: pd.Series | np.ndarray, n_trading_days: int) -> float:
    """
    Compound Annual Growth Rate.
    bar_equity: compound equity curve starting from 1.0.
    n_trading_days: number of trading days in the period.
    """
    eq = _clean(bar_equity)
    if len(eq) == 0 or n_trading_days <= 0:
        return np.nan
    final = eq[-1]
    if final <= 0:
        return -1.0
    return float(final ** (252.0 / n_trading_days) - 1.0)


def compute_calmar(cagr: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / |max_dd|. Returns nan if max_dd = 0."""
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return float(cagr / abs(max_dd))


def compute_win_rate(trade_returns: pd.Series | np.ndarray) -> float:
    """Fraction of trades with positive net return."""
    rets = _clean(trade_returns)
    if len(rets) == 0:
        return np.nan
    return float((rets > 0).mean())


def compute_all(
    daily_returns: pd.Series,
    bar_equity: pd.Series,
    n_trading_days: int,
    trade_returns: pd.Series | None = None,
) -> dict:
    """Compute the full metric suite and return as a dict."""
    sharpe = compute_sharpe(daily_returns)
    max_dd = compute_max_dd(bar_equity)
    cagr = compute_cagr(bar_equity, n_trading_days)
    calmar = compute_calmar(cagr, max_dd)
    win_rate = compute_win_rate(trade_returns) if trade_returns is not None else np.nan

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cagr": cagr,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_trading_days": n_trading_days,
    }
