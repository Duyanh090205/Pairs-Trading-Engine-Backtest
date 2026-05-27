"""Refit alpha at trading-window start.

Thin wrapper that REUSES engine_daily.alpha_refit.recompute_alpha (the same
function backtest run_v4_pipeline calls per pair). For each pair in the live
pair list, look up β (fixed from discovery) + formation residuals (from
factor_state.residual_log_prices), then call recompute_alpha to get fresh α.

Per [[week6-live-invariants]]: β is FIXED from discovery, only α is refit on
last 60 bars of formation. NO Johansen rerun.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Delegate target — SAME function as backtest.
from engine_daily.alpha_refit import recompute_alpha


@dataclass(frozen=True)
class RefitAlpha:
    pair_id: str          # "TICKERA_TICKERB"
    ticker_a: str
    ticker_b: str
    alpha_refit: float
    beta: float           # frozen from discovery
    n_bars_used: int      # actual bars in α refit (<= n_lookback)


def refit_all_pairs(pairs_df: pd.DataFrame,
                    formation_residuals: pd.DataFrame,
                    n_lookback: int = 60) -> list[RefitAlpha]:
    """For every pair in pairs_df, refit alpha against the latest formation residuals.

    Parameters
    ----------
    pairs_df            : DataFrame from discovery — must have ticker_a, ticker_b, beta_pca.
    formation_residuals : DataFrame from factor_state — columns = tickers,
                          rows = formation dates, values = residual log-prices.
    n_lookback          : Last N bars to use (V4 default 60).
    """
    out: list[RefitAlpha] = []
    for _, row in pairs_df.iterrows():
        ta = str(row["ticker_a"])
        tb = str(row["ticker_b"])
        beta = float(row["beta_pca"])
        if ta not in formation_residuals.columns or tb not in formation_residuals.columns:
            continue
        ra = formation_residuals[ta].dropna()
        rb = formation_residuals[tb].dropna()
        try:
            alpha = recompute_alpha(ra, rb, beta, n_lookback=n_lookback)
        except ValueError:
            continue
        # Determine actual bars used (matches recompute_alpha's internal logic)
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        n_used = min(n_lookback, len(aligned))
        out.append(RefitAlpha(
            pair_id=f"{ta}_{tb}",
            ticker_a=ta, ticker_b=tb,
            alpha_refit=alpha, beta=beta, n_bars_used=n_used,
        ))
    return out


def load_and_refit(pairs_path: Path | str, factor_state_path: Path | str,
                   n_lookback: int = 60) -> list[RefitAlpha]:
    """Convenience: load discovery artifacts + run refit. For offline driver."""
    import pickle
    pairs_df = pd.read_parquet(pairs_path)
    with Path(factor_state_path).open("rb") as f:
        fs = pickle.load(f)
    return refit_all_pairs(pairs_df, fs["residual_log_prices"], n_lookback=n_lookback)
