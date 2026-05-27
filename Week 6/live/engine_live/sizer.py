"""Vol-target sizing for live paper trades — scaled 10x down from backtest.

REUSES engine_daily.engine_daily.compute_vol_target_notional_daily (same math
backtest uses). Paper-scaled defaults so total gross stays within Alpaca
Reg T 2x margin:

  Backtest (1M capital, 178 pairs avg):        target=$200/day, [$10k, $40k]
  Paper    (100k capital, ~13 pairs max):      target=$20/day,  [$1k,  $4k]

At 13 simultaneous pairs * $8k gross (= 2 legs * $4k cap): ~$104k gross
which is comfortably within $200k buying power (Reg T 2x).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Delegate target — SAME function as backtest.
from engine_daily.engine_daily import compute_vol_target_notional_daily


# Paper-scaled defaults. Override via env vars for stress testing.
_PAPER_TARGET_DAILY = float(os.environ.get("PAPER_VOL_TARGET_DAILY", "20.0"))
_PAPER_FLOOR = float(os.environ.get("PAPER_NOTIONAL_FLOOR", "1000.0"))
_PAPER_CAP = float(os.environ.get("PAPER_NOTIONAL_CAP", "4000.0"))


@dataclass(frozen=True)
class PairNotional:
    pair_id: str
    ticker_a: str
    ticker_b: str
    notional: float       # per-pair notional (one-leg dollar amount)
    spread_sigma_daily: float


def compute_pair_notional(spread_formation: np.ndarray | pd.Series,
                          target_daily_vol: float = _PAPER_TARGET_DAILY,
                          floor: float = _PAPER_FLOOR,
                          cap: float = _PAPER_CAP) -> float:
    """Per-pair notional for paper trading. Same math as backtest, scaled-down defaults."""
    if isinstance(spread_formation, pd.Series):
        arr = spread_formation.values
    else:
        arr = np.asarray(spread_formation)
    return compute_vol_target_notional_daily(
        arr, target_daily_vol=target_daily_vol, floor=floor, cap=cap,
    )


def size_all_pairs(pairs_df: pd.DataFrame,
                   formation_residuals: pd.DataFrame,
                   alphas: dict[tuple[str, str], float],
                   target_daily_vol: float = _PAPER_TARGET_DAILY,
                   floor: float = _PAPER_FLOOR,
                   cap: float = _PAPER_CAP) -> list[PairNotional]:
    """For every pair, compute its vol-target notional from formation spread series.

    Parameters
    ----------
    pairs_df            : ticker_a, ticker_b, beta_pca (alpha NOT used — comes from `alphas` map)
    formation_residuals : DataFrame[date, ticker] — residual log-prices over formation
    alphas              : dict (ticker_a, ticker_b) -> alpha_refit (from TODO 6)
    target_daily_vol    : $/day P&L vol target per pair
    floor, cap          : notional clipping bounds
    """
    out: list[PairNotional] = []
    for _, row in pairs_df.iterrows():
        ta = str(row["ticker_a"])
        tb = str(row["ticker_b"])
        beta = float(row["beta_pca"])
        alpha = alphas.get((ta, tb))
        if alpha is None:
            continue
        if ta not in formation_residuals.columns or tb not in formation_residuals.columns:
            continue
        ra = formation_residuals[ta].dropna()
        rb = formation_residuals[tb].dropna()
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        if len(aligned) < 10:
            continue
        spread = aligned.iloc[:, 0].values - alpha - beta * aligned.iloc[:, 1].values
        sigma = float(np.std(np.diff(spread), ddof=1)) if len(spread) > 1 else 0.0
        notional = compute_pair_notional(
            spread, target_daily_vol=target_daily_vol, floor=floor, cap=cap,
        )
        out.append(PairNotional(
            pair_id=f"{ta}_{tb}", ticker_a=ta, ticker_b=tb,
            notional=notional, spread_sigma_daily=sigma,
        ))
    return out
