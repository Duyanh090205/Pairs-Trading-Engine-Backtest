"""
Portfolio-level helpers for V4 daily engine.

Currently contains the ticker-concentration cap (one ticker can appear in at
most `max_pairs_per_ticker` pairs, ranked by Johansen p-value ascending).
Previously lived in engine/phase2_execution/engine.py (V3 module); inlined
here so V4 has no transitive import on V3-only modules.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_MAX_PAIRS_PER_TICKER = 5


def apply_ticker_concentration_cap(
    pairs_df: pd.DataFrame,
    max_pairs_per_ticker: int = _MAX_PAIRS_PER_TICKER,
) -> pd.DataFrame:
    """
    Keep at most max_pairs_per_ticker pairs per ticker (ranked by johansen_pval asc).
    Addresses folds with 5,000+ pairs where a single ticker appears in 100+ pairs.
    """
    if pairs_df.empty:
        return pairs_df

    ranked = pairs_df.sort_values("johansen_pval").reset_index(drop=True)
    ticker_count: dict[str, int] = {}
    keep_idx: list[int] = []

    for idx, row in ranked.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        ca = ticker_count.get(ta, 0)
        cb = ticker_count.get(tb, 0)
        if ca < max_pairs_per_ticker and cb < max_pairs_per_ticker:
            keep_idx.append(idx)
            ticker_count[ta] = ca + 1
            ticker_count[tb] = cb + 1

    result = ranked.loc[keep_idx].reset_index(drop=True)
    if len(result) < len(ranked):
        log.info(
            "Concentration cap: %d -> %d pairs (max %d per ticker)",
            len(ranked), len(result), max_pairs_per_ticker,
        )
    return result
