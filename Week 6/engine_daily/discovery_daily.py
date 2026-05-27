"""
Phase 1 — Cointegration Discovery (Daily / V4)
================================================
Daily-frequency port of `engine.phase1_cointegration.discovery`.

V4 changes vs V3.0 discovery:
    - Bar frequency: daily (1 bar/day) instead of 5-min (78 bars/day)
    - HL filter: [5, 30] trading days (AL2010 mean-reversion regime) instead of
                 [1, 6] days. Daily noise dominates HL < 5d; AL2010 finds usable
                 mean-reversion regimes in [5, 30].
    - `bars_per_day` arg to `compute_ou_halflife` is 1 (daily input)
    - Min obs reduced: 100 daily bars (~5 months) instead of 200 5-min bars

Reused unchanged from V3:
    - PCA factor model (`fit_factor_model`) — frequency-agnostic
    - PCA hedge ratio + Johansen p-value (`_pca_hedge_ratio`, `_johansen_pvalue`)
    - Parallel pairwise Johansen (`_johansen_test_chunk`, `_run_pairwise_tests`)
    - Hard screens (`_apply_hard_screens`) — same liquidity/price/completeness
      thresholds (volume in daily input is sum-per-day, same units as before)
    - BH-FDR correction
    - beta>0 filter
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.phase1_cointegration.discovery import (
    _apply_hard_screens,
    _run_pairwise_tests,
    apply_beta_filter,
)
from engine.phase1_cointegration.factor_residual import fit_factor_model
from engine.utils.stats import bh_fdr_correct, compute_ou_halflife

log = logging.getLogger(__name__)

# --------- Daily-specific constants ---------
_HL_MIN_DAYS = 5.0           # AL2010-typical lower bound (faster = noise)
_HL_MAX_DAYS = 30.0          # AL2010-typical upper bound (slower = trend, not MR).
                             # Now overridable via discovery_daily.run(hl_max=...)
                             # for Day-4 grid sweeps.
_BARS_PER_DAY = 1            # daily input
_BH_FDR_Q = 0.05
_N_FACTOR_COMPONENTS = 5
_MIN_OBS_DAILY = 100         # ~5 months of daily bars
_MAX_PAIRS_POST_FILTER = 500
_BETA_ABS_CAP = 5.0          # Reject pairs with |beta_pca| > cap.
                             # Practitioner sanity ceiling for factor-residual stat-arb;
                             # |beta| > 5 on PCA residuals is almost always a Johansen
                             # numerical artifact (see audits/audit_extreme_beta_v4.py:
                             # 27 pairs with |beta| in [50, 2708] contributed 103% of
                             # total P&L in the uncapped run -> not real alpha).


def _apply_fdr_and_halflife_daily(
    results: list[dict],
    formation_data: dict[str, pd.DataFrame],
    hl_min: float = _HL_MIN_DAYS,
    hl_max: float = _HL_MAX_DAYS,
) -> pd.DataFrame:
    """
    Apply BH-FDR (q=0.05) then OU half-life filter [hl_min, hl_max] trading days.

    Daily-frequency variant: passes `bars_per_day=1` to compute_ou_halflife
    because input series are already daily. Otherwise identical structure to
    the V3 5-min version.
    """
    if not results:
        return pd.DataFrame()

    pvals = np.array([r["johansen_pval"] for r in results])
    reject = bh_fdr_correct(pvals, q=_BH_FDR_Q)
    log.info("BH-FDR: %d / %d pairs survive (q=%.2f)",
             reject.sum(), len(reject), _BH_FDR_Q)

    surviving: list[dict] = []
    n_hl_fail = 0

    for row, keep in zip(results, reject):
        if not keep:
            continue

        s_a = formation_data[row["ticker_a"]]["log_close"]
        s_b = formation_data[row["ticker_b"]]["log_close"]
        sa, sb = s_a.align(s_b, join="inner")
        valid = ~(sa.isna() | sb.isna())
        log_a = sa[valid].values
        log_b = sb[valid].values

        spread = log_a - row["alpha_pca"] - row["beta_pca"] * log_b
        hl_days = compute_ou_halflife(spread, bars_per_day=_BARS_PER_DAY)

        if np.isnan(hl_days) or not (hl_min <= hl_days <= hl_max):
            n_hl_fail += 1
            continue

        surviving.append({**row, "half_life_days": hl_days})

    n_before_cap = len(surviving)
    if len(surviving) > _MAX_PAIRS_POST_FILTER:
        surviving.sort(key=lambda r: r["johansen_pval"])
        surviving = surviving[:_MAX_PAIRS_POST_FILTER]
        log.warning(
            "PAIR-COUNT GATE: %d -> %d (top by p-value)",
            n_before_cap, _MAX_PAIRS_POST_FILTER,
        )

    log.info(
        "HL filter daily: %d passed (%d dropped outside [%.0f, %.0f]d) -> %d after cap",
        n_before_cap, n_hl_fail, hl_min, hl_max, len(surviving),
    )
    return pd.DataFrame(surviving)


def run(
    formation_data: dict[str, pd.DataFrame],
    max_pairs: int | None = 500,
    hl_min: float = _HL_MIN_DAYS,
    hl_max: float = _HL_MAX_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """
    Phase 1 V4 daily discovery.

    Parameters
    ----------
    formation_data : dict[ticker] -> daily DataFrame with `log_close`, `volume`.
                     Index = tz-naive Timestamp at midnight. From load_daily_universe.
    max_pairs      : fallback cap (top-K by Johansen p-value before BH-FDR).

    Returns
    -------
    pairs_df : columns ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise,
               johansen_pval, n_overlapping_bars, overlap_ratio, half_life_days
    factor_state : { loadings_W, tickers, diagnostics, residual_log_prices }
    """
    t0 = time.time()
    log.info("Phase 1 V4 daily | %d formation tickers", len(formation_data))
    print(f"  [v4-discovery] {len(formation_data)} formation tickers", flush=True)

    if not formation_data:
        return pd.DataFrame(), {}

    # ---- Hard screens (same logic, daily input) ----
    survivors, _, max_formation_bars = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        log.warning("Fewer than 2 survivors after hard screens.")
        return pd.DataFrame(), {}
    print(f"  [v4-discovery] hard screens: {len(survivors)} survivors", flush=True)

    formation_data = {t: formation_data[t] for t in survivors}

    # ---- PCA factor model on daily returns ----
    print(f"  [v4-discovery] fitting factor model (PCA on daily returns)...", flush=True)
    try:
        loadings_W, factor_tickers, residual_log_prices, fm_diag = fit_factor_model(
            formation_data, n_components=_N_FACTOR_COMPONENTS,
        )
    except ValueError as e:
        log.warning("factor model fit failed: %s -- empty fold", e)
        return pd.DataFrame(), {}

    # Replace each ticker's log_close with its residual log-price
    formation_data_residual: dict[str, pd.DataFrame] = {}
    for tk in factor_tickers:
        df = formation_data[tk].copy()
        df = df.loc[df.index.intersection(residual_log_prices.index)]
        df["log_close"] = residual_log_prices[tk].reindex(df.index)
        df = df.dropna(subset=["log_close"])
        if len(df) >= _MIN_OBS_DAILY:
            formation_data_residual[tk] = df

    survivors_residual = list(formation_data_residual.keys())
    if len(survivors_residual) < 2:
        log.warning("Fewer than 2 survivors after factor projection.")
        return pd.DataFrame(), {}

    max_formation_bars = max(len(df) for df in formation_data_residual.values())
    log.info(
        "factor model daily: tickers=%d, cum_var=%.3f, max_bars=%d",
        len(factor_tickers), fm_diag["cumulative_variance_explained"],
        max_formation_bars,
    )

    # ---- Pairwise Johansen (parallel, reused as-is) ----
    n_pairs_to_test = len(survivors_residual)*(len(survivors_residual)-1)//2
    print(f"  [v4-discovery] running pairwise Johansen on {len(survivors_residual)} "
          f"tickers ({n_pairs_to_test} pairs)...", flush=True)
    raw_results = _run_pairwise_tests(
        survivors_residual, formation_data_residual, max_formation_bars,
    )
    if not raw_results:
        log.warning("No pairs passed overlap filter.")
        return pd.DataFrame(), {}
    print(f"  [v4-discovery] pairwise tests done: {len(raw_results)} candidates",
          flush=True)

    # Fallback cap by Johansen p-value
    if max_pairs is not None and len(raw_results) > max_pairs:
        raw_results.sort(key=lambda r: r["johansen_pval"])
        raw_results = raw_results[:max_pairs]
        log.warning("FALLBACK CAP: kept top %d by p-value", max_pairs)

    # ---- BH-FDR + daily HL filter ----
    pairs_df = _apply_fdr_and_halflife_daily(
        raw_results, formation_data_residual, hl_min=hl_min, hl_max=hl_max,
    )

    # ---- beta>0 filter ----
    pairs_df = apply_beta_filter(pairs_df)

    # ---- |beta| <= 5 cap (numerical sanity, see _BETA_ABS_CAP docstring) ----
    if not pairs_df.empty:
        n_before_cap = len(pairs_df)
        pairs_df = pairs_df[pairs_df["beta_pca"].abs() <= _BETA_ABS_CAP].copy()
        n_dropped = n_before_cap - len(pairs_df)
        if n_dropped > 0:
            log.info("BETA-ABS CAP |beta|<=%.1f: dropped %d / %d pairs",
                     _BETA_ABS_CAP, n_dropped, n_before_cap)
            print(f"  [v4-discovery] |beta|<={_BETA_ABS_CAP:.0f} cap: "
                  f"{n_before_cap} -> {len(pairs_df)} ({n_dropped} dropped)",
                  flush=True)

    factor_state = {
        "loadings_W": loadings_W,
        "tickers": factor_tickers,
        "diagnostics": fm_diag,
        "residual_log_prices": residual_log_prices,
    }

    log.info(
        "Phase 1 V4 daily complete: %d pairs after beta>0 | %.0fs",
        len(pairs_df), time.time() - t0,
    )
    return pairs_df, factor_state
