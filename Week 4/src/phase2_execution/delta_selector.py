"""
Phase 2 — Multi-Criterion Delta Auto-Selector

Run once per fold on formation-window data for all surviving pairs.
Selects optimal Kalman delta from grid {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}.

Selection rule (pre-registered):
  optimal_delta = argmin{ median(|kurtosis - 3|) across pairs }
  subject to:
    median_HL    in [1, 10] trading days
    median_ACF78 > 0.7           (ACF at lag=78 = 1 trading day on 5-min data)

Edge case flags (logged, not fatal unless all delta fail):
  - delta at grid boundary -> "expand grid next fold"
  - no delta passes all 3 constraints -> "universal constraint fail, kick fold"
  - delta jumps >2 grid steps vs prev_delta -> "parameter instability"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from src.phase2_execution.kalman import run_kalman
from src.utils.stats import compute_ou_halflife

log = logging.getLogger(__name__)

DELTA_GRID = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
_HL_MIN = 1.0
_HL_MAX = 10.0
_ACF78_MIN = 0.7
_BARS_PER_DAY_5MIN = 78
# Discard first KALMAN_BURNIN bars from metrics to let filter converge from P_0=R*I.
# Tight pairs (small R) have high initial Kalman gain that normalises within ~500 bars.
_KALMAN_BURNIN = 500
# Cap number of pairs used for delta selection. Median statistics over 200 pairs
# are near-identical to medians over 6,000 pairs. Without this cap, fold 23 (6,008 pairs)
# would run 30,040 Kalman evaluations on formation data — ~150s vs ~1s with cap.
_MAX_DELTA_SAMPLE_PAIRS = 200


def select_delta(
    pairs_df: pd.DataFrame,
    formation_5min: dict[str, pd.DataFrame],
    prev_delta: float | None = None,
) -> tuple[float | None, dict]:
    """
    Select optimal Kalman delta for this fold.

    Parameters
    ----------
    pairs_df        : Phase 1 surviving pairs with columns
                      [ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise, half_life_days]
    formation_5min  : dict ticker -> DataFrame with 'log_close' column (formation window)
    prev_delta      : delta selected in prior fold (for instability check), or None

    Returns
    -------
    optimal_delta   : selected delta, or None if universal constraint fail
    metrics_by_delta: dict {delta: {metric_kurt, median_HL, median_ACF78}}
    """
    metrics_by_delta: dict = {}

    # Sample top pairs by Johansen p-value for efficient delta selection.
    # Median statistics are stable over 200 pairs; no need to evaluate all 6,000.
    sample_df = pairs_df.nsmallest(_MAX_DELTA_SAMPLE_PAIRS, "johansen_pval")
    if len(sample_df) < len(pairs_df):
        log.info("select_delta: sampling %d / %d pairs for grid search", len(sample_df), len(pairs_df))

    for delta in DELTA_GRID:
        kurt_list, hl_list, acf78_list = [], [], []

        for _, row in sample_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            if ta not in formation_5min or tb not in formation_5min:
                continue

            s_a = formation_5min[ta]["log_close"]
            s_b = formation_5min[tb]["log_close"]
            sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
            valid = ~(sa_aligned.isna() | sb_aligned.isna())
            log_a = sa_aligned[valid].values
            log_b = sb_aligned[valid].values
            if len(log_a) < 10:
                continue

            R = float(row["R_measurement_noise"])
            if R <= 0 or np.isnan(R):
                continue

            try:
                spread_prior, *_ = run_kalman(
                    log_a, log_b,
                    float(row["alpha_pca"]), float(row["beta_pca"]),
                    R, delta,
                )
            except Exception:
                continue

            # Discard burn-in so metrics reflect converged filter state
            spread_raw = spread_prior[_KALMAN_BURNIN:]
            spread = spread_raw[~np.isnan(spread_raw)]
            if len(spread) < 20:
                continue

            # Criterion 1: kurtosis closeness to 3 (normal)
            kurt_total = float(scipy_kurtosis(spread, fisher=False))  # total kurtosis
            kurt_list.append(abs(kurt_total - 3.0))

            # Criterion 2: OU half-life on prior spread (5-min bars)
            hl = compute_ou_halflife(spread, bars_per_day=_BARS_PER_DAY_5MIN)
            if not np.isnan(hl):
                hl_list.append(hl)

            # Criterion 3: ACF at lag=78 (= 1 trading day on 5-min data)
            s = pd.Series(spread)
            if len(s) > 78:
                acf78 = abs(float(s.autocorr(lag=78)))
                if not np.isnan(acf78):
                    acf78_list.append(acf78)

        metrics_by_delta[delta] = {
            "metric_kurt":   float(np.median(kurt_list))  if kurt_list  else np.nan,
            "median_HL":     float(np.median(hl_list))    if hl_list    else np.nan,
            "median_ACF78":  float(np.median(acf78_list)) if acf78_list else np.nan,
        }

    # Multi-criterion feasibility filter
    feasible = {
        d: m for d, m in metrics_by_delta.items()
        if (
            not np.isnan(m["median_HL"])
            and _HL_MIN <= m["median_HL"] <= _HL_MAX
            and not np.isnan(m["median_ACF78"])
            and m["median_ACF78"] > _ACF78_MIN
        )
    }

    # Log all metrics for audit trail
    for d, m in metrics_by_delta.items():
        log.info(
            "  delta=%.0e | kurt_dev=%.3f | median_HL=%.2f | ACF78=%.3f%s",
            d, m["metric_kurt"], m["median_HL"], m["median_ACF78"],
            " [FEASIBLE]" if d in feasible else "",
        )

    if not feasible:
        log.warning("UNIVERSAL CONSTRAINT FAIL — no delta passes HL+ACF78 constraints. Kick fold.")
        return None, metrics_by_delta

    optimal_delta = min(feasible, key=lambda d: feasible[d]["metric_kurt"])

    # Edge case: delta at grid boundary
    if optimal_delta == DELTA_GRID[0]:
        log.warning("DELTA AT LOWER BOUNDARY (%.0e) — kurtosis criterion degenerate, applying HL-nearest fallback", optimal_delta)
        # Fallback: pick feasible delta with median_HL closest to 5d (midpoint of [1,10])
        hl_fallback = min(feasible, key=lambda d: abs(feasible[d]["median_HL"] - 5.0))
        if hl_fallback != optimal_delta:
            log.info(
                "HL-nearest fallback: delta=%.0e (median_HL=%.2f, target=5.0d)",
                hl_fallback, feasible[hl_fallback]["median_HL"],
            )
            optimal_delta = hl_fallback
    elif optimal_delta == DELTA_GRID[-1]:
        log.warning("DELTA AT UPPER BOUNDARY (%.0e) — consider expanding grid larger", optimal_delta)

    # Edge case: instability vs prior fold
    if prev_delta is not None:
        prev_idx = DELTA_GRID.index(prev_delta) if prev_delta in DELTA_GRID else -1
        curr_idx = DELTA_GRID.index(optimal_delta)
        if prev_idx >= 0 and abs(curr_idx - prev_idx) > 2:
            log.warning(
                "DELTA INSTABILITY — jumped from %.0e (idx %d) to %.0e (idx %d)",
                prev_delta, prev_idx, optimal_delta, curr_idx,
            )

    m = feasible[optimal_delta]
    log.info(
        "Selected delta=%.0e | kurt_dev=%.3f | median_HL=%.2f | ACF78=%.3f",
        optimal_delta, m["metric_kurt"], m["median_HL"], m["median_ACF78"],
    )
    return optimal_delta, metrics_by_delta
