"""
Phase 1 — Cointegration Discovery
===================================
Called once per fold by the Phase 4 orchestrator.

Pipeline spec §1.1–1.7:
    formation data (5-min log-prices)
    → universe hard screens
    → all-pairs enumeration
    → pairwise inner join (≥80% min-overlap)
    → PCA hedge ratio (secondary eigenvector, Avellaneda-Lee)
    → Johansen trace test (p-value via chi2(8) approximation — see note below)
    → BH-FDR correction (q=0.05)
    → OU half-life filter [1, 10] trading days
    → surviving pairs DataFrame

Johansen p-value approximation:
    statsmodels coint_johansen returns critical values only, not p-values.
    For a 2-variable system at rank-0 trace test, the trace statistic is
    well-approximated by chi2(df=8):
        90% CV: chi2=13.36  vs Johansen=13.43  (error <0.5%)
        95% CV: chi2=15.51  vs Johansen=15.49  (error <0.2%)
        99% CV: chi2=20.09  vs Johansen=19.93  (error <0.8%)
    p_value = 1 - chi2.cdf(trace_stat_rank0, df=8)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.utils.io import VALIDATED_DIR, list_validated_tickers, read_5min
from src.utils.stats import bh_fdr_correct, compute_ou_halflife

log = logging.getLogger(__name__)

_MIN_MEDIAN_PRICE = 5.0
_MIN_ADV_DOLLAR = 1_000_000
_MIN_COMPLETENESS = 0.90
_MAX_ZERO_RETURN = 0.50
_MIN_OVERLAP = 0.80
_BH_FDR_Q = 0.05
_HL_MIN_DAYS = 1.0
_HL_MAX_DAYS = 10.0
_BARS_PER_DAY_5MIN = 78   # pipeline spec §1.6 convention (actual session = 77 bars)


def _load_formation_data(
    formation_start: str,
    formation_end: str,
    validated_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load 5-min log-prices for all validated tickers sliced to the formation window."""
    tickers = list_validated_tickers("5min", validated_dir)
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = read_5min(ticker, validated_dir).loc[formation_start:formation_end]
            if len(df) > 0:
                data[ticker] = df
        except FileNotFoundError:
            log.debug("Skip %s: parquet not found", ticker)
        except Exception as exc:
            log.warning("Unexpected error loading %s: %s", ticker, exc)

    log.info("Loaded %d tickers for formation %s to %s", len(data), formation_start, formation_end)
    return data


def _apply_hard_screens(
    formation_data: dict[str, pd.DataFrame],
) -> tuple[list[str], dict, int]:
    """
    Apply §1.1 screens on formation window data per ticker.
    Returns (survivor_tickers, screen_log, max_formation_bars).

    max_formation_bars: max bars observed across ALL loaded tickers (before screening),
    used as the expected-bars reference for both completeness (§1.1) and the pairwise
    overlap ratio (§1.2.1). Computing over pre-screen tickers ensures the reference
    equals the formation window's true capacity, not just the survivors.
    """
    if not formation_data:
        return [], {}, 0

    max_bars = max(len(df) for df in formation_data.values())
    survivors: list[str] = []
    screen_log: dict[str, str] = {}
    counts = Counter()

    for ticker, df in formation_data.items():
        close = np.exp(df["log_close"])
        vol = df["volume"]

        if close.median() < _MIN_MEDIAN_PRICE:
            screen_log[ticker] = "median_price"
            counts["median_price"] += 1
            continue

        daily_dollar_vol = (close * vol).groupby(df.index.normalize()).sum()
        if daily_dollar_vol.mean() < _MIN_ADV_DOLLAR:
            screen_log[ticker] = "adv_dollar"
            counts["adv_dollar"] += 1
            continue

        if len(df) / max_bars < _MIN_COMPLETENESS:
            screen_log[ticker] = "completeness"
            counts["completeness"] += 1
            continue

        log_ret = df["log_close"].diff()
        if (log_ret == 0).sum() / max(len(log_ret.dropna()), 1) >= _MAX_ZERO_RETURN:
            screen_log[ticker] = "zero_return"
            counts["zero_return"] += 1
            continue

        survivors.append(ticker)
        counts["passed"] += 1

    log.info(
        "Screens: %d passed | price=%d adv=%d completeness=%d zero_ret=%d",
        counts["passed"], counts["median_price"],
        counts["adv_dollar"], counts["completeness"], counts["zero_return"],
    )
    return survivors, {"binding_counts": dict(counts), "rejected": screen_log}, max_bars


def _pca_hedge_ratio(
    log_a: np.ndarray,
    log_b: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute PCA hedge ratio using secondary eigenvector (Avellaneda-Lee convention).

    X = [ln(A) - mean(ln(A)),  ln(B) - mean(ln(B))]  (T×2, centered)
    Secondary eigenvector v_2 (smallest eigenvalue of Cov) = cointegrating direction.

    β_PCA = -v_2[0] / v_2[1]
    α_PCA = mean(ln(A)) - β_PCA × mean(ln(B))
    R     = var(spread)   ← Kalman measurement noise init

    Returns (alpha_PCA, beta_PCA, R). All nan if degenerate.
    """
    mu_a, mu_b = log_a.mean(), log_b.mean()
    X = np.column_stack([log_a - mu_a, log_b - mu_b])
    cov = X.T @ X / (len(X) - 1)

    # np.linalg.eigh returns eigenvalues ascending (Hermitian-optimised).
    # For 2×2: ascending[0] = smallest = secondary eigenvector = cointegrating direction.
    # This is identical to spec pseudocode's descending[:, 1] — different sort, same vector.
    _, eigvecs = np.linalg.eigh(cov)
    v2 = eigvecs[:, 0]

    if abs(v2[1]) < 1e-12:
        return np.nan, np.nan, np.nan

    beta = -v2[0] / v2[1]
    alpha = mu_a - beta * mu_b
    R = float(np.var(log_a - alpha - beta * log_b, ddof=1))
    return float(alpha), float(beta), R


def _johansen_pvalue(X: np.ndarray) -> float:
    """
    Johansen trace test on a pre-stacked (T, 2) array. Returns chi2(8) p-value.
    H0: rank = 0 (no cointegration). Returns nan on failure.
    """
    try:
        res = coint_johansen(X, det_order=0, k_ar_diff=1)
    except Exception:
        return np.nan
    return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))


def _run_pairwise_tests(
    survivors: list[str],
    formation_data: dict[str, pd.DataFrame],
    max_formation_bars: int,
) -> list[dict]:
    """
    Enumerate all pairs, apply overlap filter, PCA, and Johansen.
    Stores only scalars per result — no array caching — so N workers
    each use ~100 MB instead of ~7 GB.

    max_formation_bars: expected total bars in the formation window (spec §1.2.1
    denominator). Must come from _apply_hard_screens so it reflects the full
    window capacity, not just the two tickers being tested.
    """
    all_pairs = list(combinations(survivors, 2))
    n_pairs = len(all_pairs)
    log.info("Testing %d pairs from %d survivors...", n_pairs, len(survivors))

    results: list[dict] = []
    n_skipped_overlap = 0
    t0 = time.time()

    for i, (ticker_a, ticker_b) in enumerate(all_pairs):
        if i > 0 and i % 10_000 == 0:
            log.info("  Pairs progress: %d/%d (%.0fs)", i, n_pairs, time.time() - t0)

        s_a = formation_data[ticker_a]["log_close"]
        s_b = formation_data[ticker_b]["log_close"]

        sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
        valid = ~(sa_aligned.isna() | sb_aligned.isna())
        sa_aligned, sb_aligned = sa_aligned[valid], sb_aligned[valid]

        n_aligned = len(sa_aligned)
        # Spec §1.2.1: denominator = expected total bars in the formation window,
        # not max of the two tickers' actual bars (which understates sparsity).
        overlap_ratio = n_aligned / max_formation_bars
        if overlap_ratio < _MIN_OVERLAP:
            n_skipped_overlap += 1
            continue

        log_a = sa_aligned.values
        log_b = sb_aligned.values
        X = np.column_stack([log_a, log_b])  # stack once, reuse in both PCA and Johansen

        alpha_pca, beta_pca, R = _pca_hedge_ratio(log_a, log_b)
        if np.isnan(alpha_pca):
            continue

        pval = _johansen_pvalue(X)
        if np.isnan(pval):
            continue

        results.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "alpha_pca": alpha_pca,
            "beta_pca": beta_pca,
            "R_measurement_noise": R,
            "johansen_pval": pval,
            "n_overlapping_bars": n_aligned,
            "overlap_ratio": overlap_ratio,
        })

    log.info(
        "Pairwise tests done: %d results, %d skipped (overlap<80%%) | %.0fs",
        len(results), n_skipped_overlap, time.time() - t0,
    )
    return results


def _apply_fdr_and_halflife(
    results: list[dict],
    formation_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply BH-FDR (q=0.05) then OU half-life filter [1, 10] trading days.
    Re-aligns only FDR survivors (typically <500 pairs) — no cached arrays needed.
    """
    if not results:
        return pd.DataFrame()

    pvals = np.array([r["johansen_pval"] for r in results])
    reject = bh_fdr_correct(pvals, q=_BH_FDR_Q)
    log.info("BH-FDR: %d / %d pairs survive (q=%.2f)", reject.sum(), len(reject), _BH_FDR_Q)

    surviving: list[dict] = []
    n_hl_fail = 0

    for row, keep in zip(results, reject):
        if not keep:
            continue

        s_a = formation_data[row["ticker_a"]]["log_close"]
        s_b = formation_data[row["ticker_b"]]["log_close"]
        sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
        valid = ~(sa_aligned.isna() | sb_aligned.isna())
        log_a = sa_aligned[valid].values
        log_b = sb_aligned[valid].values

        spread = log_a - row["alpha_pca"] - row["beta_pca"] * log_b
        hl_days = compute_ou_halflife(spread, bars_per_day=_BARS_PER_DAY_5MIN)

        if np.isnan(hl_days) or not (_HL_MIN_DAYS <= hl_days <= _HL_MAX_DAYS):
            n_hl_fail += 1
            continue

        surviving.append({**row, "half_life_days": hl_days})

    _MAX_PAIRS_POST_FILTER = 500
    n_before_cap = len(surviving)
    if len(surviving) > _MAX_PAIRS_POST_FILTER:
        surviving.sort(key=lambda r: r["johansen_pval"])
        surviving = surviving[:_MAX_PAIRS_POST_FILTER]
        log.warning(
            "PAIR-COUNT GATE: %d pairs capped to %d after HL filter (spike fold suppressed)",
            n_before_cap, _MAX_PAIRS_POST_FILTER,
        )

    log.info(
        "OU half-life filter: %d passed (%d dropped outside [%.0f, %.0f]d) → %d after pair-count cap",
        n_before_cap, n_hl_fail, _HL_MIN_DAYS, _HL_MAX_DAYS, len(surviving),
    )
    return pd.DataFrame(surviving)


def run(
    formation_start: str,
    formation_end: str,
    validated_dir: Path = VALIDATED_DIR,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """
    Run Phase 1 cointegration discovery for one fold.

    Parameters
    ----------
    formation_start : "YYYY-MM-DD"
    formation_end   : "YYYY-MM-DD"
    validated_dir   : location of Phase 0 outputs
    max_pairs       : fallback cap — keeps top-K by Johansen p-value before BH-FDR.
                      None = no cap (default).

    Returns
    -------
    pd.DataFrame with columns:
        ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise,
        johansen_pval, n_overlapping_bars, overlap_ratio, half_life_days
    Empty DataFrame if no pairs survive.
    """
    t0 = time.time()
    log.info("Phase 1 | formation: %s to %s", formation_start, formation_end)

    formation_data = _load_formation_data(formation_start, formation_end, validated_dir)
    if not formation_data:
        log.warning("No formation data — empty fold.")
        return pd.DataFrame()

    survivors, _, max_formation_bars = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        log.warning("Fewer than 2 survivors after hard screens.")
        return pd.DataFrame()

    formation_data = {t: formation_data[t] for t in survivors}

    raw_results = _run_pairwise_tests(survivors, formation_data, max_formation_bars)
    if not raw_results:
        log.warning("No pairs passed overlap filter.")
        return pd.DataFrame()

    if max_pairs is not None and len(raw_results) > max_pairs:
        raw_results.sort(key=lambda r: r["johansen_pval"])
        raw_results = raw_results[:max_pairs]
        log.warning("FALLBACK CAP: kept top %d pairs by Johansen p-value", max_pairs)

    pairs_df = _apply_fdr_and_halflife(raw_results, formation_data)
    log.info("Phase 1 complete: %d surviving pairs | %.0fs", len(pairs_df), time.time() - t0)
    return pairs_df
