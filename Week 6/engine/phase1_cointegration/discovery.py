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

from engine.utils.io import VALIDATED_DIR, list_validated_tickers, read_5min
from engine.utils.stats import bh_fdr_correct, compute_ou_halflife
from engine.phase1_cointegration.factor_residual import fit_factor_model

log = logging.getLogger(__name__)

_MIN_MEDIAN_PRICE = 5.0
_MIN_ADV_DOLLAR = 1_000_000
_MIN_COMPLETENESS = 0.90
_MAX_ZERO_RETURN = 0.50
_MIN_OVERLAP = 0.80
_BH_FDR_Q = 0.05
_HL_MIN_DAYS = 1.0
_HL_MAX_DAYS = 6.0           # V3.0 F3: tightened from 10.0 (was double-capped by runner)
_BARS_PER_DAY_5MIN = 78      # pipeline spec §1.6 convention (actual session = 77 bars)
_N_FACTOR_COMPONENTS = 5     # V3.0 F1: PCA factors projected out before Johansen


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


def _johansen_test_chunk(args):
    """
    Worker function for parallel Johansen testing. Must be at module level so
    multiprocessing can pickle it on Windows (spawn mode).

    args = (pair_indices, X, tickers, max_formation_bars)
        pair_indices: list of (i, j) column-index pairs to test
        X:            (T, n_tickers) numpy array, NaN where ticker has missing bar
        tickers:      list of ticker symbols, X column order
        max_formation_bars: spec §1.2.1 denominator for overlap_ratio

    Returns: list of result dicts.
    """
    # Force single-threaded BLAS in each worker. Without this, BLAS thread
    # scheduling produces non-deterministic floating-point sums in LAPACK calls
    # (Johansen test uses these), causing pairs near the top-500 p-value cap
    # to shuffle in/out unpredictably and changing downstream trade selection.
    # Each Johansen call is on a 2-variable system, so single-thread BLAS is
    # not slower (often faster — eliminates thread launch overhead per call).
    import os as _os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ[var] = "1"

    # Re-imported per worker on Windows spawn; cached after first call thanks
    # to Defender exclusions and Python module cache.
    from scipy.stats import chi2 as _chi2
    from statsmodels.tsa.vector_ar.vecm import coint_johansen as _coint_johansen

    pair_indices, X, tickers, max_formation_bars = args
    results: list[dict] = []

    for i, j in pair_indices:
        log_a_full = X[:, i]
        log_b_full = X[:, j]
        valid = ~(np.isnan(log_a_full) | np.isnan(log_b_full))
        n_aligned = int(valid.sum())
        overlap_ratio = n_aligned / max_formation_bars
        if overlap_ratio < _MIN_OVERLAP:
            continue

        log_a = log_a_full[valid]
        log_b = log_b_full[valid]
        Xpair = np.column_stack([log_a, log_b])

        alpha_pca, beta_pca, R = _pca_hedge_ratio(log_a, log_b)
        if np.isnan(alpha_pca):
            continue

        try:
            res = _coint_johansen(Xpair, det_order=0, k_ar_diff=1)
            pval = float(1.0 - _chi2.cdf(float(res.lr1[0]), df=8))
        except Exception:
            pval = float("nan")
        if np.isnan(pval):
            continue

        results.append({
            "ticker_a": tickers[i],
            "ticker_b": tickers[j],
            "alpha_pca": alpha_pca,
            "beta_pca": beta_pca,
            "R_measurement_noise": R,
            "johansen_pval": pval,
            "n_overlapping_bars": n_aligned,
            "overlap_ratio": overlap_ratio,
        })
    return results


def _run_pairwise_tests(
    survivors: list[str],
    formation_data: dict[str, pd.DataFrame],
    max_formation_bars: int,
    n_workers: int = 8,
) -> list[dict]:
    """
    Enumerate all pairs, apply overlap filter, PCA, and Johansen — parallelized.

    Optimization (2026-05-20): instead of per-pair pandas .align()+dropna() and
    sequential execution, we precompute the aligned (T, n_tickers) numpy matrix
    once (outer-join, NaN preserved), then dispatch pair-index chunks to a Pool
    of n_workers processes. On i5-12500H (12 cores), ~4-5x speedup observed.
    Bit-identical results to serial version (verified via test_parallel_safe.py).

    max_formation_bars: expected total bars in the formation window (spec §1.2.1
    denominator). Must come from _apply_hard_screens so it reflects the full
    window capacity, not just the two tickers being tested.
    """
    from itertools import combinations as _combinations
    from multiprocessing import Pool

    n_pairs = len(survivors) * (len(survivors) - 1) // 2
    log.info("Testing %d pairs from %d survivors (%d workers)...",
             n_pairs, len(survivors), n_workers)
    t0 = time.time()

    # Precompute aligned matrix (NaN where ticker missing at that bar).
    # Outer-join so per-pair valid-bar count can be measured from the NaN mask;
    # eliminates 124,750 pandas .align() calls.
    aligned = pd.concat(
        {tk: formation_data[tk]["log_close"] for tk in survivors},
        axis=1,
        join="outer",
    )
    tickers = list(aligned.columns)
    X = aligned.values  # (T, n_tickers), NaN where missing
    log.info("  precomputed (T=%d, n=%d) matrix in %.1fs", X.shape[0], X.shape[1], time.time() - t0)

    all_pairs = list(_combinations(range(len(tickers)), 2))

    # Split into chunks for worker distribution.
    chunk_size = (len(all_pairs) + n_workers - 1) // n_workers
    chunks = [all_pairs[k:k + chunk_size] for k in range(0, len(all_pairs), chunk_size)]
    log.info("  dispatching %d chunks (chunk_size=%d) across %d workers",
             len(chunks), chunk_size, n_workers)

    # Parallel execution.
    t1 = time.time()
    if n_workers <= 1:
        # Serial fallback path (also useful for debugging).
        chunk_results = [_johansen_test_chunk((c, X, tickers, max_formation_bars))
                         for c in chunks]
    else:
        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(
                _johansen_test_chunk,
                [(c, X, tickers, max_formation_bars) for c in chunks],
            )

    results = [r for chunk in chunk_results for r in chunk]
    log.info(
        "Pairwise tests done: %d results in %.1fs (workers: %.1fs, total: %.1fs)",
        len(results), time.time() - t0, time.time() - t1, time.time() - t0,
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


def apply_beta_filter(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    V3.0 R5: drop pairs with beta_pca <= 0.

    Negative-beta "pairs" are co-trending (both legs move same direction), not
    a real long-short pair trade. V2.0 had 16.7% of trades with beta<0, which
    is silent factor exposure mislabeled as dollar-neutral.
    """
    # ---- HARD STOPS ----
    if pairs_df is None or len(pairs_df) == 0:
        return pd.DataFrame() if pairs_df is None else pairs_df
    if "beta_pca" not in pairs_df.columns:
        raise KeyError("apply_beta_filter: input pairs_df missing 'beta_pca' column")

    n_before = len(pairs_df)
    out = pairs_df[pairs_df["beta_pca"] > 0].reset_index(drop=True)
    n_dropped = n_before - len(out)
    pct = 100.0 * n_dropped / max(n_before, 1)
    log.info(
        "beta_pca>0 filter: %d -> %d (%d dropped = %.1f%%)",
        n_before, len(out), n_dropped, pct,
    )
    return out


def run(
    formation_start: str,
    formation_end: str,
    validated_dir: Path = VALIDATED_DIR,
    max_pairs: int | None = None,
    formation_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Phase 1 cointegration discovery for one fold (V3.0).

    V3.0 changes vs V2.0:
        - Fits PCA factor model on hard-screen survivors (5 components)
        - Replaces raw log-prices with factor-residual log-prices everywhere downstream
        - Tightens HL filter to [1, 6d] (was [1, 10d])
        - Adds beta_pca > 0 filter after BH-FDR + HL

    Parameters
    ----------
    formation_start : "YYYY-MM-DD"
    formation_end   : "YYYY-MM-DD"
    validated_dir   : location of Phase 0 outputs
    max_pairs       : fallback cap — keeps top-K by Johansen p-value before BH-FDR.

    Returns
    -------
    pairs_df : pd.DataFrame with columns
        ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise,
        johansen_pval, n_overlapping_bars, overlap_ratio, half_life_days
    factor_state : dict with keys
        'loadings_W'  : (n_components, n_tickers) numpy array
        'tickers'     : list[str] matching W column order
        'diagnostics' : dict from fit_factor_model

    Empty pairs_df + empty factor_state if no pairs survive.
    """
    t0 = time.time()
    log.info("Phase 1 V3.0 | formation: %s to %s", formation_start, formation_end)
    print(f"  [discovery] starting fold {formation_start} -> {formation_end}", flush=True)

    if formation_data is None:
        formation_data = _load_formation_data(formation_start, formation_end, validated_dir)
    if not formation_data:
        log.warning("No formation data — empty fold.")
        return pd.DataFrame(), {}
    print(f"  [discovery] loaded {len(formation_data)} formation tickers", flush=True)

    survivors, _, max_formation_bars = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        log.warning("Fewer than 2 survivors after hard screens.")
        return pd.DataFrame(), {}
    print(f"  [discovery] hard screens: {len(survivors)} survivors", flush=True)

    formation_data = {t: formation_data[t] for t in survivors}

    # ----- V3.0 F1: factor model fit + residual substitution -----
    print(f"  [discovery] fitting factor model (PCA via SVD)...", flush=True)
    try:
        loadings_W, factor_tickers, residual_log_prices, fm_diag = fit_factor_model(
            formation_data, n_components=_N_FACTOR_COMPONENTS,
        )
    except ValueError as e:
        log.warning("factor_residual_fit failed: %s — empty fold", e)
        return pd.DataFrame(), {}

    # Replace each ticker's log_close with its residual log-price.
    formation_data_residual: dict[str, pd.DataFrame] = {}
    for tk in factor_tickers:
        df = formation_data[tk].copy()
        df = df.loc[df.index.intersection(residual_log_prices.index)]
        df["log_close"] = residual_log_prices[tk].reindex(df.index)
        df = df.dropna(subset=["log_close"])
        if len(df) >= 200:
            formation_data_residual[tk] = df

    survivors_residual = list(formation_data_residual.keys())
    if len(survivors_residual) < 2:
        log.warning("Fewer than 2 survivors after factor projection.")
        return pd.DataFrame(), {}

    # V3.0 F1: residual data has fewer bars than raw (inner-join across all tickers
    # in factor model). Recompute max_bars on the residual to keep overlap_ratio
    # well-calibrated for the §1.2.1 80% gate.
    max_formation_bars = max(len(df) for df in formation_data_residual.values())

    log.info(
        "factor_residual_fit: tickers=%d, cum_var=%.3f, max_bars=%d",
        len(factor_tickers), fm_diag["cumulative_variance_explained"], max_formation_bars,
    )
    # -------------------------------------------------------------

    print(f"  [discovery] running pairwise Johansen on {len(survivors_residual)} tickers "
          f"({len(survivors_residual)*(len(survivors_residual)-1)//2} pairs)...", flush=True)
    raw_results = _run_pairwise_tests(
        survivors_residual, formation_data_residual, max_formation_bars,
    )
    if not raw_results:
        log.warning("No pairs passed overlap filter.")
        return pd.DataFrame(), {}
    print(f"  [discovery] pairwise tests done: {len(raw_results)} pairs passed overlap+PCA+Johansen", flush=True)

    if max_pairs is not None and len(raw_results) > max_pairs:
        raw_results.sort(key=lambda r: r["johansen_pval"])
        raw_results = raw_results[:max_pairs]
        log.warning("FALLBACK CAP: kept top %d pairs by Johansen p-value", max_pairs)

    pairs_df = _apply_fdr_and_halflife(raw_results, formation_data_residual)

    # ----- V3.0 R5: beta_pca > 0 filter -----
    pairs_df = apply_beta_filter(pairs_df)

    factor_state = {
        "loadings_W": loadings_W,
        "tickers": factor_tickers,
        "diagnostics": fm_diag,
        # Pass the IN-SAMPLE residuals (inner-joined, the same ones used for
        # Johansen + PCA hedge ratio above) so downstream callers can use the
        # SAME residual series as discovery used — keeps persistence gate
        # consistent with discovery, not testing on a different ffill-augmented
        # outer-join projection.
        "residual_log_prices": residual_log_prices,
    }

    log.info(
        "Phase 1 V3.0 complete: %d pairs (after beta>0 filter) | %.0fs",
        len(pairs_df), time.time() - t0,
    )
    return pairs_df, factor_state
