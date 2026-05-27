"""
Phase 4e — Overfitting Diagnostics

Implements Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting
(PBO) from scratch without mlfinlab, using fold equity parquets as input.

DSR: Bailey & López de Prado (2014) — adjusts Sharpe for number of trials
     and non-normality of returns.

PBO: Combinatorial cross-validation on fold Sharpes — estimates the probability
     that the selected strategy (by in-sample Sharpe) underperforms out-of-sample.

Outputs:
  results/metrics/overfitting_diagnostics.csv
"""

from __future__ import annotations

import itertools
import logging
import random

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from engine.phase4_defense.orchestrator import (
    load_fold_metrics,
    get_daily_returns,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DSR — Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Adjusts the observed Sharpe for:
      - Finite sample (n_obs)
      - Multiple trials (n_trials) → inflated max expected Sharpe
      - Non-normality (skewness, excess kurtosis of returns)

    Returns DSR ∈ [0, 1]: probability that true Sharpe > benchmark.
    """
    if n_obs < 5 or n_trials < 1:
        return np.nan

    # Expected maximum Sharpe under repeated trials (standard normal order stat)
    e_max_sr = _expected_max_sharpe(n_trials)

    # Variance of Sharpe estimator under non-normality
    # Var(SR_hat) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt+3)/4*SR^2) / (n-1)
    # Using the simplified form from Bailey & López de Prado
    sr_var = (
        1.0
        + (0.5 * sharpe_obs ** 2)
        - (skew * sharpe_obs)
        + ((kurt_excess / 4.0) * sharpe_obs ** 2)
    ) / max(n_obs - 1, 1)

    sr_std = np.sqrt(max(sr_var, 1e-12))
    # Deflate: compare observed SR against max expected SR (selection bias) plus benchmark
    dsr = float(scipy_stats.norm.cdf(
        (sharpe_obs - max(e_max_sr, sharpe_benchmark)) / sr_std
    ))
    return dsr


def _expected_max_sharpe(n_trials: int) -> float:
    """
    E[max(SR_1,...,SR_T)] for T iid N(0,1) trials.
    Approximation: E[max] ≈ z(1 - 1/T) where z is the normal quantile.
    """
    if n_trials <= 1:
        return 0.0
    p = 1.0 - 1.0 / n_trials
    return float(scipy_stats.norm.ppf(p))


def compute_dsr(
    fold_metrics: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute DSR for the strategy.

    - Observed Sharpe = mean fold Sharpe (equal-weighted)
    - n_obs = number of trading days across all folds
    - n_trials = number of completed folds (each fold is a "trial")
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    completed = len(fold_metrics)
    if completed < 2:
        return {"dsr": np.nan, "mean_sharpe": np.nan, "n_trials": completed}

    # Aggregate daily returns across all folds
    all_daily: list[float] = []
    for fold_n in fold_metrics["fold"].tolist():
        dr = get_daily_returns(fold_n)
        if dr is not None and len(dr) > 0:
            all_daily.extend(dr.tolist())

    if len(all_daily) < 10:
        return {"dsr": np.nan, "mean_sharpe": fold_metrics["sharpe"].mean(), "n_trials": completed}

    arr = np.array(all_daily)
    n_obs   = len(arr)
    mu      = float(np.mean(arr))
    sigma   = float(np.std(arr, ddof=1))
    skew    = float(scipy_stats.skew(arr))
    kurt_ex = float(scipy_stats.kurtosis(arr, fisher=True))  # excess kurtosis

    # Annualised Sharpe from daily returns
    sr_annual = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else np.nan

    dsr = deflated_sharpe_ratio(
        sharpe_obs       = sr_annual,
        n_obs            = n_obs,
        n_trials         = completed,
        skew             = skew,
        kurt_excess      = kurt_ex,
        sharpe_benchmark = 0.0,
    )

    return {
        "dsr":          dsr,
        "mean_sharpe":  sr_annual,
        "n_trials":     completed,
        "n_daily_obs":  n_obs,
        "daily_skew":   skew,
        "daily_kurt":   kurt_ex,
    }


# ---------------------------------------------------------------------------
# PBO — Probability of Backtest Overfitting (combinatorial)
# ---------------------------------------------------------------------------

def compute_pbo(
    fold_metrics: pd.DataFrame | None = None,
    n_splits: int = 4,
) -> dict[str, float]:
    """
    Simplified PBO via combinatorial cross-validation on fold Sharpes.

    For each combination of (n_splits/2) folds as "IS" and the rest as "OOS":
      - Select best "strategy" in IS = fold with highest IS Sharpe
      - Evaluate same strategy in OOS
      - Count: does IS-best also beat OOS median?

    PBO = fraction of paths where IS-best underperforms OOS median.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    folds = fold_metrics["fold"].tolist()
    n     = len(folds)

    if n < 4:
        return {"pbo": np.nan, "n_paths": 0, "n_folds": n}

    sharpe_by_fold = dict(zip(fold_metrics["fold"], fold_metrics["sharpe"]))
    k = max(2, n // n_splits)   # IS set size

    n_overfit = 0
    n_total   = 0
    rng       = random.Random(42)  # fixed seed for reproducibility

    # C(32, 8) ≈ 10M paths — enumerate a random sample to avoid sequential bias.
    # Sequential iteration over itertools.combinations would over-sample paths
    # where early folds are always in IS, biasing the PBO estimate.
    all_combos = list(itertools.combinations(range(n), k))
    if len(all_combos) > 10_000:
        sampled = rng.sample(all_combos, 10_000)
    else:
        sampled = all_combos

    for is_combo in sampled:
        is_idx  = set(is_combo)
        oos_idx = [i for i in range(n) if i not in is_idx]

        if not oos_idx:
            continue

        is_folds  = [folds[i] for i in is_idx]
        oos_folds = [folds[i] for i in oos_idx]

        is_sharpes  = [sharpe_by_fold[f] for f in is_folds]
        oos_sharpes = [sharpe_by_fold[f] for f in oos_folds]

        # IS-best fold (highest Sharpe in IS)
        is_best_idx  = int(np.argmax(is_sharpes))
        is_best_fold = is_folds[is_best_idx]

        # PBO: does IS-best underperform OOS median?
        oos_median     = float(np.median(oos_sharpes))
        is_best_sharpe = sharpe_by_fold[is_best_fold]

        if is_best_sharpe < oos_median:
            n_overfit += 1
        n_total += 1

    pbo = n_overfit / n_total if n_total > 0 else np.nan
    return {
        "pbo":     pbo,
        "n_paths": n_total,
        "n_folds": n,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_overfitting_diagnostics(
    fold_metrics: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute DSR and PBO, return as single-row DataFrame.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    log.info("Computing DSR...")
    dsr_result = compute_dsr(fold_metrics)

    log.info("Computing PBO...")
    pbo_result = compute_pbo(fold_metrics)

    row = {
        "raw_sharpe_annual":   dsr_result.get("mean_sharpe", np.nan),
        "n_folds_completed":   len(fold_metrics),
        "n_daily_obs":         dsr_result.get("n_daily_obs", np.nan),
        "daily_skew":          dsr_result.get("daily_skew", np.nan),
        "daily_kurt_excess":   dsr_result.get("daily_kurt", np.nan),
        "n_trials":            dsr_result.get("n_trials", np.nan),
        "dsr":                 dsr_result.get("dsr", np.nan),
        "pbo":                 pbo_result.get("pbo", np.nan),
        "pbo_n_paths":         pbo_result.get("n_paths", np.nan),
    }

    result = pd.DataFrame([row])

    if save:
        out = _METRICS_DIR / "overfitting_diagnostics.csv"
        result.to_csv(out, index=False)
        log.info("Overfitting diagnostics saved → %s", out)

    return result


def print_overfitting_report(result: pd.DataFrame | None = None) -> None:
    if result is None:
        result = run_overfitting_diagnostics()

    row = result.iloc[0]
    print("\n=== Overfitting Diagnostics ===")
    print(f"  N folds completed:      {int(row['n_folds_completed'])}")
    print(f"  N trading days (obs):   {int(row['n_daily_obs'])}")
    print(f"  Raw Sharpe (annual):    {row['raw_sharpe_annual']:.3f}")
    print(f"  Daily return skew:      {row['daily_skew']:.3f}")
    print(f"  Daily return kurt(ex):  {row['daily_kurt_excess']:.3f}")
    print(f"  N trials (folds):       {int(row['n_trials'])}")
    print(f"  DSR:                    {row['dsr']:.4f}")
    print(f"  PBO:                    {row['pbo']:.4f}  ({int(row['pbo_n_paths'])} paths)")

    dsr_v = row["dsr"]
    pbo_v = row["pbo"]
    if not np.isnan(dsr_v):
        if dsr_v < 0.05:
            print("  [WARN] DSR < 0.05 — strategy likely overfit")
        elif dsr_v < 0.50:
            print("  [CAUTION] DSR < 0.50 — weak evidence of genuine alpha")
        else:
            print("  [OK] DSR ≥ 0.50")

    if not np.isnan(pbo_v):
        if pbo_v > 0.50:
            print("  [WARN] PBO > 0.50 -- IS selection likely does not generalise OOS")
        else:
            print("  [OK] PBO <= 0.50")
