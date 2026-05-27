"""
Phase 2 — Static Spread Compute (V3.0)
=======================================

Replaces the misnamed Kalman filter (with δ=1e-7 pinning hedge ratio constant).
With V3.0's PCA hedge ratio from formation, the spread is genuinely static:

    spread[t] = log_a[t] - alpha_pca - beta_pca * log_b[t]

No state, no innovation, no posterior. Hedge ratio does not adapt at runtime —
this is the honest version of what V2.0 was doing under the Kalman wrapper.

Beta drift / rebalance: with static beta, drift = 0, rebalance never fires.
The engine's rebalance code path is left intact (dead branch) for schema
compatibility. rebalance_log.csv will be empty under V3.0.
"""

from __future__ import annotations

import numpy as np


def compute_static_spread(
    log_a: np.ndarray,
    log_b: np.ndarray,
    alpha_pca: float,
    beta_pca: float,
) -> np.ndarray:
    """
    Static spread per bar.

    Parameters
    ----------
    log_a, log_b : np.ndarray, same length, log-prices of two legs
    alpha_pca    : float, intercept from PCA hedge ratio (discovery output)
    beta_pca     : float, slope from PCA hedge ratio (discovery output, must be > 0)

    Returns
    -------
    spread : np.ndarray, shape (n,)
    """
    # ---- HARD STOPS (input) ----
    assert isinstance(log_a, np.ndarray), f"log_a must be ndarray, got {type(log_a)}"
    assert isinstance(log_b, np.ndarray), f"log_b must be ndarray, got {type(log_b)}"
    assert len(log_a) == len(log_b), (
        f"length mismatch: len(log_a)={len(log_a)}, len(log_b)={len(log_b)}"
    )
    assert len(log_a) > 0, "log_a is empty"
    assert np.isfinite(alpha_pca), f"alpha_pca not finite: {alpha_pca}"
    assert np.isfinite(beta_pca), f"beta_pca not finite: {beta_pca}"
    assert beta_pca > 0, (
        f"V3.0 invariant violated: beta_pca={beta_pca} <= 0; "
        "R5 beta>0 filter should have stripped this pair"
    )

    # ---- COMPUTE ----
    spread = log_a - alpha_pca - beta_pca * log_b

    # ---- INVARIANT (output finite) ----
    assert np.all(np.isfinite(spread)), "compute_static_spread: output has nan/inf"

    return spread


def compute_formation_z_params(
    log_a_form: np.ndarray,
    log_b_form: np.ndarray,
    alpha_pca: float,
    beta_pca: float,
) -> tuple[float, float]:
    """
    Compute formation-window spread mean/std for locked Z-score normalization.

    Replaces the `run_kalman(...)` call at Week 4 engine.py L406 — that call
    was computing formation spread via Kalman to extract mean/std. Under V3.0
    static spread, this collapses to direct mean/std of the static spread.

    Parameters
    ----------
    log_a_form, log_b_form : np.ndarray, formation-window log-prices (5-min freq)
    alpha_pca, beta_pca    : PCA hedge ratio from discovery

    Returns
    -------
    (spread_mean_form, spread_std_form) : (float, float)
    """
    # ---- HARD STOPS (input) ----
    assert isinstance(log_a_form, np.ndarray), "log_a_form must be ndarray"
    assert isinstance(log_b_form, np.ndarray), "log_b_form must be ndarray"
    assert len(log_a_form) == len(log_b_form), "length mismatch"
    if len(log_a_form) <= 20:
        raise ValueError(
            f"compute_formation_z_params: need >20 bars, got {len(log_a_form)}"
        )

    # ---- COMPUTE ----
    spread_form = compute_static_spread(log_a_form, log_b_form, alpha_pca, beta_pca)
    spread_clean = spread_form[np.isfinite(spread_form)]

    if len(spread_clean) <= 20:
        raise ValueError(
            f"compute_formation_z_params: too few finite bars ({len(spread_clean)})"
        )

    spread_mean = float(np.mean(spread_clean))
    spread_std = float(np.std(spread_clean))

    # ---- INVARIANT (degenerate pair check) ----
    if spread_std <= 1e-10:
        raise ValueError(
            f"compute_formation_z_params: degenerate pair, spread_std={spread_std:.2e}"
        )
    if not (np.isfinite(spread_mean) and np.isfinite(spread_std)):
        raise ValueError("compute_formation_z_params: non-finite output")

    return spread_mean, spread_std
