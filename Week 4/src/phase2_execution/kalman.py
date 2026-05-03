"""
Phase 2 — Kalman Filter (2D state [alpha, beta])

State equation:      theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q),  Q = delta * R * I_2
Observation:         ln(A_t) = alpha_t + beta_t * ln(B_t) + eps_t,  eps_t ~ N(0, R)

NOTE: Q = delta * R * I (not delta * I as in raw spec).
Reason: R spans 0.001-33 across pairs. With Q=delta*I the delta/R adaptation rate varies
5 orders of magnitude — tight pairs get near-infinite gain, making prior spread white noise.
Using Q=delta*R*I normalises delta/R = delta for all pairs. Spec §2.1 states dynamics are
controlled by delta/R; this enforces it uniformly. Validated on Fold 1 (24 pairs).

Init: theta_0 = [alpha_PCA, beta_PCA],  P_0 = R * I_2

Signal uses PRIOR spread (before incorporating A_t observation).
Posterior theta used only for sizing and threshold rebalance.

Performance: inner loop is Numba JIT-compiled. ~50x faster than pure Python for
typical series lengths (7,800 1-min bars or 9,800 5-min bars per pair).
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _kalman_inner(log_a, log_b, alpha_init, beta_init, R, delta):
    """
    Numba-compiled 2x2 Kalman filter inner loop.
    All 2x2 matrix ops are inlined as scalar arithmetic — avoids numpy overhead per bar.

    Returns (spread_prior, alpha_prior, beta_prior, alpha_post, beta_post),
    each length-n array.
    """
    n = len(log_a)
    Q_scale = delta * R     # Q = Q_scale * I_2

    # P stored as 4 scalars (symmetric 2x2)
    P00 = R;  P01 = 0.0
    P10 = 0.0; P11 = R

    th0 = alpha_init   # alpha state
    th1 = beta_init    # beta state

    spread_prior = np.empty(n)
    alpha_prior  = np.empty(n)
    beta_prior   = np.empty(n)
    alpha_post   = np.empty(n)
    beta_post    = np.empty(n)

    for i in range(n):
        lb = log_b[i]

        # PREDICT: P_pred = P + Q*I
        Pp00 = P00 + Q_scale
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + Q_scale

        # Prior state = theta_pred (no dynamics noise in mean)
        alpha_prior[i]  = th0
        beta_prior[i]   = th1
        spread_prior[i] = log_a[i] - th0 - th1 * lb

        # H = [1, lb]  (observation vector)
        h0 = 1.0
        h1 = lb

        # Innovation covariance S = H @ Pp @ H' + R  (scalar)
        S = h0*h0*Pp00 + h0*h1*Pp01 + h1*h0*Pp10 + h1*h1*Pp11 + R
        if S < 1e-300:
            alpha_post[i] = th0
            beta_post[i]  = th1
            continue

        # Kalman gain K = Pp @ H' / S  (2x1)
        K0 = (Pp00*h0 + Pp01*h1) / S
        K1 = (Pp10*h0 + Pp11*h1) / S

        # Innovation
        innov = log_a[i] - th0*h0 - th1*h1

        # Update state
        th0 = th0 + K0 * innov
        th1 = th1 + K1 * innov

        # Joseph stabilised covariance: (I-KH) Pp (I-KH)' + R*KK'
        # IKH = I - outer(K, H)
        IKH00 = 1.0 - K0*h0;  IKH01 = -K0*h1
        IKH10 = -K1*h0;       IKH11 = 1.0 - K1*h1

        # M = IKH @ Pp
        M00 = IKH00*Pp00 + IKH01*Pp10
        M01 = IKH00*Pp01 + IKH01*Pp11
        M10 = IKH10*Pp00 + IKH11*Pp10
        M11 = IKH10*Pp01 + IKH11*Pp11

        # P = M @ IKH' + R * outer(K, K)
        P00 = M00*IKH00 + M01*IKH01 + R*K0*K0
        P01 = M00*IKH10 + M01*IKH11 + R*K0*K1
        P10 = M10*IKH00 + M11*IKH01 + R*K1*K0
        P11 = M10*IKH10 + M11*IKH11 + R*K1*K1

        alpha_post[i] = th0
        beta_post[i]  = th1

    return spread_prior, alpha_prior, beta_prior, alpha_post, beta_post


def run_kalman(
    log_a: np.ndarray,
    log_b: np.ndarray,
    alpha_init: float,
    beta_init: float,
    R: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Kalman filter for one pair over a price series.

    Parameters
    ----------
    log_a, log_b : 1-D arrays of aligned log-prices (same length T)
    alpha_init   : PCA intercept from Phase 1
    beta_init    : PCA hedge ratio from Phase 1
    R            : observation noise variance (PCA residual variance from Phase 1)
    delta        : state noise scale; Q = delta * R * I_2

    Returns (all length-T arrays)
    -------
    spread_prior : prior spread S_t^prior = ln(A_t) - alpha_{t|t-1} - beta_{t|t-1}*ln(B_t)
    alpha_prior  : alpha_{t|t-1}   (use for signal)
    beta_prior   : beta_{t|t-1}    (use for signal)
    alpha_post   : alpha_{t|t}     (use for sizing / rebalance)
    beta_post    : beta_{t|t}      (use for sizing / rebalance)
    """
    if R <= 0 or np.isnan(R):
        raise ValueError(f"R must be positive finite, got {R}")

    log_a = np.asarray(log_a, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    return _kalman_inner(log_a, log_b,
                         float(alpha_init), float(beta_init),
                         float(R), float(delta))


def warmup_kalman():
    """Pre-compile Numba Kalman JIT. Call once at process startup."""
    dummy = np.linspace(0.1, 0.2, 50)
    _kalman_inner(dummy, dummy, 0.0, 1.0, 0.01, 1e-7)
