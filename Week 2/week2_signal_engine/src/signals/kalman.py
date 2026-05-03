"""
Kalman Filter Dynamic Hedge Ratio

Estimates time-varying alpha(t) and beta(t) using a state-space model
where the hedge ratio follows a random walk.  The filter is causal:
beta(t) uses only data from bars 1..t.

State-space model
-----------------
State:       theta_t = [alpha_t, beta_t]
Transition:  theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q)   (random walk)
Observation: log(A_t) = [1, log(B_t)] @ theta_t + v_t,  v_t ~ N(0, R)

Parameters
----------
Q = (delta / (1 - delta)) * I_2    -- process noise; controls adaptation speed
R                                   -- observation noise; estimated from OLS
                                       residuals on the first 500 bars only,
                                       never from the trading period
theta_0 = [0.0, 1.0]               -- neutral prior: zero intercept, unit beta
P_0 = I_2                          -- moderate initial uncertainty

delta guide
-----------
1e-6  very slow   (beta half-life ~1800 trading days -- essentially static)
1e-5  slow        (beta half-life ~180 trading days  -- default; weekly-regime)
1e-4  moderate    (beta half-life ~18 trading days)
1e-3  fast        (beta half-life ~1.8 trading days  -- fits noise)

Usage (in main() / validate_kalman.py)
---------------------------------------
Run on the FULL price series (Jan-Dec). The formation period warms up the
filter; the trading-period beta is sliced afterward.  Never re-initialize
the filter at the formation/trading boundary.

    alpha_full, beta_full, diag = kalman_hedge_ratio(
        merged["close_a"], merged["close_b"], delta=1e-5
    )
    alpha_trade = alpha_full.loc[trading.index]
    beta_trade  = beta_full.loc[trading.index]

!! FLAG -- Python loop performance:
    The Kalman recursion is a Python loop over ~97k bars per pair.  On modern
    hardware this runs in ~0.5-1 second.  Numba is NOT applied here: the loop
    body mixes 2x2 numpy matrix ops with Python control flow, and the compile
    overhead would exceed the runtime for a single pair.  For an 11-pair sweep
    the total cost is ~10 seconds -- acceptable without JIT.

!! FLAG -- R initialization window:
    R is estimated from the first min(500, n//10) bars via OLS.  This window
    must be short relative to the formation period to avoid using too much data
    for prior calibration.  It is deterministic given the data and introduces
    no lookahead because it uses only the earliest available bars.
"""
import numpy as np
import pandas as pd


def kalman_hedge_ratio(
    close_a: pd.Series,
    close_b: pd.Series,
    delta: float = 1e-5,
) -> tuple[pd.Series, pd.Series, dict]:
    """Estimate time-varying hedge ratio via Kalman filter.

    Args:
        close_a : Price series for leg A.  Must share a DatetimeIndex with close_b.
        close_b : Price series for leg B.
        delta   : State noise scalar.  Q = (delta / (1 - delta)) * I_2.
                  Default 1e-5 gives a slow-adapting beta (half-life ~180 trading
                  days) appropriate for utility-pair regime changes.

    Returns:
        alpha_series : pd.Series — time-varying intercept alpha(t), same index as inputs.
        beta_series  : pd.Series — time-varying hedge ratio beta(t), same index as inputs.
        diagnostics  : dict with keys:
                           R               -- observation noise variance used
                           delta           -- adaptation speed parameter
                           final_alpha     -- filter state at last bar
                           final_beta      -- filter state at last bar
                           innovations_std -- std of prediction errors (health check)
                           n_bars          -- total bars processed
    """
    if len(close_a) != len(close_b):
        raise ValueError(
            f"close_a and close_b must have the same length "
            f"({len(close_a)} vs {len(close_b)})"
        )

    n     = len(close_a)
    log_a = np.log(close_a.values.astype(np.float64))
    log_b = np.log(close_b.values.astype(np.float64))

    # ── Process noise covariance ──────────────────────────────────────────────
    Ve = delta / (1.0 - delta)
    Q  = Ve * np.eye(2)

    # ── Observation noise R — estimated from short OLS warm-up window ─────────
    N_init   = min(500, n)
    X_init   = np.column_stack([np.ones(N_init), log_b[:N_init]])
    p_init   = np.linalg.lstsq(X_init, log_a[:N_init], rcond=None)[0]
    resid    = log_a[:N_init] - X_init @ p_init
    R        = max(float(np.var(resid, ddof=1)), 1e-8)   # must be positive

    # ── Initial state and covariance ──────────────────────────────────────────
    theta = np.array([0.0, 1.0])   # neutral prior: alpha=0, beta=1
    P     = np.eye(2)              # moderate uncertainty; converges within ~200 bars

    # ── Storage ───────────────────────────────────────────────────────────────
    alpha_vals  = np.empty(n)
    beta_vals   = np.empty(n)
    innovations = np.empty(n)
    I2          = np.eye(2)

    # ── Kalman recursion ──────────────────────────────────────────────────────
    for i in range(n):
        H = np.array([1.0, log_b[i]])   # observation vector (1x2)

        # Predict
        P_pred = P + Q

        # Innovation
        innov        = log_a[i] - (H @ theta)
        innovations[i] = innov

        # Innovation variance (scalar)
        S = float(H @ P_pred @ H) + R

        # Kalman gain (2,)
        K = (P_pred @ H) / S

        # Update state and covariance
        theta = theta + K * innov
        P     = (I2 - np.outer(K, H)) @ P_pred

        alpha_vals[i] = theta[0]
        beta_vals[i]  = theta[1]

    diagnostics = {
        "R":               round(R, 8),
        "delta":           delta,
        "final_alpha":     round(float(theta[0]), 6),
        "final_beta":      round(float(theta[1]), 6),
        "innovations_std": round(float(np.std(innovations)), 6),
        "n_bars":          n,
    }

    return (
        pd.Series(alpha_vals, index=close_a.index, name="alpha_kalman"),
        pd.Series(beta_vals,  index=close_a.index, name="beta_kalman"),
        diagnostics,
    )
