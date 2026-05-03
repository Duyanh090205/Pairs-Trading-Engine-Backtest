"""
Spread Characterization

Half-life (OU AR(1) discretization), Hurst exponent (variance-ratio estimator),
and the window guardrail that converts half-life to a rolling window size.
"""
import numpy as np
import pandas as pd

# Guardrail constants matching configs/params_example.yaml
MIN_WINDOW     = 10
MAX_WINDOW     = 2000
DEFAULT_WINDOW = 60


def compute_half_life(spread: pd.Series) -> tuple[float, float]:
    """Estimate mean-reversion speed via OU AR(1) discretization.

    Regresses the first-difference of the spread on its lagged level:
        ΔS(t) = a + lambda * S(t-1) + epsilon

    The intercept (a) is included to avoid bias in spreads with non-zero mean.
    Without it, lambda — and therefore half-life — can be systematically wrong.

    Half-life = -ln(2) / lambda  when lambda < 0 (mean-reverting).

    Returns:
        (half_life, lambda_hat)
        half_life = np.inf  when lambda >= 0 (no mean reversion detected)
        half_life = np.nan  when the regression is numerically degenerate

    !! FLAG — interpretation caveats:
    1. Lambda is estimated on 1-min bars, so half-life is in MINUTES, not days.
       Divide by 390 if you want days.  The pipeline uses it directly as bars.
    2. For CMS/DUK formation (~47k bars) we expect a moderately short half-life.
       If HL > 240 (> 4 hours), the spread reverts very slowly and the engine
       will use the MAX_WINDOW = 240 cap.
    3. If spread variance is near-zero (constant series, trading halt), lstsq
       returns near-zero lambda -> very long HL -> capped at MAX_WINDOW (2000).
    """
    s = spread.dropna()
    lag   = s.shift(1)
    delta = s.diff()
    df_reg = pd.DataFrame({"delta": delta, "lag": lag}).dropna()

    if len(df_reg) < 2:
        return np.nan, np.nan

    X = np.column_stack([np.ones(len(df_reg)), df_reg["lag"].values])
    y = df_reg["delta"].values
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    _, lam = float(params[0]), float(params[1])

    if np.isnan(lam):
        return np.nan, np.nan
    if lam >= 0:
        # Non-negative lambda means no mean reversion (or explosive process).
        return np.inf, lam

    half_life = -np.log(2) / lam
    return half_life, lam


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Variance-ratio Hurst exponent estimator.

    For each lag τ in [2, max_lag), computes:
        tau(τ) = std(S(t+τ) − S(t))

    Then fits log(tau) ~ H * log(lag) via OLS. The slope H is the Hurst exponent.

    H < 0.5  → mean-reverting (favourable for pairs trading)
    H = 0.5  → random walk
    H > 0.5  → trending

    Returns np.nan if the estimator is degenerate (too few observations,
    zero-variance series, or a tau of 0 causing log(0)).

    !! FLAG — reliability caveats:
    1. This estimator is NOISY on short windows.  On the 47k-bar formation period
       it is reasonably stable, but on rolling sub-windows (Step 9) results
       should be treated as directional, not precise.
    2. The estimator is NOT bounded to [0, 1].  Extreme values (<0 or >1) signal
       a degenerate series or a very small sample — check the input.
    3. We use std (not var).  The slope from regressing log(std) on log(lag) is H
       directly (since std ~ lag^H → log(std) = H*log(lag) + const).
    4. max_lag=100 means we probe up to 100-minute (1.67h) autocorrelation structure.
       For pairs with longer-cycle mean reversion this may underestimate H.
    """
    vals = np.asarray(series.dropna(), dtype=np.float64)
    n = len(vals)

    if n < max_lag + 10:
        return np.nan

    lags = range(2, min(max_lag, n // 2))
    tau  = [np.std(vals[lag:] - vals[:-lag], ddof=1) for lag in lags]

    # Guard against zero-variance differences (constant spread segment)
    tau_arr = np.array(tau)
    if np.any(tau_arr <= 0):
        return np.nan

    slope = np.polyfit(np.log(list(lags)), np.log(tau_arr), 1)[0]
    return float(slope)


def window_from_half_life(
    half_life: float,
    min_w: int = MIN_WINDOW,
    max_w: int = MAX_WINDOW,
    default_w: int = DEFAULT_WINDOW,
) -> tuple[int, str]:
    """Convert a half-life (in bars) to a rolling window size with guardrails.

    Returns:
        (window, note)  where note explains any clamping or fallback applied.

    Guardrails (from research plan Step 3):
    - NaN, Inf, or <= 0  → use default_w  (no mean reversion detected)
    - Valid but < min_w  → clamp to min_w  (very fast reversion)
    - Valid but > max_w  → clamp to max_w  (very slow reversion)
    """
    if np.isnan(half_life) or np.isinf(half_life) or half_life <= 0:
        return default_w, f"half_life={half_life} invalid, using default {default_w}"

    raw = int(round(half_life))
    if raw < min_w:
        return min_w, f"half_life={half_life:.1f}, raw={raw} clamped up to min {min_w}"
    if raw > max_w:
        return max_w, f"half_life={half_life:.1f}, raw={raw} clamped down to max {max_w}"
    return raw, f"half_life={half_life:.1f}, window={raw}"
