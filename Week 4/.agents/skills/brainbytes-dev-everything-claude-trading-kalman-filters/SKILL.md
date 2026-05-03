# Kalman Filters

name: kalman-filters
description: Kalman filter applications in finance — dynamic beta estimation, pairs trading, state-space models. Use for dynamic parameter estimation.

## When to Activate

- User needs dynamically estimated parameters (betas, hedge ratios) that evolve over time
- Building pairs trading strategies with adaptive hedge ratios
- Modeling unobserved state variables (e.g., true value, latent factors)
- Tracking time-varying relationships between assets
- Smoothing noisy financial signals without excessive lag

## First Questions

1. What relationship are you trying to track (beta, hedge ratio, spread, trend)?
2. How fast do you expect the parameters to change (slowly over months, or rapidly)?
3. What is your data frequency (tick, minute, daily)?
4. Are you using this for signal generation, risk management, or both?
5. Do you need online (real-time, streaming) updates or batch estimation?

## Core Concepts

### State-Space Representation

The Kalman filter operates on a state-space model with two equations:

```
State equation:     x_t = F * x_{t-1} + w_t,    w_t ~ N(0, Q)
Observation equation: y_t = H * x_t + v_t,      v_t ~ N(0, R)

x_t: unobserved state vector (what we want to estimate)
y_t: observed data
F:   state transition matrix (how states evolve)
H:   observation matrix (how states map to observations)
Q:   state noise covariance (how fast states change)
R:   observation noise covariance (measurement error)
```

The filter recursively estimates x_t given all observations up to time t — no look-ahead bias by construction.

### Kalman Filter Algorithm

```
--- Predict ---
x_hat_t|t-1 = F * x_hat_{t-1|t-1}           (predicted state)
P_t|t-1 = F * P_{t-1|t-1} * F' + Q          (predicted covariance)

--- Update ---
innovation: e_t = y_t - H * x_hat_t|t-1
innovation covariance: S_t = H * P_t|t-1 * H' + R
Kalman gain: K_t = P_t|t-1 * H' * S_t^(-1)
x_hat_t|t = x_hat_t|t-1 + K_t * e_t         (updated state)
P_t|t = (I - K_t * H) * P_t|t-1             (updated covariance)
```

The Kalman gain K_t controls how much weight to give new observations vs the prior estimate. When Q is large relative to R, the filter is responsive (trusts data). When Q is small, the filter is smooth (trusts the model).

### Why Kalman Filters in Finance

- **No look-ahead bias:** Filter only uses past and current observations
- **Adaptive:** Captures time-varying relationships naturally
- **Optimal:** Best linear unbiased estimator under Gaussian assumptions
- **Recursive:** Efficient for real-time applications
- **Uncertainty quantification:** Provides confidence intervals around estimates

## Detailed Methodology

### Dynamic Beta Estimation

Track the time-varying beta of an asset relative to a benchmark:

```
State:       beta_t = beta_{t-1} + w_t,    w_t ~ N(0, Q)
Observation: r_asset_t = alpha_t + beta_t * r_benchmark_t + v_t

Setup:
  x_t = [alpha_t, beta_t]'
  F = I_2 (random walk for both alpha and beta)
  H_t = [1, r_benchmark_t] (time-varying observation matrix)
  Q = diag(q_alpha, q_beta) — tuning parameters
  R = sigma_v^2 — observation noise

Typical Q values:
  q_beta = 1e-5 to 1e-3 (controls beta smoothness)
  Smaller Q = smoother beta, larger Q = more responsive
```

**Advantages over rolling OLS:**
- No arbitrary window length choice
- No abrupt changes when old data drops out of window
- Naturally weights recent data more when relationships change

### Pairs Trading with Kalman Filter

The Kalman filter dynamically estimates the hedge ratio in a pairs trade:

```
Model: y1_t = beta_t * y2_t + alpha_t + epsilon_t

State: [alpha_t, beta_t] follows a random walk
Observation: y1_t with H_t = [1, y2_t]

Trading signal:
  spread_t = y1_t - beta_t_estimated * y2_t - alpha_t_estimated
  z_score_t = spread_t / sqrt(S_t)  (normalized by innovation variance)

Entry: |z_score| > 2.0
Exit:  |z_score| < 0.5 or sign reversal
Stop:  |z_score| > 4.0 (relationship may have broken)
```

The Kalman filter hedge ratio adapts to structural changes in the relationship, preventing the "cointegration breakdown" problem that plagues static approaches.

### Q and R Calibration

The most critical and difficult aspect of Kalman filter implementation:

**Method 1: Maximum Likelihood Estimation (MLE)**
- Treat Q and R as parameters, maximize the log-likelihood of innovations
- log L = -0.5 * sum(log|S_t| + e_t' * S_t^{-1} * e_t)
- Use numerical optimization (L-BFGS-B, Nelder-Mead)
- Best theoretically but can overfit on short samples

**Method 2: EM Algorithm**
- E-step: Run Kalman smoother given current Q, R
- M-step: Update Q, R from smoothed state estimates
- Iterate until convergence
- More stable than direct MLE optimization

**Method 3: Grid Search / Cross-Validation**
- Define a grid of Q/R values
- For each combination, run the filter and evaluate out-of-sample
- Select parameters that maximize OOS trading performance
- Most practical for trading applications

**Method 4: Heuristic**
- Set R from observed residual variance of static regression
- Set Q = delta * R where delta = 1e-4 to 1e-2
- Delta controls responsiveness: start with 1e-3 and adjust

### Kalman Smoother

The smoother uses the full dataset (past AND future observations) to estimate states — useful for analysis but NOT for trading (look-ahead bias).

```
Smoother runs backward after the forward filter pass:
x_hat_t|T: optimal state estimate given ALL data
P_t|T: smoothed covariance

Use cases:
- Historical regime analysis
- Parameter calibration
- Research and hypothesis testing
NEVER for real-time trading signals
```

### Unobserved Component Models

Model a price series as the sum of unobserved components:

```
Price_t = Trend_t + Cycle_t + Noise_t

Each component has its own state equation:
  Trend_t = Trend_{t-1} + drift_{t-1} + w_trend_t     (local linear trend)
  drift_t = drift_{t-1} + w_drift_t
  Cycle_t = rho * cos(lambda) * Cycle_{t-1} + rho * sin(lambda) * Cycle*_{t-1} + w_cycle_t

Applications:
- Extract trend for trend-following strategies
- Isolate cyclical component for mean-reversion
- Filter noise for cleaner signals
```

## Templates / Examples

### Dynamic Beta Implementation Template

```
Initialization:
  x_0 = [0, 1]  (alpha=0, beta=1 as starting estimates)
  P_0 = diag(1, 1)  (high initial uncertainty)
  Q = diag(1e-6, 1e-4)  (alpha changes slowly, beta changes moderately)
  R = var(OLS_residuals)  (observation noise from static model)

For each time step t:
  1. Predict: x_pred = F * x_prev, P_pred = F * P_prev * F' + Q
  2. Form H_t = [1, r_benchmark_t]
  3. Innovation: e_t = r_asset_t - H_t * x_pred
  4. S_t = H_t * P_pred * H_t' + R
  5. K_t = P_pred * H_t' / S_t
  6. Update: x_t = x_pred + K_t * e_t
  7. P_t = (I - K_t * H_t) * P_pred
  8. Store: beta_t = x_t[1], alpha_t = x_t[0], stderr_beta = sqrt(P_t[1,1])

Output:
  Time series of beta_t with confidence bands: beta_t +/- 2*stderr_beta
```

### Kalman Filter Diagnostic Checklist

```
[ ] Innovation sequence e_t is approximately white noise (Ljung-Box test)
[ ] Innovation variance S_t is stable (not trending up/down)
[ ] Normalized innovations e_t/sqrt(S_t) are approximately N(0,1)
[ ] Filter is not diverging (P_t not growing unboundedly)
[ ] Q/R ratio produces appropriate responsiveness for the application
[ ] Comparison with rolling OLS shows smoother, more stable estimates
[ ] Out-of-sample prediction error is reasonable
[ ] State estimates are economically plausible (beta in reasonable range)
```

### Pairs Trading Kalman Report

```
Pair: [Asset A] vs [Asset B]
Period: [Start] to [End]
Q_beta: 1e-4 | R: calibrated via MLE

Dynamic Hedge Ratio:
  Mean:    1.25
  Std:     0.15
  Range:   [0.92, 1.58]

Spread Statistics:
  Mean innovation variance (sqrt(S)):  $0.85
  Z-score range: [-3.2, +3.8]
  Number of round-trip trades: 42
  Average holding period: 8 days

Performance:
  Sharpe ratio: 1.85
  Win rate: 68%
  Avg P&L per trade: $1,250
  Max drawdown: -4.2%

vs Static OLS Hedge:
  Sharpe improvement: +0.45
  Fewer stop-outs: 3 vs 8
```

## Quality Gate

Before deploying Kalman filter models in production:

- [ ] Q and R calibrated using MLE, EM, or validated via out-of-sample performance
- [ ] Innovation diagnostics pass (white noise, approximately normal, stable variance)
- [ ] Filter uses only causal (filtering) estimates, not smoothed estimates, for trading
- [ ] State estimates are economically plausible and bounded
- [ ] Comparison with simpler methods (rolling regression) shows meaningful improvement
- [ ] Sensitivity analysis: small changes in Q/R do not drastically change results
- [ ] Numerical stability verified (use Joseph form for P update if needed)
- [ ] Initialization does not unduly influence results after burn-in period
- [ ] Re-calibration schedule defined for Q and R parameters
- [ ] Extreme state values trigger alerts or model reset procedures
