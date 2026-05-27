# Deep research brief — HMM regime detector for pairs trading

## Context

I'm building a **regime detection module** for a daily-frequency statistical
arbitrage strategy (factor-residual cointegration pairs trading on S&P 500
constituents). The strategy works in calm/normal regimes but loses money in
high-vol / low-correlation regimes (specifically observed in Q1 2026:
realized vol 33–40% annualized, avg pairwise correlation collapsed from 0.24
to 0.10).

Goal: build a **Hidden Markov Model (HMM)** that classifies each trading day
into a regime (e.g., calm / stress) and outputs `P(regime = stress | data
up to today)`. Strategy uses this probability to halt new entries when
P(stress) exceeds a threshold.

## Constraints

- **Universe**: 528 liquid S&P 500 tickers (no point-in-time, end-of-period membership)
- **Historical data**: ~4 years daily (2022-01 to 2026-03), ~1,000 trading days total
- **Live deployment target**: paper trading next week, full prod in months
- **Implementation skill**: solo quant, comfortable with Python/numpy/pandas, moderate stats background
- **Library preference**: `hmmlearn` (already in env), but open to alternatives if justified
- **Compute**: single machine, no GPU needed
- **Refit cadence**: monthly or quarterly acceptable; not online/streaming

## Research questions (in priority order)

### Q1 — Feature engineering for regime detection in equities
1. What features have **published evidence** of regime predictive power for
   equity pairs trading specifically? (Not just generic equity regime.)
2. Should features be: levels (e.g., realized vol), changes (Δvol),
   z-scores (vs trailing 252d), or composite indices?
3. Are there standard **lag structures** (e.g., 5d, 20d, 60d windows)?
4. How to handle features at **different frequencies** (daily vol vs monthly
   PCA cum_var)? Resample, lag-pad, or skip?
5. **Stationarity**: do features need differencing before HMM? Most
   `hmmlearn` examples don't differentiate, but Hamilton 1989 strongly
   recommends.
6. Specific to my problem: candidate features I'm considering — please
   critique:
   - `realized_vol_60d` (rolling 60d EW index vol, annualized)
   - `avg_pairwise_corr_60d` (rolling avg pairwise correlation)
   - `cross_sectional_dispersion` (cross-sectional std of returns)
   - `daily_return` (EW index)
   - `cum_var_pca_60d` (top-5 PCA factor explained variance)

### Q2 — HMM model selection
1. **Number of states (K)**: how to choose? BIC, AIC, cross-validated
   log-likelihood, gap statistic, or domain knowledge (e.g., K=2 calm/stress
   vs K=3 calm/normal/stress)?
2. **Covariance type**: full, diagonal, tied, spherical — what's standard
   for financial regime HMM with ~1000 obs?
3. **Initialization**: random vs k-means vs Bayesian prior. Does
   initialization sensitivity make HMM unreliable for live use?
4. **Bayesian HMM** (e.g., `pomegranate`, `pymc`) vs MLE (`hmmlearn`): is
   the regularization worth the implementation complexity at my sample size?
5. **Categorical vs Gaussian emissions**: should I discretize features into
   bins (low/med/high) and use Categorical HMM? Pros/cons?

### Q3 — Validation methodology for small-sample HMM
1. Standard train/test split is wrong for time-series — what's the
   accepted methodology? **Rolling-origin cross-validation** or **expanding
   window**?
2. With ~1000 daily observations and ~3-4 regime shifts visible, how do I
   prevent overfitting to specific past regimes?
3. **How to detect overfit HMM**: signs to watch for (e.g., perfect
   in-sample classification, sudden state flips on tiny perturbations)?
4. **Stability tests**: should I bootstrap-fit the HMM N times and check
   transition matrix stability?
5. What's a **realistic Sharpe lift** to expect from HMM filter over a
   simple tertile/threshold rule? Published estimates?

### Q4 — Look-ahead bias and data leakage
1. The **Viterbi smoothed regime** assignment uses future data — must use
   **filtered (forward-pass) probability** for live decisions. Are there
   other subtle leak paths?
2. **Forward-pass with rolling refit**: at decision time t, do I refit HMM
   on data [0, t] or use a model trained on [0, t-N months]? What's
   standard?
3. **Regime labels**: if I label "stress" as states with high vol, I'm
   bringing in posterior knowledge. Should label assignment happen
   post-hoc on training data only, then frozen?

### Q5 — Live deployment considerations
1. **Refit cadence**: monthly, quarterly, or only when transition matrix
   drifts beyond threshold?
2. **Cold-start problem**: when refitting, does HMM state numbering change
   between fits? How to maintain consistent "stress" label across refits?
   (Label switching problem.)
3. **Latency**: at decision time t, I need P(stress | features up to t).
   What's the computational cost? Acceptable for next-day decisions.
4. **Failure modes**: how does HMM degrade if input features have
   missing/extreme values? What guardrails?

### Q6 — Honest comparison with simpler alternatives
For each alternative below, please give:
- Implementation effort
- Expected lift vs HMM
- Sample size requirements
- Production maturity

Alternatives:
1. **Threshold tertile rule** (current): skip month if vol in top tertile
   of trailing 12m
2. **Composite stress z-score**: z(vol) − z(corr) + z(dispersion);
   threshold-based halt
3. **Bayesian change-point detection** (e.g., `ruptures` library, PELT
   algorithm)
4. **Markov-switching regression** (statsmodels' MarkovRegression /
   MarkovAutoregression)
5. **GARCH-based vol forecasting** + threshold
6. **Two-stage**: HMM-lite (composite score) for high-frequency monitor,
   HMM (Bayesian) for monthly refit

### Q7 — Published references and case studies
Please cite (with year and journal/venue where possible):
1. Foundational HMM in finance: Hamilton 1989, Ang & Bekaert 2002
2. HMM for pairs trading / stat arb specifically (most relevant)
3. HMM with small samples (regularization techniques)
4. Production deployment case studies (industry blogs, papers)
5. Critiques / failure cases (when HMM didn't work)

## Deliverable format

Please return:

1. **Executive recommendation** (≤200 words): given my constraints, should
   I implement HMM at all, or stick with the composite z-score rule?
   Concrete model spec if HMM recommended.

2. **Per-question answers** with citations where applicable. Mark unknowns
   honestly — "literature is silent on this" is acceptable.

3. **Implementation checklist**: ordered TODO list with effort estimates.

4. **Validation plan**: specific OOS test procedure (e.g., walk-forward
   with rolling origin) and pass criteria.

5. **Risk register**: top 5 ways this can fail in production + mitigations.

## What I want to AVOID

- **Hype**: "ML for regime detection is state of the art" without
  evidence vs simpler methods
- **Overfit**: anything that requires tuning 10+ hyperparams on 1000 obs
- **Black boxes**: I need to explain decisions to non-technical stakeholders
- **Hand-wavy validation**: "we backtested and it worked"
- **Generic ML papers**: looking for **finance-specific**, regime-detection
  specific guidance

## Honest disclosure I want from researcher

State your confidence in conclusions:
- HIGH: well-established in literature, multiple sources
- MEDIUM: some sources, but contested or context-dependent
- LOW: speculative or your own inference
