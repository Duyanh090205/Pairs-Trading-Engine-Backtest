# HMM Regime Detection for a Factor-Residual Pairs Strategy: An Opinionated Review

## TL;DR
- **Do not build a full HMM regime detector for paper trading next week.** With ~1,000 daily observations, 90 trades across 25 folds, and a strategy already failing the Deflated Sharpe screen, a Gaussian HMM will add hyperparameters, label-switching headaches, and a documented ~25-day detection latency without addressing your actual failure mode (heterogeneous bilateral spread drift). Ship the composite z-score halt first.
- **Your "Late Bull" failure is a bilateral-spread phenomenon, not a market-regime phenomenon.** Q1 2026 vol of 33–40% with average pairwise correlation collapsing from 0.24 to 0.10 IS a market-stress signature, but a simple z(vol) − z(corr) + z(dispersion) composite, gated by the top tertile of trailing 252d, will catch most of what an HMM would catch with a fraction of the engineering risk (this is the author's judgement, not a sourced figure).
- **If you still want a probabilistic regime model after 3–6 months of live paper trading, skip vanilla Gaussian HMM and use a statistical jump model (Nystrup, Lindström & Madsen 2020, *Expert Systems with Applications* 150:113307).** It is specifically designed for the "unrealistically rapid switching dynamics" failure mode HMMs exhibit on financial returns, and it co-estimates state persistence as a regularizer.

---

## 1. EXECUTIVE RECOMMENDATION (≤200 words)

**Verdict: Composite z-score now, jump model in 3–6 months — skip vanilla HMM entirely.**

Three facts dominate the decision. (1) Rydén, Teräsvirta & Åsbrink (1998, *Journal of Applied Econometrics* 13:217–244) fit 2-state Gaussian HMMs on ~1,700-day S&P 500 subseries and reported that "parameter estimates of the model sometimes vary considerably from one subseries to the next" — you have ~1,000 days. (2) Bulla, Mergner, Bulla, Sesboüé & Chesneau (2011, *Journal of Asset Management* 12:310–321) report (per the published abstract) that on 40 years of equity index data, regime switching reduces volatility on average by 41% with annualized excess returns of only 18.5 to 201.6 basis points — a modest economic gain. (3) Your strategy already fails DSR (PBO=12.3% with E[max SR]=2.05 across 50 variants); Bailey & López de Prado (2014, *JPM* 40(5):94–107) warn: "a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be."

**Concrete model spec if you nonetheless implement:** 2-state Gaussian HMM via `hmmlearn`, diagonal covariance, features = [realized_vol_60d, avg_pairwise_corr_60d, cross_sectional_dispersion_20d] each z-scored on rolling 252d, quarterly refit, label persistence by mean-vol comparison, halt entries when filtered P(stress) > 0.70 confirmed on 2 of 3 consecutive days.

---

## 2. PER-QUESTION ANSWERS

### Q1 — Feature engineering for equity regime detection

**Q1.1: Features with published evidence for pairs trading specifically.** **CONFIDENCE: MEDIUM-LOW.**
Direct published evidence for pairs-trading regime features is thin. The closest is Johnson-Skinner, Liang, Yu & Morariu (2021), "A Novel Algorithmic Trading Strategy using Hidden Markov Model for Kalman Filtering Innovations" (IEEE COMPSAC 2021, pp. 1766–1771), which uses an HMM to optimize Kalman-filtered spread thresholds. The independent replication by Wang (2023, Medium) found Sharpe 5.47 on one utility pair (AOS-DUK) but negative Sharpe on 11 of 12 cointegrated utility pairs — the headline result was overfit. The author attributes failure to "the momentum of the spread divergence" and threshold overfitting — almost exactly your "factor momentum accumulation" failure mode.

For *general equity* regime features that transfer reasonably to pairs:
- **Realized volatility**: Rydén, Teräsvirta & Åsbrink (1998) — canonical input.
- **Cross-sectional return dispersion**: Stivers (2003, *Journal of Financial Economics*), Byun (2016, *Journal of Empirical Finance* 36:162–180), Fei, Liu & Wen (2019, *Pacific-Basin Finance Journal* 58). Byun: cross-sectional dispersion improves GARCH out-of-sample. Most relevant to pairs trading: dispersion approximates the idiosyncratic-vs-systematic ratio, which directly maps to pair convergence likelihood.
- **Average pairwise correlation**: Ang & Bekaert (2002, *Review of Financial Studies* 15:1137–1187), abstract verbatim: "Correlations between international equity market returns tend to increase in highly volatile bear markets" and regimes "may be characterized by correlations and volatilities that increase in bad times." This is the exact signature you observed in Q1 2026 — except your correlation *fell* (0.24 → 0.10), which is the unusual feature of late-bull factor momentum.

**Q1.2: Levels vs changes vs z-scores vs composite indices.** **CONFIDENCE: MEDIUM.**
Z-scores against trailing 252d are the production-standard choice in the Nystrup et al. line of work (e.g., Nystrup, Kolm & Lindström 2020, *Journal of Financial Data Science* 2(3):25–39 — uses "realized intraday volatility features" standardized). Raw levels create non-stationarity that breaks Gaussian emissions. Changes (Δvol) emphasize transitions but discard the level information that distinguishes "calm" from "stress." **Recommended: z-scores on rolling 252d window for all features.**

**Q1.3: Standard lag/window structures.** **CONFIDENCE: MEDIUM.**
Common choices in published HMM finance work: 20-day (1 month), 60-day (1 quarter), 252-day (1 year) rolling windows. There is no canonical rule. Your existing 60-day choice is defensible. The HAR-RV literature (Corsi 2009) — heterogeneous mixing of 1d, 5d, 22d windows — is one principled alternative.

**Q1.4: Features at different frequencies.** **CONFIDENCE: LOW.**
Literature is largely silent on mixed-frequency HMM emissions in finance. Standard practice is to forward-fill monthly features to daily resolution. This introduces autocorrelation that violates the HMM's i.i.d.-given-state emission assumption. Practical workaround: compute everything daily on rolling windows so the frequency is unified.

**Q1.5: Stationarity / differencing.** **CONFIDENCE: MEDIUM.**
Hamilton (1989, *Econometrica* 57:357–384) used GNP *growth rates* (differenced log levels) — but that's because levels were trended. For financial features that are *already* stationary or near-stationary (vol, correlation, dispersion are bounded), differencing is unnecessary and discards information. Z-scoring on rolling windows accomplishes the "remove slow drift" goal without the differencing cost. Do NOT difference vol/corr/dispersion.

**Q1.6: Critique of candidate features.** **CONFIDENCE: MEDIUM-HIGH.**
- `realized_vol_60d`: ✅ Keep. Core regime signal.
- `avg_pairwise_corr_60d`: ✅ Keep. Ang & Bekaert (2002) explicitly identifies correlation as a regime variable. Note: Q1 2026's *falling* correlation (0.24 → 0.10) is the opposite of classic bear-regime behavior, so this feature alone won't flag your failure regime — it's a **dispersion + vol** regime, not a correlation-spike regime.
- `cross_sectional_dispersion`: ✅ Strong keep. Stivers (2003), Byun (2016). Most directly maps to pair-divergence risk.
- `daily_return`: ⚠️ Marginal. EW index return is high-noise, low-signal at daily frequency. Drop or weight low.
- `cum_var_pca_60d` (top-5 explained variance): ⚠️ Conceptually appealing (high cum_var = factor-dominated = bad for residual mean-reversion) but redundant with avg_pairwise_corr_60d (they're mathematically linked). Pick one — recommend keeping correlation, dropping cum_var, since cum_var is monthly while correlation is daily.

### Q2 — HMM model selection at ~1,000 obs

**Q2.1: Number of states K.** **CONFIDENCE: HIGH.**
With ~1,000 obs and 3–4 visible regime shifts (2022 bear, 2023 early bull, 2024 mid bull, 2025–Q1 2026 late bull stress), **K=2 is the only defensible choice**. BIC is the conventional selector (Rydén et al. 1998 used BIC), but Dupont et al. (2025, *Methods in Ecology and Evolution*) shows BIC's penalty grows slower than likelihood under misspecification — i.e., BIC tends to *overestimate* K when emissions are misspecified, which they always are for financial returns. K=3 (calm/normal/stress) sounds appealing but requires roughly 2× the data to estimate stably and introduces a "normal" state that is ill-defined and label-unstable. **Recommendation: K=2, justified by domain knowledge, BIC as a sanity check only.**

**Q2.2: Covariance type.** **CONFIDENCE: HIGH.**
With 3 features and 2 states, full covariance = 12 parameters; diagonal = 6; tied = 6 (one shared). With ~1,000 obs that's still adequate for full covariance, but features will be correlated (vol-dispersion correlation is empirically high), which destabilizes full covariance estimation. **Diagonal covariance is the standard production choice** for hmmlearn financial HMMs (default in QuantStart, hmmlearn docs, and Nystrup-line papers). Use diagonal.

**Q2.3: Initialization sensitivity.** **CONFIDENCE: HIGH.**
This is a serious concern. Hess (2009, "A Check on the Robustness of Hamilton's Markov Switching Model") found that Hamilton's (1989) two-regime GNP model had **multiple local maxima with different economic interpretations** when starting values varied — and that "when the sample period is extended, there is no longer a local maximum near the parameter set reported by Hamilton." For ~1,000-obs financial HMMs, EM/Baum-Welch will frequently land in different local optima. **Mitigation: run 20–50 random initializations, pick the one with the highest log-likelihood, AND check that the top-3 initializations agree on the state label assignment.** If they disagree, the model is unidentified for your data — fall back to the composite z-score.

**Q2.4: Bayesian HMM vs MLE.** **CONFIDENCE: MEDIUM.**
At ~1,000 obs with 6 parameters, MLE is fine — the regularization benefit of priors is small relative to implementation cost. `pomegranate` and `pymc` introduce label-switching post-processing (Stephens 2000, *JRSS-B*; Papastamoulis 2014, `label.switching` R package). For your solo-quant constraint, **stick with `hmmlearn` MLE + random restarts.** Don't add Bayesian complexity unless OOS validation explicitly demands it.

**Q2.5: Categorical vs Gaussian emissions.** **CONFIDENCE: MEDIUM.**
Discretizing into bins throws away information and creates a discretization-threshold hyperparameter. Gaussian is the published default. Bulla (2011, *Quantitative Finance* 11:459–475) shows t-distributed emissions slightly improve persistence — but the gain is modest and t-distributions add a degrees-of-freedom parameter. **Stick with Gaussian.**

### Q3 — Validation methodology for small-sample HMM

**Q3.1: Rolling-origin vs expanding window.** **CONFIDENCE: HIGH.**
Both are accepted in the literature; López de Prado (2018, *Advances in Financial Machine Learning*, Wiley) advocates rolling-origin (purged k-fold or combinatorial purged CV) explicitly to avoid the "data lake" leakage of expanding windows. For HMM specifically, expanding-window leaks the eventual "stress" label distribution to early-fold parameter estimates. **Use rolling-origin with 504-day train (~2 years), 63-day OOS (1 quarter), step forward 21 days.**

**Q3.2: Preventing overfit with ~1,000 obs and 3–4 regime shifts.** **CONFIDENCE: MEDIUM-HIGH.**
You effectively have ~3 "regime observations" — a tiny effective sample for any latent-state model. Mitigations: (a) constrain K=2; (b) constrain covariance to diagonal; (c) require persistence (min self-transition probability ≥ 0.95 or use HSMM); (d) ensemble across random restarts and report disagreement rate.

**Q3.3: Signs of overfit HMM.** **CONFIDENCE: MEDIUM-HIGH.**
- Perfect in-sample classification of historical events.
- State sequence with median run length < 5 days (Nystrup, Lindström & Madsen 2020 specifically motivate jump models because vanilla HMMs produce "unrealistically rapid switching dynamics").
- Filtered P(stress) flipping >40% on adding one new data point.
- Best two random restarts disagreeing on >10% of historical state assignments.
- BIC monotonically improving from K=2 to K=5 with no elbow.

**Q3.4: Bootstrap stability tests.** **CONFIDENCE: MEDIUM.**
Yes — block bootstrap (preserving autocorrelation) the training data, refit HMM N=100 times, compute the distribution of transition probabilities. If the 95% CI of P(stay-in-stress) spans more than ±0.10, the model is too unstable for production. Standard in the academic HMM literature; less standard in industry, but cheap to run and informative.

**Q3.5: Realistic Sharpe lift expected from HMM filter.** **CONFIDENCE: MEDIUM.**
Bulla et al. (2011) abstract: "the volatility reduces on average by 41 per cent. In addition, annualized excess returns attain 18.5 to 201.6 basis points." (Secondary sources transcribe Sharpe figures of 0.342 → 0.437–0.646 but these are not in the open abstract; treat as indicative only.) Shu, Yu & Mulvey (2024, arXiv 2402.05272v3, abstract verbatim): "The JM-informed strategy improves annualized returns by approximately 1% to 4% across different regions, leading to improved risk-adjusted return metrics." For your current dynamic net Sharpe of 0.443, an HMM filter realistically gets you to ~0.55–0.65 if everything goes well — still well below the 0.8–1.0 institutional threshold and well below your reported in-sample mean Sharpe of 0.995. **The HMM does not rescue the strategy — it modestly improves its risk-adjusted return.**

### Q4 — Look-ahead bias and data leakage

**Q4.1: Filtered vs Viterbi/smoothed.** **CONFIDENCE: HIGH.**
Confirmed: Viterbi and forward-backward (smoothed) posterior P(s_t | y_{1:T}) use future data and are forbidden for live decisions. You must use the filtered forward probability P(s_t | y_{1:t}), computed by the forward-only pass. In `hmmlearn`, this is **not** what `.predict_proba()` returns by default — `predict_proba` returns smoothed (forward-backward) posteriors. You need to either (a) implement the forward recursion yourself (Borst 2024 Medium provides a reference Python implementation), or (b) at each decision time t, refit on data [0, t] and read out P(s_t | y_{1:t}) as the last row of the forward lattice.

Other subtle leak paths:
- **Feature normalization leakage**: rolling 252d z-score is fine; full-sample z-score is leakage.
- **State labeling leakage**: see Q4.3.
- **Universe selection leakage**: your "528 liquid 2026 tickers" is point-in-time survivorship leakage. Already flagged in your context.

**Q4.2: Refit on [0,t] vs frozen model from [0,t-N].** **CONFIDENCE: MEDIUM.**
Both are used in published work. Refit-every-day on [0,t] is most realistic but introduces label-switching across refits and computational cost. Frozen model with periodic refit (monthly/quarterly) is standard. **For paper trading: train on first 504 days, freeze, walk forward 63 days OOS, then refit monthly.** This matches your existing 45-monthly-fold walk-forward.

**Q4.3: Regime label assignment.** **CONFIDENCE: HIGH.**
Critical leak path. If you assign "stress" to the state with higher empirical realized variance on the full sample, you've used out-of-sample knowledge. Correct procedure: at the end of every training window, compute mean (or median) of `realized_vol_60d` for observations assigned to each state in the training set only, label the higher-vol state "stress," freeze the mapping, and apply to OOS. Re-derive the mapping on each refit.

### Q5 — Live deployment considerations

**Q5.1: Refit cadence.** **CONFIDENCE: MEDIUM.**
Monthly is standard. Quarterly is acceptable. Avoid daily — label switching across refits will create signal flicker. Drift-triggered refit (refit when log-likelihood of recent window drops > X std below historical) is more elegant but requires another threshold; not worth the complexity for a solo quant.

**Q5.2: Label switching across refits.** **CONFIDENCE: HIGH.**
After each refit, re-derive the "stress" label by the mean-vol-of-state rule from Q4.3. Also: enforce that the prior refit's filtered P(stress) at time t-1 is approximately continuous with the new refit's filtered P(stress) at time t-1 (a "matching" step). If the new refit's stress label disagrees with the old refit's stress label on >5% of the overlapping training window, abort the refit and stick with the prior model — this is a sign of instability.

**Q5.3: Latency.** **CONFIDENCE: HIGH.**
Computational latency: forward pass for K=2, T=1000 is microseconds. Training (Baum-Welch with 20 restarts) is ~1–5 seconds. Not a constraint.

Detection latency: Nystrup, Hansen, Larsen, Madsen & Lindström (2018a, *Journal of Portfolio Management* 44(2):62–73), as cited by Shu, Yu & Mulvey (2024, arXiv 2402.05272v2), report "the median [latency] in detecting regime changes is 25 (calendar) days." **This is the killer fact for your use case** — by the time the HMM detects the Q1-2026-style regime shift, you're 25 days into the drawdown. (Bracketed [latency] is the citing author's clarifying insertion.)

**Q5.4: Failure modes / guardrails.** **CONFIDENCE: MEDIUM.**
- Missing daily observation → forward-fill with last value, flag.
- Extreme value (>10σ) → winsorize at 5σ before feeding to HMM, log the event.
- EM convergence failure → fall back to prior month's model, alert.
- Filtered P(stress) NaN or pinned at 1.0 or 0.0 for >5 consecutive days → fall back to composite z-score rule.

### Q6 — Honest comparison with simpler alternatives

| Alternative | Impl. effort | Expected Sharpe lift vs HMM | Sample requirements | Production maturity | Key citation |
|---|---|---|---|---|---|
| (1) Tertile rule (your current) | 0.5 day | HMM ≈ +0.05 to +0.10 better (author estimate) | 252d minimum | High; widely used | — (heuristic) |
| (2) **Composite z-score** | 1 day | HMM ≈ +0.05 to +0.15 better, maybe (author estimate) | 252d | High | Cross-sectional dispersion as predictor: Stivers (2003), Byun (2016) |
| (3) Change-point detection (PELT, ruptures) | 2 days | HMM ≈ comparable; CPD better at detection latency, worse at state identification | 500+ obs | Medium; Truong, Oudre & Vayatis (2020) `ruptures` library; Nystrup et al. (2016, *Journal of Asset Management* 17:361–374) applied to VIX/SPX | Truong et al. arXiv 1801.00826 |
| (4) Markov-switching regression (statsmodels) | 3 days | Essentially equivalent to HMM | Same as HMM | High; standard econometrics | Hamilton (1989) |
| (5) GARCH + threshold | 2 days | HMM ≈ +0.05 to +0.10 better (author estimate); GARCH is purely vol, misses correlation/dispersion regime | 500+ obs | Very high | Bollerslev (1986) |
| (6) Two-stage: composite z-score (daily) + jump model (monthly) | 5–7 days | Probably best long-run; +0.10 to +0.25 over composite alone (author estimate) | 1000+ obs | Emerging; Nystrup, Lindström & Madsen (2020) ESWA 150:113307 | github.com/Yizhan-Oliver-Shu/jump-models |

*All Sharpe-lift estimates in this table are the author's order-of-magnitude judgements based on Bulla et al. (2011) and Shu et al. (2024) headline numbers, not direct apples-to-apples published comparisons.*

**Recommendation rank for YOUR situation (in order):**
1. **Composite z-score (option 2)** — ship next week.
2. **Statistical jump model (option 6 second stage)** — evaluate in 3–6 months.
3. PELT change-point as supplementary (not primary) signal.
4. Gaussian HMM (vanilla) — skip.

### Q7 — Published references and case studies

**Foundational HMM in finance:** **CONFIDENCE: HIGH.**
- Hamilton, J.D. (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle," *Econometrica* 57:357–384.
- Hamilton, J.D. (1994) *Time Series Analysis*, Princeton — Ch. 22 on regime switching.
- Ang, A. & Bekaert, G. (2002) "International Asset Allocation with Regime Shifts," *Review of Financial Studies* 15(4):1137–1187. Abstract verbatim: "International diversification is still valuable with regime changes... The costs of ignoring the regimes are small for all-equity portfolios but increase when a conditionally risk-free asset can be held" — directly relevant: regime models help most when you have a CASH option (you do — "halt entries").
- Rydén, T., Teräsvirta, T. & Åsbrink, S. (1998) "Stylized facts of daily return series and the hidden Markov model," *Journal of Applied Econometrics* 13(3):217–244 — the canonical small-sample-instability reference.

**HMM for pairs trading specifically:** **CONFIDENCE: MEDIUM-LOW.**
- Johnson-Skinner, Liang, Yu & Morariu (2021), IEEE COMPSAC 2021, pp. 1766–1771 — Kalman + HMM pairs.
- Wang, K. (2023) Medium replication — found Sharpe 5.47 on one pair (AOS-DUK), negative Sharpe on 11 of 12 cointegrated utility pairs — the replication is itself an important data point: **HMM-pairs results in the published literature are highly overfit to specific pair selection.** Author quote: "the majority of the annualized Sharpe Ratio is negative, indicating that it is not effective in real-world trading scenarios."
- A separate "Pairs Trading Strategy for A and H Shares Based on Kalman-HMM Approach" (2021, ResearchGate) — claims "the holding yield increased from 1.6% to 16.2% and the maximum pullback reduced to 0.02%." Treat with caution (Chinese A/H equity, small dataset, single replication).
- The published peer-reviewed pairs-trading-HMM literature is genuinely thin. Most "HMM + pairs trading" work is grey literature (Medium, GitHub, course projects).

**HMM with small samples / regularization:** **CONFIDENCE: HIGH.**
- Nystrup, P., Lindström, E. & Madsen, H. (2020) "Learning hidden Markov models with persistent states by penalizing jumps," *Expert Systems with Applications* 150:113307. Quote: "When the model is misspecified or misestimated, however, it often leads to unrealistically rapid switching dynamics."
- Nystrup, P., Kolm, P.N. & Lindström, E. (2020) "Greedy Online Classification of Persistent Market States Using Realized Intraday Volatility Features," *Journal of Financial Data Science* 2(3):25–39. Quote: "in most settings our new classifier remarkably obtains a higher accuracy than the correctly specified maximum likelihood estimator."
- Bulla, J. & Bulla, I. (2006) "Stylized facts of financial time series and hidden semi-Markov models," *Computational Statistics & Data Analysis* 51(4):2192–2209 — HSMMs improve regime persistence over geometric-sojourn HMMs.
- Bulla, J. (2011) "Hidden Markov models with t components," *Quantitative Finance* 11(3):459–475.

**Production deployment case studies:** **CONFIDENCE: MEDIUM-LOW.**
- Bulla, J., Mergner, S., Bulla, I., Sesboüé, A. & Chesneau, C. (2011) "Markov-switching asset allocation: Do profitable strategies exist?", *Journal of Asset Management* 12(5):310–321. Abstract: "the volatility reduces on average by 41 per cent. In addition, annualized excess returns attain 18.5 to 201.6 basis points."
- Nystrup, P., Hansen, B.W., Madsen, H. & Lindström, E. (2015) "Regime-Based Versus Static Asset Allocation," *Journal of Portfolio Management* 42(1):103–109.
- Mulvey, J.M. & Liu, H. (2016) "Identifying Economic Regimes: Reducing Downside Risks for University Endowments and Foundations," *Journal of Portfolio Management*.
- Shu, Y., Yu, X. & Mulvey, J.M. (2024) "Downside risk reduction using regime-switching signals: a statistical jump model approach," *Journal of Asset Management* / arXiv 2402.05272 — JM beats HMM; "The JM-informed strategy improves annualized returns by approximately 1% to 4% across different regions."

**Critiques / failure cases:** **CONFIDENCE: HIGH.**
- Nystrup et al. (2020a, 2020b) — entire jump-models research program is motivated by HMM's instability on financial returns.
- Pomorski, P. & Gorse, D. (2023, arXiv 2310.04536) "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes" — key critique: "The HMM however suffers from an inability to predict regime switches without experiencing the onset of the new regime: the HMM can in effect only predict continuations of already-changed regimes."
- Hess, M.K. (2009) "A Check on the Robustness of Hamilton's Markov Switching Model" — Hamilton's seminal result fails to replicate on extended data; multiple local maxima.
- Asness, C. (2016) "The Siren Song of Factor Timing," *Journal of Portfolio Management* 42(5) — broader skepticism on regime-based factor timing.

**López de Prado on DSR and overfitting:** **CONFIDENCE: HIGH.**
- Bailey, D.H. & López de Prado, M. (2014) "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* 40(5):94–107. Quote directly relevant to your DSR p-value = 0.000: "Without this information [number of trials], it is impossible to assess the relevance of a backtest. Put bluntly, a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be."
- Bailey, Borwein, López de Prado & Zhu (2014) "Pseudo-Mathematics and Financial Charlatanism," *Notices of the AMS* 61(5):458–471.
- López de Prado, M. (2018) *Advances in Financial Machine Learning*, Wiley — proposes ONC algorithm to estimate effective number of independent trials. Directly relevant: with your 50 strategy variants, the effective N may be 10–20, but adding HMM hyperparameters (K, covariance type, n_restarts, refit cadence, halt threshold) inflates N back toward 50+, eroding any DSR gain.

---

## 3. IMPLEMENTATION CHECKLIST

**Phase 0 — Composite z-score halt (THIS WEEK, paper trading):**
1. [0.5 day] Compute features daily: `realized_vol_60d`, `avg_pairwise_corr_60d` (Spearman, 60d EW), `cross_sectional_dispersion_20d` (cross-sectional std of daily returns averaged over 20d).
2. [0.5 day] Compute rolling 252d z-scores of each feature.
3. [0.5 day] Composite: `stress_z = z(vol) - z(corr) + z(dispersion)`. Halt new entries when `stress_z > q_67` (top tertile of trailing 252d) AND vol level > median trailing 252d (double-threshold to avoid halting in benign high-dispersion regimes).
4. [1 day] Plug into existing walk-forward; recompute 45 monthly folds.
5. [0.5 day] Sanity check: regime mask correctly flags 2022 bear AND Q1 2026 late-bull stress. If not, recalibrate composite weights.

**Phase 1 — Paper trading (Weeks 1–8):**
6. [ongoing] Run the composite-z-halt strategy live-paper for 6–8 weeks.
7. [1 day] Compute live OOS Sharpe, halt frequency, and percentage of halts that *would* have been losing trades. Pass criterion: ≥ 60% of halts correspond to losing trade counterfactuals.

**Phase 2 — If composite z-score is insufficient (Month 3+):**
8. [2 days] Implement vanilla 2-state Gaussian HMM in `hmmlearn` with diagonal covariance, 20 random restarts, monthly refit. Use as a benchmark, NOT for live decisions.
9. [3 days] Implement statistical jump model (Nystrup, Lindström & Madsen 2020) — code at github.com/Yizhan-Oliver-Shu/jump-models.
10. [3 days] OOS validation: rolling-origin walk-forward, 504-day train / 63-day OOS / 21-day step, compare composite-z vs HMM vs JM on Sharpe lift, halt accuracy, detection latency.
11. [2 days] If JM wins by >+0.10 Sharpe and passes the bootstrap stability test, promote to production.

**Phase 3 — Production hardening (Month 4–6):**
12. [3 days] Build guardrails (Q5.4), monitoring dashboard, alert system.
13. [2 days] Recompute DSR with all hyperparameters counted as trials; require DSR p-value < 0.05.

**Total estimated effort:** Phase 0+1 ≈ 4 days. Phase 2+3 ≈ 15 days. Skip Phase 2+3 unless Phase 1 OOS results justify it.

---

## 4. VALIDATION PLAN

**OOS test procedure (rolling-origin walk-forward):**
- Train window: 504 days (~2 years). Justification: covers at least one regime cycle.
- OOS window: 63 days (~1 quarter).
- Step: 21 days.
- Total folds available on your 1,000-day dataset: ~20 folds (less than your existing 45 monthly folds because of the larger train window; this is the price of stable HMM fit).
- Refit cadence: every fold (~monthly).

**Metrics to compute per fold and aggregate:**
- Sharpe (net of transaction costs, your existing 50bps assumption).
- Halt accuracy: fraction of halted days where the strategy (counterfactually run) would have lost money.
- Detection latency: days between true regime start (defined ex post by realized vol crossing top quintile) and HMM flagging.
- State persistence: median run length in OOS.
- Bootstrap stability: block-bootstrap training data 100×, compute std of P(stay-in-stress); require std < 0.05.

**Pass criteria for promoting from paper to production:**
- OOS net Sharpe ≥ 0.7 (vs your current 0.443) AND
- OOS Late-Bull-style fold Sharpe ≥ 0 (currently −0.75) AND
- Halt accuracy ≥ 55% (better than coin flip on a meaningful margin) AND
- Detection latency median ≤ 15 trading days AND
- Deflated Sharpe p-value < 0.10, counting ALL hyperparameter choices in N AND
- PBO ≤ 20%.

**Fail criteria (any one triggers reversion to composite z-score):**
- Two consecutive refits disagree on >5% of overlapping training labels.
- OOS state median run length < 10 days (rapid-switching pathology, per Nystrup, Lindström & Madsen 2020).
- Filtered P(stress) NaN or pegged at 0/1 for >5 consecutive days.

---

## 5. RISK REGISTER

1. **Failure-mode mismatch (HIGHEST risk).** Your failure is heterogeneous bilateral spread drift, not a market-wide regime. A market-regime HMM cannot see pair-specific drift. Mitigation: complement the regime halt with a per-pair OU-half-life monitor that closes positions whose realized spread autocorrelation > 0.95 over the last 20 days (signals "drift, not mean-reversion"). The HMM is a complement, not a substitute, for pair-level risk management.

2. **Detection latency exceeds payoff window.** Nystrup et al. (2018a) document a 25-calendar-day median detection latency. Your monthly holding periods mean by the time the HMM detects Q1-2026-style stress, the bad month is half over. Mitigation: use HMM output as a position-size dampener (reduce size to 50% as P(stress) climbs from 0.5 to 0.7), not as a binary halt — captures partial signal before full conviction.

3. **Label switching destroys live signal.** Across monthly refits, the "stress" state can swap with the "calm" state. Mitigation: re-derive labels from training mean-vol-of-state (Q4.3); enforce continuity check across refits (abort refit if >5% label disagreement on overlap).

4. **DSR penalty from new hyperparameters cancels signal lift.** You already failed DSR with 50 variants. Adding HMM hyperparameters (K, covariance type, refit cadence, halt threshold, restart count, feature set) reverses progress. Per Bailey & López de Prado (2014): "a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be." Mitigation: fix all hyperparameters ex ante from this report; do NOT grid-search anything. Document the choices and pre-register them before running OOS.

5. **Spurious confidence from in-sample fitting.** ~1,000 obs and ~3 visible regime transitions is too few to fit a state-space model reliably (Rydén et al. 1998 showed parameter instability even at 1,700 obs). Mitigation: bootstrap stability test (Q3.4) as a hard gate; if it fails, kill the HMM and ship the composite z-score.

---

## 6. CAVEATS

- The published HMM-for-pairs-trading literature is unusually thin. Most evidence is from HMM-for-regime-detection-in-equity-indices, which I extrapolate to your pairs context with appropriate caution. The Johnson-Skinner et al. (2021) IEEE paper is the closest direct precedent and its independent replication (Wang 2023) found overfitting.
- Bulla et al. (2011) headline figures are taken verbatim from the published abstract (41% vol reduction; 18.5–201.6 bp excess return). Sharpe figures of 0.342 → 0.437–0.646 reported in some secondary literature transcriptions could not be confirmed from open sources; treat as directional only.
- The Nystrup et al. (2020) ESWA paper's specific simulation-accuracy tables sit behind Elsevier's paywall and could not be verified verbatim. The qualitative claim ("higher accuracy than MLE") is corroborated by multiple citing papers.
- The "25-day detection latency" figure is from Nystrup et al. (2018a, *JPM* 44(2):62–73), as cited by Shu, Yu & Mulvey (2024, arXiv 2402.05272). The original JPM paper is paywalled; the bracketed [latency] in the quoted text is the citing author's clarifying insertion.
- All "expected Sharpe lift" estimates in the Q6 comparison table are the author's order-of-magnitude judgements, NOT directly published apples-to-apples comparisons. Your actual lift depends critically on whether the Q1 2026 failure mode recurs out-of-sample.
- No paper gives a clean "N ≥ X observations" threshold for fitting a 2-state Gaussian HMM. The closest evidence is Rydén et al. (1998) showing instability at ~1,700 obs. This is itself a finding: there is no clean academic rule of thumb, which itself argues for caution at ~1,000 obs.
- The recommendation to use a statistical jump model in Phase 2 is based on a small but consistent academic literature from the Nystrup/Kolm/Mulvey group at DTU/NYU/Princeton; it is not yet a widely-deployed industry standard. Solo implementation will rely on the open-source github.com/Yizhan-Oliver-Shu/jump-models package, which has limited production hardening.