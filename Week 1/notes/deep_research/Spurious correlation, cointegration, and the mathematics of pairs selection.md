# Spurious correlation, cointegration, and the mathematics of pairs selection

**Statistical cointegration between two asset prices is meaningless without an underlying economic mechanism — and the mathematics proves exactly why.** When two non-stationary time series are regressed against each other, standard OLS produces misleadingly high R² values and t-statistics that actually *diverge* with more data, guaranteeing false significance. This report provides the complete mathematical, statistical, visual, and economic framework for distinguishing genuine cointegrated pairs from spurious artifacts — the kind of framework that would reject "butter production in Bangladesh predicts the S&P 500" while validating Pepsi–Coca-Cola. The stakes are real: scanning 500 stocks yields 124,750 possible pairs, and at α = 0.05, over **6,000 pairs** will appear "significant" by pure chance. What follows is the rigorous toolkit for a Pairs Selection Report that can withstand an AI audit trap exercise on spurious correlation.

---

## 1. The statistical foundations: from correlation to cointegration

### Pearson correlation and its dangerous limitation

The Pearson correlation coefficient measures the strength of the *linear* relationship between two variables:

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \cdot \sum_i (y_i - \bar{y})^2}}$$

This is equivalent to the ratio of covariance to the product of standard deviations: **ρ = Cov(X,Y) / (σ_X · σ_Y)**. While covariance is scale-dependent (making cross-pair comparisons impossible), correlation normalizes it to the bounded range [−1, +1]. The critical limitation for finance: Pearson correlation only captures linear relationships, is sensitive to outliers, and — most dangerously — **produces spurious values when applied to non-stationary price series**. Correlation should be computed on *returns* (stationary), never on raw prices (non-stationary). As the quantitative finance maxim states: "Correlation is a short-term relationship between returns; cointegration is a long-term relationship between prices."

### Stationarity: the I(0) vs. I(1) distinction that changes everything

A time series is **weakly stationary** if its mean, variance, and autocovariance structure are all constant over time. A series is **integrated of order d**, written **I(d)**, if it requires differencing d times to become stationary. An **I(0)** process is stationary and mean-reverting — shocks are transitory. An **I(1)** process contains a unit root (the canonical random walk Y_t = Y_{t-1} + ε_t) where shocks are permanent and variance grows linearly with time. Most financial price series are I(1); their returns are I(0).

This distinction is not academic. In 1974, **Granger and Newbold** demonstrated through Monte Carlo simulation that regressing one independent random walk on another produces R² values averaging **0.77** and significant t-statistics approximately **75%** of the time — despite zero true relationship. Their rule of thumb remains definitive: **if R² exceeds the Durbin-Watson statistic, the regression is almost certainly spurious**. Peter Phillips (1986) then proved this formally: the t-statistic in a spurious regression diverges at rate √T, meaning *more data makes the problem worse, not better*.

### The Augmented Dickey-Fuller test: detecting unit roots

The ADF test is the gatekeeper for cointegration analysis. Starting from the AR(1) process Y_t = φY_{t-1} + ε_t, subtract Y_{t-1} from both sides:

**ΔY_t = α + βt + γY_{t-1} + Σ_{j=1}^{p} δ_j ΔY_{t-j} + ε_t**

where γ = φ − 1. The null hypothesis is **H₀: γ = 0** (unit root exists, series is non-stationary); the alternative is **H₁: γ < 0** (series is stationary). The test statistic τ = γ̂ / SE(γ̂) follows a non-standard Dickey-Fuller distribution, with approximate critical values of **−3.51** (1%), **−2.89** (5%), and **−2.58** (10%) for T ≈ 100 with intercept and no trend. Lag length p is selected by minimizing BIC or AIC; too few lags leave residual autocorrelation, too many destroy power.

### The Engle-Granger two-step cointegration test

Two I(1) series Y_t and X_t are **cointegrated** if there exists a linear combination Y_t − βX_t = ε_t that is I(0). The Engle-Granger procedure (Engle & Granger, 1987, *Econometrica*) formalizes this in two steps. **Step 1**: Regress Y_t = α + βX_t + ε̂_t by OLS to obtain the cointegrating coefficient β̂ (the hedge ratio) and residuals ε̂_t. The OLS estimator is **super-consistent**, converging at rate T rather than √T. **Step 2**: Apply the ADF test to ε̂_t. If residuals are stationary (I(0)), the series are cointegrated; if they contain a unit root, the regression is spurious.

Critically, standard ADF critical values **do not apply** to estimated residuals. The Engle-Granger critical values are more stringent (approximately **−3.90** at 1%, **−3.34** at 5% for two variables) because the OLS step optimizes residuals to appear stationary. The test's limitations include sensitivity to variable ordering and restriction to a single cointegrating vector.

### The Johansen test: a more powerful multivariate alternative

The Johansen procedure (1988, 1991) embeds cointegration testing within a Vector Error Correction Model (VECM):

**ΔY_t = ΠY_{t-1} + Σ_{i=1}^{p-1} Γ_i ΔY_{t-i} + ΦD_t + ε_t**

where the **rank r of the matrix Π** determines the cointegration structure. If r = 0, no cointegration exists. If 0 < r < n, there are r cointegrating vectors, and Π decomposes as **Π = αβ'**, where β contains the cointegrating vectors and α contains the speed-of-adjustment coefficients. The **trace test** (λ_trace = −T Σ ln(1 − λ̂_i)) tests H₀: r ≤ r₀ against H₁: r > r₀, while the **maximum eigenvalue test** (λ_max = −T ln(1 − λ̂_{r₀+1})) tests H₀: r = r₀ against H₁: r = r₀ + 1. Sequential testing from r = 0 upward identifies the cointegration rank. The Johansen test handles multiple cointegrating vectors, avoids the asymmetry problem of Engle-Granger, and uses full-information maximum likelihood estimation.

### Error Correction Models formalize the equilibrium mechanism

The **Granger Representation Theorem** (Engle & Granger, 1987) states that if two I(1) variables are cointegrated, they necessarily have an Error Correction Model representation:

**ΔY_t = α(Y_{t-1} − βX_{t-1}) + Σ γ_i ΔY_{t-i} + Σ δ_j ΔX_{t-j} + ε_t**

The term (Y_{t-1} − βX_{t-1}) is the error correction term — the lagged disequilibrium. The coefficient **α must be negative** for mean reversion: when the spread is above equilibrium, the negative α pulls Y back down. The magnitude |α| determines the speed of adjustment. This formalization connects the statistical concept of cointegration directly to the economic concept of equilibrium restoration.

### Spread construction and the half-life of mean reversion

The spread **S_t = Y_t − β̂·X_t** should be stationary for a valid pair, oscillating around its mean. Trading signals are generated via z-scores: z_t = (S_t − μ̂_S) / σ̂_S, with positions entered at ±2σ and exited near zero.

The **half-life** of mean reversion — how many periods until a deviation from the mean is expected to halve — is estimated via the Ornstein-Uhlenbeck process discretization **ΔS_t = λS_{t-1} + μ* + ε_t**. Running OLS to estimate λ̂ (which must be negative), the half-life is:

**τ_{1/2} = −ln(2) / ln(1 + λ̂) ≈ −ln(2) / λ̂**

A half-life of **5–60 days** is the practical sweet spot for daily data: too short and transaction costs dominate; too long and regime changes may invalidate the relationship before reversion occurs. Ernie Chan (2013) recommends exiting trades that extend beyond 2× the half-life.

---

## 2. What spurious correlation is and why it is mathematically inevitable

### The Yule–Granger–Phillips lineage of nonsense regressions

G. Udny Yule's 1926 paper "Why Do We Sometimes Get Nonsense-Correlations Between Time-Series?" demonstrated that the proportion of Church of England marriages and the standardized mortality rate (1866–1911) produced a correlation of **r = 0.9512** — an absurdity he described as "sheer nonsense; it has no meaning whatever." The correlation was 6.45 times its standard error, yielding odds of "many millions to one" against chance. Yet the result was entirely an artifact of two declining trends coinciding.

Granger and Newbold (1974) formalized this as **spurious regression**: when two independent I(1) series of length T = 50 are regressed against each other, the null hypothesis β = 0 is rejected at the 5% level roughly **75%** of the time, with R² averaging 0.77 and Durbin-Watson statistics near zero. Phillips (1986) then delivered the devastating asymptotic proof: the t-statistic grows as O(√T), meaning it **diverges to infinity** with sample size. In simulations, T = 200 yields t > 7 with R² = 20%; T = 49,000 yields t > 212 with R² = 48%. More data guarantees eventual spurious significance. Ernst, Shepp, and Wyner (2017) confirmed analytically that the standard deviation of the correlation between two independent Wiener processes is approximately **0.5** — the correlation is "frequently large in absolute value" despite complete independence.

### Why this happens: the mathematics of accumulation

Two independent random walks X_t = X_{t-1} + ε_t and Y_t = Y_{t-1} + η_t (with independent innovations) can be written as cumulative sums: X_T = X_0 + Σε_t. Each series is "self-correlated" — an integral of noise — so Var(X_T) = Tσ² grows linearly with time. Both series exhibit apparent trends purely by chance. When both happen to drift in the same direction, the sample correlation is high. The regression residuals inherit a unit root (are themselves non-stationary), the standard error of β̂ is inconsistently underestimated, and the t-statistic is inflated. Increasing sample size strengthens this artifact rather than correcting it.

### The common stochastic trend, confounding, and multiple testing problems

Beyond the pure random walk mechanism, spurious correlation arises through **common stochastic trends** (X_t = μ_t + u_t, Y_t = μ_t + v_t, where μ_t is a shared trend like GDP growth or population) and **omitted confounders** (ice cream sales and drowning deaths both driven by summer heat). In pairs trading, **multiple testing** is the most insidious source: testing N = 124,750 pairs at α = 0.05 yields an expected **6,238 false positives**. The Bonferroni correction (α_adj = α/N ≈ 4 × 10⁻⁷) is extremely conservative and destroys power. The **Benjamini-Hochberg FDR procedure** (1995) offers a superior alternative: rank p-values, find the largest k where p_(k) ≤ (k/N)·Q, and reject all hypotheses up to that rank. This controls the expected proportion of false discoveries rather than the probability of any single false discovery.

A low p-value from cointegration testing does not guarantee economic meaning. Research shows **32% of pairs** identified by distance methods fail to converge in the trading period (Do & Faff, 2010), and cointegration is often not persistent — management decisions, competition, or regulatory changes can permanently break the equilibrium.

---

## 3. Visual diagnostics that distinguish real pairs from statistical mirages

### Time series plots and spread stationarity

Plotting two series on the same time axis can be deceptive: trending together does not imply cointegration, since two independent random walks can appear to co-move. The definitive visual diagnostic is the **spread plot** S_t = Y_t − β̂·X_t. For a genuine cointegrated pair, the spread oscillates around a constant mean, appears bounded, and crosses the mean frequently (every 10–30 days for daily data). For a spurious pair, the spread drifts, wanders like a random walk, and shows infrequent mean-crossings with expanding range. A histogram of the spread should approximate normality with stable mean and variance.

### Rolling correlation reveals structural instability

Computing Pearson correlation on *returns* (not levels) over rolling windows of 60, 120, and 252 days produces a time series of correlation values. A genuine relationship shows **consistently high correlation (>0.6)** across windows with modest fluctuations; a coefficient of variation below 0.3–0.5 is the quantitative benchmark. A spurious relationship swings wildly — high in some periods, near-zero or negative in others. Monitoring at multiple timeframes simultaneously (short, medium, long) provides the strongest signal: if correlation drops below 0.3 across all windows, the relationship is likely decoupling.

### Residual diagnostics expose the spurious regression signature

After OLS regression, plotting residuals over time provides immediate visual evidence. **Stationary residuals** fluctuate around zero with constant variance; **non-stationary residuals** trend, drift, and show persistent runs above or below zero. The ACF of valid residuals decays quickly to zero within a few lags; the ACF of spurious residuals decays glacially, with lag-1 autocorrelation exceeding 0.95 — the hallmark of an I(1) process. The **Durbin-Watson statistic** serves as a quick check: DW ≈ 2 indicates no first-order autocorrelation (valid regression); **DW → 0** signals strong positive autocorrelation (spurious). The Granger-Newbold threshold — DW < R² — is a reliable red flag.

### Structural break tests guard against regime-dependent illusions

The **CUSUM test** (Brown, Durbin & Evans, 1975) plots cumulative sums of recursive residuals against significance boundaries; crossing indicates parameter instability. The **Chow test** evaluates a break at a known date via F-statistic. The **Bai-Perron test** (1998, 2003) detects multiple unknown breaks using dynamic programming, with a minimum segment size of 15% of the sample. A cointegrating relationship with structural breaks may appear stationary overall while having shifted its hedge ratio, equilibrium mean, or spread variance — making it unreliable for trading. The Gregory-Hansen (1996) test specifically accommodates one unknown structural break in the cointegrating relationship.

---

## 4. Why economic fundamentals are the ultimate filter

### The no-arbitrage argument demands a fundamental link

True cointegration in asset prices is sustained by **arbitrage forces** enforcing the Law of One Price. Gatev, Goetzmann, and Rouwenhorst (2006) explicitly linked pairs trading profitability to a "near-LOP" — two securities that are close substitutes should trade at comparable prices, and arbitrageurs are compensated for restoring equilibrium after perturbations. Their study found annualized excess returns of up to **11%** over 1962–2002. The Granger Representation Theorem ensures that cointegrated pairs must have an ECM representation with error correction pulling prices back to equilibrium. Without a fundamental economic mechanism driving this correction, any observed cointegration is a data-mining artifact that will break out-of-sample.

**Pepsi–Coca-Cola** makes sense: same industry, same consumers, same macro exposures (consumer spending, sugar prices, exchange rates), same regulatory framework. **S&P 500–Bangladesh butter** does not: there is no economic mechanism linking Bangladeshi dairy production to U.S. equity markets. The concept of **economic distance** — how far apart two assets are in fundamental drivers — provides a useful heuristic. A practitioner's fundamental scorecard should evaluate: industry/sector match, business model similarity, revenue driver overlap, macro factor alignment, supply chain linkage, regulatory commonality, geographic market overlap, and size comparability. Only pairs exceeding a threshold composite score should proceed to statistical testing.

### Granger causality provides weak but useful supporting evidence

Granger causality (Granger, 1969) tests whether past values of X improve predictions of Y beyond Y's own history, using an F-test on the restricted vs. unrestricted model. By the Granger Representation Theorem, **if two series are cointegrated, Granger causality must exist in at least one direction**. The converse is not true. The test requires stationary data (use the Toda-Yamamoto procedure with non-stationary series) and is sensitive to lag selection. Granger himself later preferred "temporally related" over "causal," and cautioned that "many ridiculous papers appeared" misinterpreting his test. Bidirectional Granger causality indicates a feedback relationship — common in genuinely linked financial assets.

### Market microstructure can create or destroy apparent relationships

Illiquid assets generate stale prices that create artificial co-movement. Bid-ask bounce introduces noise that distorts statistical tests. Trading-hour mismatches between assets in different time zones produce lagged correlations that are microstructure artifacts, not genuine relationships. Even with genuine cointegration, transaction costs (commissions, spreads, short-selling costs) can eliminate profitability — Gatev et al. acknowledged that "part of these profits may be due to market microstructure effects."

---

## 5. A ten-step framework for definitively rejecting spurious correlation

The following framework provides a complete, ordered checklist for a Pairs Selection Report. Steps 1–5 are **minimum requirements**; steps 6–10 provide additional confidence.

**Step 1 — ADF unit root test on individual series.** Both X and Y must be confirmed I(1): fail to reject the ADF null on levels (p > 0.05), reject on first differences (p < 0.05). Confirm with KPSS as a complementary test. If one series is I(0) and the other I(1), cointegration is mathematically impossible — stop immediately.

**Step 2 — Engle-Granger residual-based cointegration test.** Regress Y on X, obtain residuals, apply ADF with Engle-Granger critical values (−3.34 at 5% for two variables, more stringent than standard ADF). If the test statistic exceeds the critical value (is less negative), reject cointegration. Use p < 0.05 as the baseline threshold; p < 0.01 for conservative screening.

**Step 3 — Johansen trace and maximum eigenvalue tests.** Run within a VECM framework with lag length selected by BIC on the underlying VAR. For a bivariate pair, rank = 1 confirms a single cointegrating relationship. The Johansen test avoids the asymmetry problem and provides direct estimates of cointegrating vectors and adjustment speeds.

**Step 4 — Out-of-sample validation.** This is the **most powerful defense against spurious relationships**. Split data 70/30 into training and test periods. Estimate all parameters (hedge ratio β, spread mean, spread standard deviation, half-life) on training data. Apply fixed parameters to test data and verify: (a) spread remains stationary (ADF on out-of-sample spread), (b) mean and variance are stable, (c) half-life is consistent. If the spread drifts or variance explodes out-of-sample, the relationship is spurious or unstable.

**Step 5 — Economic plausibility filter.** Is there a fundamental reason these assets should be linked? Same industry, common demand drivers, supply chain relationship, or shared macro exposures? If no plausible economic mechanism exists, treat any statistical evidence with extreme skepticism — require p < 0.01 and impeccable out-of-sample performance. This qualitative filter is arguably the most important safeguard against data-snooping.

**Step 6 — Multiple testing correction.** Apply Benjamini-Hochberg FDR at 0.05 (or Bonferroni at α/N for maximum conservatism). For 1,225 pairs from 50 stocks, Bonferroni requires p < 0.0000408 per pair. BH-FDR is less conservative and more practical.

**Step 7 — Rolling correlation stability.** Compute return correlation at 60-day, 120-day, and 252-day windows. Require: coefficient of variation of rolling correlation < 0.5, minimum correlation > 0.3, and consistency across all three timeframes.

**Step 8 — Granger causality.** Confirm predictive relationship in at least one direction using the Toda-Yamamoto procedure (to handle non-stationary data correctly). Absence of Granger causality in either direction, when cointegration is claimed, is a red flag.

**Step 9 — Information criteria for model selection.** Compare cointegrated vs. non-cointegrated model specifications using BIC (preferred over AIC for its stronger penalty against overfitting). Lower BIC for the cointegrated model supports the relationship.

**Step 10 — Structural break tests.** Apply Bai-Perron (multiple unknown breaks) and CUSUM to the cointegrating regression. Significant breaks indicate the hedge ratio or spread equilibrium has shifted — the pair may not be reliably mean-reverting for trading.

| Step | Test | Pass criterion | Role |
|------|------|---------------|------|
| 1 | ADF on levels + differences | Both series I(1) | **Gate** |
| 2 | Engle-Granger | Residual ADF < −3.34 (5%) | Core evidence |
| 3 | Johansen | Trace/eigenvalue reject rank = 0 | Confirmation |
| 4 | Out-of-sample | Spread stationary on test data | **Most powerful** |
| 5 | Economic plausibility | Fundamental link exists | **Most important** |
| 6 | BH-FDR / Bonferroni | Survives correction | Multiple testing guard |
| 7 | Rolling correlation | CV < 0.5, min > 0.3 | Stability check |
| 8 | Granger causality | Significant in ≥1 direction | Supporting evidence |
| 9 | BIC comparison | Cointegrated model preferred | Model selection |
| 10 | Bai-Perron / CUSUM | No significant breaks | Reliability check |

---

## 6. The historical examples that prove the point

### Leinweber's Bangladesh butter: the canonical data-mining cautionary tale

In 1995, David Leinweber (Harvard Ph.D. in Applied Mathematics, MIT physics) deliberately searched a UN CD-ROM covering 140 countries to find the most absurd predictor of the S&P 500. Using annual data from 1983–1993 (just 10 data points), he found that **butter production in Bangladesh alone explained 75% of S&P 500 variation (R² = 0.75)**. Adding U.S. cheese production pushed R² to approximately 95%. Adding the sheep population of Bangladesh and the United States achieved **R² = 99%** — "an awesome fit." The correlation was, as Leinweber wrote, "utterly useless for anything outside the fitted period, a total crock before 1983 or after 1993." Published formally as "Stupid Data Miner Tricks: Overfitting the S&P 500" (*Journal of Investing*, 2007), the example became legendary — yet Leinweber continued receiving calls from investors asking about Bangladeshi butter production **twenty years later**.

### Tyler Vigen's automated absurdity engine

Tyler Vigen (Harvard Law, U.S. Army intelligence analyst) built software to systematically pair thousands of public data series and compute correlations, publishing the results at tylervigen.com and in his 2015 book *Spurious Correlations*. His most cited examples include: per capita cheese consumption vs. deaths by bedsheet tangling (**r = 0.947**), Nicolas Cage films vs. swimming pool drownings (**r = 0.666**), divorce rate in Maine vs. margarine consumption, and U.S. science spending vs. suicides by hanging. Every example exploits the same mechanism Yule identified in 1926: two series with trending means over the observed period create apparent correlation without causal connection.

### Financial folklore: Super Bowl indicators and hemline indexes

The **Super Bowl Indicator**, identified by sportswriter Leonard Koppett in 1978, claimed that NFC victories predict bull markets. It was correct for **23 consecutive years** and over 90% of the first 31 Super Bowls — then went 0-for-4 from 1998–2001. By 2024, accuracy had declined to roughly 71% overall and just **29% in recent decades**. The **Hemline Index** (attributed to George Taylor, 1920s) claims shorter skirts signal bull markets; van Baardwijk and Franses (2010) found hemlines *lag* economic conditions by approximately three years — a lagging indicator at best, not predictive. The **January Effect** (small-cap outperformance in January) has largely disappeared since its publicization, consistent with market efficiency arbitraging known anomalies away.

### The 2010 Flash Crash: when all correlations break at once

On May 6, 2010, the Dow plunged approximately **1,000 points (9%)** in minutes after Waddell & Reed initiated a sell program for 75,000 E-Mini S&P 500 futures contracts using an algorithm set to target 9% of prior-minute volume without price or time constraints. High-frequency trading firms entered a "hot potato" loop, then simultaneously withdrew liquidity, creating a vacuum where normal intermarket correlations and relative pricing relationships collapsed. Some stocks traded at $0.01 or $100,000; over **20,000 trades** were later cancelled. The crash demonstrated that statistical relationships — including cointegration between pairs — can break catastrophically during market stress. For pairs traders, the lesson is existential: cointegration is a long-run property that provides no protection during tail events.

---

## Conclusion: the hierarchy of evidence for genuine pairs

The mathematics is unambiguous. Phillips (1986) proved that t-statistics in spurious regressions diverge as √T — a result that inverts the normal logic of statistical inference. Leinweber showed that with 140 countries' worth of UN data, *any* target variable can be "explained" with R² = 99%. Vigen demonstrated that automated scanning of thousands of series reliably produces correlations exceeding 0.94 between cheese consumption and bedsheet deaths. These are not curiosities; they are the **default outcome** of naïve statistical analysis on non-stationary data.

The framework presented here establishes a clear hierarchy of evidence. **Economic plausibility** is the first and most important filter — without a fundamental mechanism, no amount of statistical significance should be trusted. **Out-of-sample validation** is the most powerful statistical safeguard against overfitting. **Cointegration testing** (Engle-Granger and Johansen) with proper critical values replaces misleading OLS regression. **Multiple testing correction** via Benjamini-Hochberg FDR protects against the combinatorial explosion of pair-wise testing. And **structural break tests** ensure the relationship is stable enough to trade.

The difference between a genuine pair (Pepsi–Coca-Cola, sustained by shared consumer demand and arbitrage forces) and a spurious one (S&P 500–Bangladesh butter, sustained by nothing) is not merely statistical — it is the difference between a tradeable equilibrium enforced by economic forces and a numerical coincidence that vanishes the moment it is acted upon. A rigorous Pairs Selection Report must demonstrate both statistical and economic evidence, survive out-of-sample testing, and withstand the scrutiny of every diagnostic in this framework.

### Key academic references

The foundational papers that underpin this framework include: Yule (1926, *JRSS*) on nonsense correlations; Granger & Newbold (1974, *Journal of Econometrics*) on spurious regression; Phillips (1986, *Journal of Econometrics*) on the asymptotic theory of diverging t-statistics; Engle & Granger (1987, *Econometrica*) on cointegration and error correction (Nobel Prize 2003); Johansen (1988, *JEDC*; 1991, *Econometrica*) on multivariate cointegration; Dickey & Fuller (1979, *JASA*) on unit root testing; Benjamini & Hochberg (1995, *JRSS-B*) on false discovery rate control; Gatev, Goetzmann & Rouwenhorst (2006, *Review of Financial Studies*) on pairs trading performance; Leinweber (2007, *Journal of Investing*) on data-mining overfitting; Ernst, Shepp & Wyner (2017, *Annals of Statistics*) on the analytical distribution of correlation between independent Wiener processes; and Vidyamurthy (2004, Wiley) and Chan (2013, Wiley) as practitioner references.