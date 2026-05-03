# Pairs trading methodology: a rigorous guide to cointegration-based pair selection

**Cointegration-based pairs trading remains one of the most principled statistical arbitrage strategies, but its practical implementation is riddled with subtle errors that can invalidate results.** This guide covers the complete methodology for testing ~500 equity pairs using the Engle-Granger framework with daily data over 1–3 years, from interpreting ADF test statistics on residuals through constructing a defensible composite ranking. The central challenge is not finding pairs that pass a cointegration test — at 5% significance, roughly 25 out of 500 will pass by chance alone — but rather building a rigorous pipeline that controls false discoveries, filters for practical tradability, and produces results that hold up under academic scrutiny. Every threshold, formula, and framework below is grounded in the academic literature (Engle & Granger 1987, Gatev et al. 2006, Avellaneda & Lee 2010) and practitioner best practices (Chan 2013, Hudson & Thames).

---

## 1. What the ADF test actually measures on cointegration residuals

The Augmented Dickey-Fuller test, when applied to residuals from an OLS regression of one price series on another, tests whether the **linear combination** Y − βX is stationary. This is fundamentally different from applying ADF to a raw price series. On a raw series, ADF tests whether that single series has a unit root. On estimated residuals, it tests whether two non-stationary series share a common stochastic trend — the definition of cointegration.

The **Engle-Granger two-step procedure** operationalizes this. Step 1: regress Y on X via OLS (Yₜ = α + βXₜ + uₜ), obtaining the hedge ratio β and residuals ûₜ. Step 2: apply ADF to ûₜ, testing H₀: residuals have a unit root (no cointegration). The OLS estimator of β is "super-consistent" — it converges at rate T rather than √T — but the residuals are *estimated*, not observed. This distinction has a critical consequence for critical values.

**The standard Dickey-Fuller critical values are wrong for this test.** Because OLS minimizes residual variance, it biases the test toward finding stationarity even when none exists. The correct critical values, derived by MacKinnon (1991, 2010) via Monte Carlo simulation, are substantially more negative:

| Significance | Standard ADF | Engle-Granger (2 variables) | Difference |
|---|---|---|---|
| 1% | −3.43 | **−3.96** | 0.53 stricter |
| 5% | −2.86 | **−3.41** | 0.55 stricter |
| 10% | −2.57 | **−3.12** | 0.55 stricter |

A test statistic of −3.20, for example, would reject the null under standard ADF at 5% but **fail to reject** under the correct Engle-Granger distribution. Using standard critical values leads to systematic over-rejection — falsely concluding cointegration exists in far too many pairs. MacKinnon's response surface formula C(p) = β∞ + β₁/T + β₂/T² provides finite-sample corrections; for T = 250 observations at 5%, the critical value tightens to approximately **−3.36**.

For interpreting p-values: **p < 0.01** represents very strong evidence of cointegration (excellent candidate), **p < 0.05** is the standard academic threshold (good candidate), **p < 0.10** is marginal, and **p ≥ 0.10** means the pair should be rejected. In Python, the `statsmodels.tsa.stattools.coint()` function correctly uses MacKinnon cointegration-specific critical values. However, manually running `adfuller()` on OLS residuals returns **standard** Dickey-Fuller critical values — one of the most common implementation errors in practice.

---

## 2. The hedge ratio determines everything about the spread

The hedge ratio β, estimated as the slope from Yₜ = α + βXₜ + εₜ, is the cointegrating coefficient that defines how the tradable spread is constructed: **Spread = Y − β·X**. If β = 1.3, then for every 1 share of Y purchased, 1.3 shares of X are sold short. This ratio ensures the portfolio is approximately market-neutral — without it, a 1% move in a $200 stock creates a $2 P&L impact versus $0.50 for a $50 stock, producing a massively unbalanced position.

Three estimation approaches exist, with markedly different properties:

**Static OLS** uses the full historical window to estimate a single β. It is simple and super-consistent, but assumes the relationship is constant — often unrealistic over 1–3 years — and introduces look-ahead bias if applied to the full sample including future data. **Rolling OLS** re-estimates β at each time step using only the most recent W observations, adapting to changing conditions but introducing sensitivity to window length. Short windows (30–60 days) produce noisy estimates; long windows (120–252 days) adapt slowly. Empirical analysis of the classic EWA-EWC pair shows rolling OLS hedge ratios **varying wildly between 0.6 and 1.2**. **The Kalman filter** treats β as an unobserved state variable evolving via a random walk (βₜ = βₜ₋₁ + wₜ) and recursively updates estimates through a predict-correct cycle. For the same EWA-EWC pair, Kalman-estimated β stayed between **0.55 and 0.65** — dramatically more stable. Chan (2013) and multiple empirical comparisons show Kalman filter strategies achieving 2–5× the cumulative returns of rolling OLS approaches.

A pair should be disqualified when the hedge ratio is unstable. Specifically: if β **flips sign** during the formation period (the fundamental relationship has reversed), the pair must be rejected outright. If the coefficient of variation of rolling β exceeds **0.3–0.5**, the spread definition is unreliable. If 60-day and 252-day β estimates diverge by more than 30%, exercise caution. OLS hedge ratios are also asymmetric — regressing Y on X produces a different β than regressing X on Y — so practitioners should test both orderings and select the one producing the more stationary spread (more negative ADF statistic).

---

## 3. Half-life of mean reversion quantifies practical tradability

Even when cointegration is statistically confirmed, the **speed** of mean reversion determines whether a pair is practically tradable. The Ornstein-Uhlenbeck (OU) process provides the theoretical foundation: dX(t) = θ(μ − X(t))dt + σdW(t), where θ is the speed of mean reversion, μ is the long-run mean, and σ is volatility. The key parameter θ governs how quickly the spread reverts: deviations from equilibrium decay as e^(−θt).

To estimate the half-life from data, run an AR(1) regression on the spread: Δz(t) = λ·z(t−1) + ε, where a **negative λ** indicates mean reversion. The half-life formula is:

> **half_life = −ln(2) / λ**

Equivalently, if φ is the AR(1) coefficient from z(t) = φ·z(t−1) + ε, then half_life = −ln(2)/ln(φ). This works because deviations decay exponentially, and the half-life is the time for a deviation to shrink to 50% of its initial value.

Practitioner consensus converges on **5–30 trading days** as the optimal range for daily equity pairs. Chan (2013) uses half-life to set lookback windows for Bollinger-band signals. Avellaneda & Lee (2010) require the annualized mean-reversion speed κ > 252/30 ≈ 8.4, implying reversion within ~30 days. Half-lives below **2 days** are problematic because reversion is indistinguishable from noise at daily frequency and transaction costs consume any profit. Half-lives above **60 days** tie up capital for months, expose the trade to regime changes, and produce fewer round-trips per year (lower Sharpe ratio). A spread with a 100-day half-life takes ~200 days for full reversion — during which corporate events, earnings surprises, or sector rotations can permanently break the relationship.

**Including half-life in a screening notebook is recommended** as a secondary filter after cointegration is confirmed. ADF tells you *whether* the spread mean-reverts; half-life tells you *how fast*. However, the estimate is sensitive to lookback window choice, and the calculation assumes the spread follows an OU process — which may not hold exactly. Use it as a range filter (5–60 days) and an input for trading parameters (lookback period, maximum holding time), not as a standalone selection criterion.

---

## 4. Correlation is a pre-filter, not a selection criterion

The distinction between correlation and cointegration is arguably the single most important conceptual point in pairs trading. **Correlation** measures short-term co-movement of *returns* — whether two stocks tend to go up or down on the same days. **Cointegration** measures whether two *price levels* maintain a stable long-run equilibrium. These are fundamentally different properties, and confusing them is a common source of failed trades.

The Portfolio Optimization textbook (Palomar, HKUST) demonstrates this mathematically: two synthetically constructed series can have **correlation of 0.95** yet completely fail the Engle-Granger cointegration test — their spread drifts permanently. Conversely, two series can be perfectly cointegrated with a **correlation of only 0.11** — their price levels stay tethered but daily returns are driven by uncorrelated idiosyncratic noise. Hudson & Thames confirmed experimentally that two highly correlated price series can produce an Engle-Granger ADF statistic of only 0.41 — nowhere near significance.

The trap is clear: entering a pairs trade based on high correlation alone, "you might watch the spread widen indefinitely." Huck & Afawubo (2015) provided the definitive empirical evidence: correlation-based pair selection produced weak excess returns, while **cointegration-based methods generated robust excess returns of 1.38% to 5% per month** after transaction costs. Do & Faff (2010) found that 32% of pairs selected via distance-based methods (essentially correlation-adjacent) failed to converge during the trading period.

Correlation's legitimate role is **computational**: for N = 100 stocks, there are 4,950 possible pairs. Pre-filtering by correlation > 0.5–0.7 or restricting to same-sector pairs dramatically reduces the number of expensive cointegration tests. But in a composite ranking score, correlation should receive **at most 5% weight** — or zero. No major academic or practitioner source recommends giving correlation significant weight alongside cointegration metrics.

---

## 5. Testing 500 pairs demands multiple testing correction

At a 5% significance level, testing 500 pairs produces an expected **25 false positives** by pure chance. The probability of at least one false positive is effectively 100%. Without correction, a substantial fraction of "cointegrated" pairs are statistical artifacts that will fail out-of-sample.

**Bonferroni correction** divides the significance level by the number of tests: α_adjusted = 0.05/500 = 0.0001. This is mathematically valid but devastatingly conservative for pairs screening — Harlacher (2016) documented that it "impedes the discovery of even truly cointegrated combinations." The preferred approach is the **Benjamini-Hochberg (BH) procedure**, which controls the False Discovery Rate (FDR) — the expected proportion of false positives among rejected hypotheses, rather than the probability of any single error.

The BH procedure works as follows: sort all 500 p-values in ascending order, compute the critical value cᵢ = (i/500) × q for FDR level q (typically 0.05), find the largest rank k where p_(k) ≤ c_k, and reject all hypotheses with rank ≤ k. In Python:

```python
from statsmodels.stats.multitest import multipletests

reject, pvals_adjusted, _, _ = multipletests(
    pvals=adf_pvalues,   # array of 500 raw ADF p-values
    alpha=0.05,           # FDR level
    method='fdr_bh'       # Benjamini-Hochberg
)
passing_pairs = np.where(reject)[0]
```

At FDR q = 0.05, approximately 5% of accepted pairs are expected to be false discoveries — a far more practical threshold than Bonferroni's near-zero tolerance. The correction should be applied **first**, before other filters. The rationale: multiple testing correction addresses the statistical validity of the cointegration claim itself. Half-life, hedge ratio stability, and zero crossings are quality refinements applied to the pool of statistically valid pairs. An effective complementary strategy is **sector pre-partitioning**: restricting tests to within-sector pairs reduces the number of comparisons from ~500 to ~50–100 per sector, improving both statistical power and economic rationale (Sarmento & Horta 2020).

---

## 6. A defensible pass/fail framework with specific thresholds

The selection pipeline should apply hard filters sequentially, check for red flags, then score surviving pairs on soft criteria. The following thresholds are synthesized from the academic literature and practitioner consensus:

**Hard filters (must-pass):**

| Criterion | Threshold | Rationale |
|---|---|---|
| Minimum data length | ≥ 252 trading days (1 year) | ADF power requires sufficient span; 504 days preferred for 1–3 year horizons |
| ADF adjusted p-value | < 0.05 after BH FDR correction | Cointegration must survive multiple testing correction |
| Hurst exponent | < 0.5 | Confirms mean-reverting regime (H = 0.5 is random walk) |
| Hedge ratio sign | β > 0 for same-sector pairs | Negative β implies both legs move same direction — not a mean-reverting spread |
| Sub-window stability | ADF passes in ≥ 2 of 3 sub-periods | Ensures cointegration is not confined to one lucky window |

**Soft filters (score penalties if violated):**

| Criterion | Preferred range | Rationale |
|---|---|---|
| Half-life | 5–60 trading days | < 5: noise/transaction costs; > 60: too slow, regime risk |
| Zero crossings | ≥ 12 per year | More crossings = more trading opportunities |
| Spread Sharpe ratio | > 0.5 (annualized, in-sample) | Below 0.5 unlikely to survive transaction costs |
| Hedge ratio CV | < 0.30 | Rolling β coefficient of variation; higher = unstable spread |

**Red flags (automatic disqualification):**

Regardless of ADF results, a pair should be rejected if: the hedge ratio **flips sign** during the formation period; average daily dollar volume is below **$1M** for either leg; the pair lacks any **economic rationale** (e.g., a biotech paired with a utility); a **structural break** is detected in the spread (via Chow test or CUSUM); either stock faces **delisting risk** or has experienced a major corporate event (M&A, spin-off); or the hedge ratio is extreme (β > 5 or β < 0.2), creating impractical position imbalances.

---

## 7. Ranking surviving pairs with a composite score

After filtering, rank surviving pairs using **percentile-rank scoring**, which converts each metric to a 0–100 scale across the surviving pool, then computes a weighted average. This approach is robust to outliers and puts all metrics on comparable footing — unlike z-score normalization, which assumes approximate normality and is sensitive to extreme values.

The recommended composite score uses the following weights, informed by Caldeira & Moura (2013), who found that cointegration strength and mean-reversion speed are the primary drivers of strategy performance, and Avellaneda & Lee (2010), who emphasized the importance of reversion speed:

| Metric | Direction | Weight | Description |
|---|---|---|---|
| ADF test statistic (absolute) | Higher = better | **25%** | More negative ADF → stronger stationarity evidence |
| Half-life proximity | Closer to 15–25 days = better | **20%** | Score = 100 − |HL − 20| scaled; penalizes extreme values |
| Spread Sharpe ratio | Higher = better | **15%** | In-sample risk-adjusted mean-reversion profitability |
| Zero crossings per year | Higher = better | **15%** | More crossings → more independent trading opportunities |
| Hedge ratio stability | Lower CV = better | **10%** | CV of 60-day rolling β; captures spread reliability |
| ADF adjusted p-value | Lower = better | **10%** | Provides FDR-adjusted confidence level |
| Hurst exponent | Lower = better | **5%** | Reinforces mean-reversion confirmation (redundant with ADF but useful) |

Correlation receives zero explicit weight. It serves only as a computational pre-filter (threshold ≥ 0.5–0.7) before running cointegration tests. If included at all, cap it at 5% weight by reallocating from one of the other metrics.

An example ranking output table should include these columns:

| Column | Description |
|---|---|
| `pair_id` | Ticker pair identifier (e.g., "XOM-CVX") |
| `sector` | GICS sector of both stocks |
| `adf_tstat` | Engle-Granger ADF test statistic |
| `adf_pval_raw` | Raw (unadjusted) p-value |
| `adf_pval_adj` | BH-adjusted p-value |
| `hedge_ratio` | OLS β from cointegrating regression |
| `hedge_ratio_cv` | Coefficient of variation of 60-day rolling β |
| `half_life` | OU half-life in trading days |
| `hurst` | Hurst exponent of the spread |
| `zero_crossings` | Annual zero-crossings of the spread |
| `spread_sharpe` | Annualized Sharpe ratio of the spread |
| `return_corr` | Pearson correlation of daily returns |
| `composite_score` | Weighted percentile-rank composite |
| `rank` | Final rank (1 = best) |

Select the **top 10–20 pairs** for the final report, consistent with Gatev et al.'s (2006) top-20 portfolio construction that produced average annualized excess returns of up to 11%.

---

## 8. Seven errors that invalidate pairs trading analysis

**Using standard ADF critical values on estimated residuals.** This is the most technically damaging error. The Engle-Granger critical values at 5% are −3.41, not the standard −2.86. A test statistic of −3.20 would pass under standard ADF but correctly fail under Engle-Granger — meaning the pair is not actually cointegrated. Always use `statsmodels.tsa.stattools.coint()`, never manually apply `adfuller()` to OLS residuals.

**Confusing correlation with cointegration.** Two stocks with 0.95 return correlation can have a permanently diverging spread. The mathematical proof is straightforward: correlation captures co-movement of first differences while cointegration requires stationarity of levels. Huck & Afawubo (2015) showed empirically that correlation-based selection produces weak returns while cointegration-based methods generate robust alpha.

**Data snooping from mass testing without correction.** Testing 500 pairs at 5% significance yields ~25 spurious results. López de Prado (2018) argues that "most backtests published in journals are flawed, as the result of selection bias on multiple tests." Anghel (2021) estimated that at least **50% of positive findings** in trading rule research may be false discoveries. The BH procedure is the minimum acceptable correction.

**Overfitting to in-sample cointegration.** Cointegration relationships break down. Clegg (2014) provides evidence against persistent cointegration, finding spreads are "typically affected by a steady stream of permanent shocks." Do & Faff (2010) confirmed that **32% of identified pairs did not converge** during the trading period. Sub-sample testing, walk-forward analysis, and periodic re-estimation are essential safeguards.

**Survivorship bias in universe construction.** Using only currently listed stocks excludes bankruptcies (catastrophic losses on long positions) and delistings. Research shows survivorship bias inflates annual returns by **1–4%** in backtests. Gatev et al. (2006) tested robustness by assuming −100% return on delisted long positions. Point-in-time databases (e.g., CRSP) that include delisted stocks are the proper data source.

**Look-ahead bias in hedge ratio estimation.** Full-sample OLS produces the "best possible" β by minimizing residual variance across the *entire* period, including future data. In live trading, the hedge ratio at time t should use only data up to time t. Rolling or expanding window estimation, or Kalman filtering, is necessary for honest backtests. As a Deutsche Bank quantitative strategy paper noted, look-ahead bias is "probably the most common bias in backtesting."

**Spurious cointegration from shared sector exposure.** Two tech stocks may appear cointegrated simply because both track the Nasdaq. This is not a tradable equilibrium relationship — it is driven by a confounding factor. To detect this, regress both stocks on the sector ETF first, then test cointegration of the residuals. If cointegration vanishes after removing the common factor, the original finding was spurious. Avellaneda & Lee (2010) addressed this directly by decomposing returns via PCA or sector ETFs before modeling the idiosyncratic component.

---

## How to avoid overclaiming in your research report

When writing conclusions about pairs trading results, epistemic honesty is non-negotiable. Use language that matches the actual strength of evidence: "suggests" rather than "proves," "appears to generate positive excess returns in our sample period" rather than "generates profits," and "provides evidence consistent with" rather than "establishes." Always report the number of pairs tested alongside the correction method applied — stating "we identified 15 cointegrated pairs" without mentioning you tested 500 is misleading.

Every conclusion section should explicitly acknowledge at minimum four limitations: potential data-snooping bias (even after BH correction, some false discoveries remain), survivorship bias if using currently listed stocks, the omission of transaction costs and short-selling costs from the analysis, and the well-documented instability of cointegration relationships out-of-sample. Compare results to the benchmark literature — Gatev et al. (2006) found up to 11% annualized excess returns but with declining profitability over time; Do & Faff (2010, 2012) showed that after transaction costs, profits diminished substantially post-2002.

A well-calibrated concluding statement reads: "Our analysis provides suggestive evidence that cointegration-based pair selection identifies mean-reverting spreads in-sample, though the persistence of these relationships out-of-sample remains uncertain. We applied Benjamini-Hochberg FDR correction at q = 0.05 across 500 tested pairs, and the surviving pairs exhibited half-lives and zero-crossing frequencies consistent with practical tradability. However, these findings should be viewed as exploratory and would benefit from validation on independent datasets and realistic transaction cost modeling."

---

## Conclusion

The pairs trading methodology described here rests on a sequence of principled decisions, each supported by academic evidence. The Engle-Granger procedure with MacKinnon critical values provides the statistical foundation. The Benjamini-Hochberg correction transforms a mass-testing exercise into a statistically defensible screening. The composite ranking — weighted toward ADF strength, half-life proximity to the 15–25 day sweet spot, and spread Sharpe ratio — translates statistical significance into practical tradability. Three insights deserve emphasis beyond what is commonly discussed. First, the **~0.55 gap** between standard and Engle-Granger critical values at 5% significance means that a naive implementation using `adfuller()` could misclassify a substantial fraction of pairs. Second, **half-life is not merely a nice-to-have** — it is the bridge between statistical cointegration and economic viability, and Avellaneda & Lee's κ > 8.4 filter is effectively a half-life ceiling of ~25 days. Third, the multiple testing problem is not theoretical — with 500 tests, it is the single largest source of false positives, and ignoring it undermines the credibility of any pair selection report.