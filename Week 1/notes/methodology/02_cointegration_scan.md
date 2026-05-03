# Notebook 02 — Cointegration Scan: Methodology

## Purpose

This is the core analytical notebook. It takes the clean panel of 50 log-price series from Notebook 01 and tests every possible pair (1,225 combinations) for **cointegration** — a statistical property meaning two non-stationary price series share a long-run equilibrium that the spread between them tends to revert to. If found, this property is the theoretical basis for a pairs trading strategy.

---

## Step 1: Data Loading & Validation

### What the code does
Loads the `log_prices_5min.parquet` file produced by Notebook 01. Runs five assertions (DatetimeIndex, no duplicates, monotonic, no NaN, all float64) to confirm data integrity. Re-localizes the timezone to `US/Eastern` if Parquet stripped it.

**Why re-localize?** Parquet files sometimes drop timezone metadata. The code defensively checks and re-applies it. This doesn't change any values — it just ensures downstream code that references timezone-aware operations works correctly.

---

## Step 2: Sector Mapping & Economic Rationale

### What the code does
Every ticker is manually mapped to a GICS (Global Industry Classification Standard) sector. Additionally, a detailed dictionary `ECON_RATIONALE` provides pre-written explanations for why specific pairs of stocks might be economically linked.

### Why this exists
Statistical significance alone is not enough. The code implements a "dual-layer" anti-spurious-correlation defense:

1. **Statistical layer** — BH-FDR correction (covered in Step 5)
2. **Economic layer** — A pair must have a defensible economic reason to be co-integrated

The economic rationale dictionary categorizes pairs into tiers:
- **Tier 1:** Direct competitors in the same sub-industry (e.g., AMD-INTC are both CPU makers)
- **Tier 2:** Same broader sector with shared demand drivers (e.g., AMD-MU share data center demand)  
- **Tier 3:** Same sector but weak connection (e.g., BAC-COIN are both "Financials" but one is a bank and the other is a crypto exchange)

### The assertion
```python
assert len(unmapped) == 0
```
This ensures every ticker has a sector assignment. If a new ticker enters the universe without a mapping, the code crashes immediately rather than silently skipping the economic filter.

---

## Step 3: Pair Generation

### What the code does
Uses `itertools.combinations(sorted(tickers), 2)` to generate all unique unordered pairs.

### The math
For `n = 50` tickers, the number of unique pairs is:
```
C(50, 2) = 50! / (2! × 48!) = 50 × 49 / 2 = 1,225
```

Each pair is assigned a canonical ID using alphabetical ordering: if the two tickers are AMD and INTC, the pair ID is always `AMD-INTC`, never `INTC-AMD`. This prevents testing the same pair twice and ensures consistency across all downstream analysis.

For each pair, the code also records whether it's "within-sector" (both tickers share the same GICS sector) or "cross-sector."

---

## Step 4: The Cointegration Testing Workflow

This is the mathematical heart of the notebook. For each of the 1,225 pairs, two analyses are performed sequentially.

### Phase 1: The Engle-Granger Cointegration Test

**What is cointegration?**  
Two time series are cointegrated if:
- Each series individually is **non-stationary** (has a unit root — it wanders randomly like a stock price)
- But there exists a **linear combination** of the two that IS stationary (mean-reverting)

This is different from correlation. Two stocks can be highly correlated (they both went up this year) but not cointegrated (the gap between them keeps growing). Cointegration means the gap between them tends to snap back to a stable level.

**The Engle-Granger two-step procedure:**

**Step 1 — OLS Regression:**
Given two log-price series `A_t` and `B_t`:
```
A_t = α + β × B_t + ε_t
```
where:
- `α` (alpha) is the intercept
- `β` (beta) is the **hedge ratio** — how many dollars of B you need to short for every dollar of A you buy
- `ε_t` is the residual (the "spread")

The OLS regression minimizes the sum of squared residuals to find `α` and `β`:
```
minimize Σ(A_t - α - β × B_t)²
```

The estimated spread is:
```
Spread_t = A_t - α̂ - β̂ × B_t
```

**Step 2 — ADF Test on Residuals:**
The spread series is tested for stationarity using an Augmented Dickey-Fuller (ADF) test. The ADF test fits an autoregressive model:
```
ΔSpread_t = γ × Spread_{t-1} + Σ(δ_i × ΔSpread_{t-i}) + v_t
```
where:
- `γ` (gamma) is the key coefficient
- The sum adds lagged differences to absorb serial correlation
- `v_t` is white noise error

The hypothesis test:
- **H₀ (null):** `γ = 0` → the spread has a unit root → NOT stationary → NO cointegration
- **H₁ (alternative):** `γ < 0` → the spread is mean-reverting → cointegration exists

The more negative `γ` is, the faster the spread reverts to its mean.

**How `coint()` works in the code:**
```python
coint_t, pvalue, crit_values = coint(
    series_a, series_b,
    trend='c',        # include a constant in the ADF regression
    autolag='aic',    # automatically select lag length using AIC criterion
    maxlag=30         # don't test more than 30 lags
)
```

The function internally:
1. Runs OLS of A on B
2. Takes residuals
3. Runs ADF on residuals
4. Computes the p-value using **MacKinnon critical values for N=2 variables**

These MacKinnon critical values are special — they account for the fact that we're testing residuals from a regression (not a raw series), which changes the distribution of the test statistic. Using standard ADF tables would give incorrect p-values.

**What `autolag='aic'` does:**
The ADF test can include lagged differences (the `Σ(δ_i × ΔSpread_{t-i})` terms) to absorb autocorrelation. But how many lags? Too few → residual autocorrelation biases the test. Too many → wastes degrees of freedom and reduces power. The AIC (Akaike Information Criterion) automatically picks the lag count that balances fit vs. complexity, up to a maximum of 30.

### Phase 2: OLS for Hedge Ratio & Spread Construction
After `coint()` provides the verdict (p-value), a separate OLS regression computes the hedge ratio and constructs the actual spread series:

```python
ols_model = sm.OLS(series_a, sm.add_constant(series_b)).fit()
hedge_ratio = ols_model.params[1]
spread = series_a - hedge_ratio * series_b
```

The spread is stored in memory for later half-life computation and plotting.

---

## Step 5: Statistical Filtering — The Three-Stage Funnel

After all 1,225 pairs are tested, three sequential filters are applied. Each pair must pass ALL three to survive.

### Stage 1: Benjamini-Hochberg False Discovery Rate (BH-FDR)

**The problem of multiple testing:**
When you run 1,225 independent statistical tests at α = 0.05, you expect about `1,225 × 0.05 ≈ 61` false positives purely by chance — pairs that appear cointegrated but aren't. If you just picked everything with p < 0.05, most of your "discoveries" would be noise.

**How BH-FDR works:**

The BH procedure controls the **expected proportion of false discoveries** among all discoveries (the False Discovery Rate), rather than the probability of any single false positive (which Bonferroni controls).

Algorithm:
1. Sort all 1,225 raw p-values from smallest to largest: `p_(1) ≤ p_(2) ≤ ... ≤ p_(m)` where `m = 1,225`
2. For each rank `k`, compute the BH critical value: `BH_critical_k = (k / m) × q` where `q = 0.05`
3. Find the largest `k` such that `p_(k) ≤ BH_critical_k`
4. Reject all hypotheses with rank `1` through `k`

**Concrete example from this data:**
- Rank 1 critical value: `(1/1225) × 0.05 = 0.0000408`
- Rank 2 critical value: `(2/1225) × 0.05 = 0.0000816`
- Rank 3 critical value: `(3/1225) × 0.05 = 0.0001224`

So the smallest p-value must be below 0.0000408 — an incredibly strict threshold. This is why BH-FDR is so powerful at eliminating false positives when doin many simultaneous tests.

**In the code:**
```python
reject, pvals_adj, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
```
The `reject` array tells you which pairs passed. The function also returns adjusted p-values that can be compared directly against `q`.

### Stage 2: Ornstein-Uhlenbeck Half-Life

For pairs that survive BH-FDR, the code measures **how fast** the spread reverts to its mean.

**The OU Process:**
The Ornstein-Uhlenbeck process models mean-reversion as a stochastic differential equation:
```
dX_t = θ(μ - X_t)dt + σdW_t
```
where:
- `X_t` is the spread at time t
- `μ` is the long-run mean
- `θ` is the speed of mean-reversion (higher = faster reversion)
- `σ` is volatility
- `W_t` is a Wiener process (random noise)

**Discrete-time approximation (what the code actually computes):**
```
ΔSpread_t = λ × Spread_{t-1} + error
```
This is an AR(1) regression of the spread's change on its lagged level.

The code runs this via OLS:
```python
model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
lambda = model.params['lag']
```

**Interpreting λ (lambda):**
- `λ < 0`: The spread reverts toward its mean (good — this is what we want)
- `λ = 0`: Random walk (no reversion, useless for pairs trading)
- `λ > 0`: The spread is explosive (dangerous — the spread grows without bound)

**Computing half-life:**
The half-life measures how many bars it takes for the spread to close half the gap between its current value and the mean:
```
half_life_bars = -ln(2) / λ
half_life_days = half_life_bars / 77    (77 five-minute bars per trading day)
```

**The [5, 60] day filter:**
- **Below 5 days:** The spread reverts too fast. By the time you detect the deviation, enter the trade, and pay transaction costs, the opportunity is already gone. This also picks up microstructure noise rather than genuine economic reversion.
- **Above 60 days:** The spread reverts too slowly. You'd have to hold the position for months, tying up capital and accumulating carry costs (margin interest, borrow costs for the short leg). The risk-adjusted return is too low.

### Stage 3: Positive Hedge Ratio

The hedge ratio `β` from OLS must be positive (`β > 0`).

**Why:** A positive hedge ratio means the two stocks move in the same direction on average. This is the economic requirement for a long-short pairs trade — you go long one and short the other. If `β < 0`, the two stocks move in opposite directions, and the "spread" doesn't represent a meaningful economic relationship — it's just shorting two things that both go up, which is a directional bet, not a pairs trade.

### Fallback Rules
If fewer than 10 pairs survive all three stages:
1. **Relax half-life** to [3, 90] days
2. If still fewer than 5: **relax BH-FDR** to q = 0.10

These fallbacks are documented in the code and flagged in the output as "relaxed filters" so the reader knows the results used weaker criteria.

---

## Step 6: Economic Logic Filter

Any pair surviving the statistical funnel faces a final check: **does an economic reason exist for these two stocks to be cointegrated?**

### The algorithm
```
if pair_id in ECON_RATIONALE:
    → PASS (with documented rationale and tier)
elif both tickers in same sector:
    → PASS (auto-assigned as "same sector, shared macro exposure")
else:
    → REJECT ("cross-sector, no identifiable economic linkage")
```

**Why this filter exists:** Statistical tests can find patterns in random data. Two stocks from completely different industries (say, an airline and a pharmaceutical company) might show a statistically significant co-movement purely by coincidence during the test period. The economic filter ensures that only pairs with a **logical reason** for their linkage are promoted.

---

## Step 7: Ranking Approved Pairs

Approved pairs are ranked by two criteria:
1. **Primary:** BH-adjusted p-value (lower is better)
2. **Secondary:** Distance of half-life from 20 trading days (closer to 20 is better)

The "ideal" half-life of ~20 days represents a sweet spot: fast enough to profit within a month, slow enough to allow comfortable position management.

---

## Step 8: Visualization

The code generates several diagnostic plots:

- **Pair plots** (for approved, near-miss, and rejected pairs): Top panel shows normalized log prices overlaid; bottom panel shows the spread with ±1σ and ±2σ bands
- **P-value histogram:** Distribution of all 1,225 raw p-values, with a vertical line at 0.05
- **Rejection funnel:** Bar chart showing how many pairs remain after each filter stage
- **Half-life distribution:** Histogram of half-lives for BH-passing pairs with filter boundaries marked

---

## Appendix A: Sensitivity Checks (within Notebook 02)

### A1: Excluding ETF-Involved Pairs
Removes the 5 ETFs (EEM, FXI, GLD, IWM, KWEB) and COIN from the universe, leaving 44 pure equity tickers. Re-runs BH-FDR on the smaller test set (C(44,2) = 946 pairs). Tests whether the zero-pair result is driven by ETFs inflating the test count.

### A2: Bidirectional Engle-Granger
The EG test is asymmetric — `coint(A, B)` can give a different p-value than `coint(B, A)` because OLS minimizes residuals along different axes. This check runs both directions for the top 10 near-miss pairs and takes the better (smaller) p-value. Tests whether the alphabetical ordering convention accidentally penalized promising pairs.
