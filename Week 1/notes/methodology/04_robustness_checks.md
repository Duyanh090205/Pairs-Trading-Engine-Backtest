# Notebook 04 — Robustness Checks: Methodology

## Purpose

When the main pipeline produces a null result (zero approved pairs), a natural follow-up question is: **would a slightly different but equally reasonable design have found something?** This notebook explores three alternative specifications to test the **robustness** of the null result. 

**Critical distinction:** These are appendix-level explorations. They do NOT replace or override the official main result from Notebook 02. Any candidates found here are labeled "exploratory" and would require formally revising the approved methodology before being promoted.

---

## Alternative Spec 1: Same-Sector-Only Universe

### The idea
Instead of testing all 1,225 pairs, restrict the universe to only **within-sector** pairs. Many real-world pairs trading desks do this — they only trade pairs within the same industry because cross-sector pairs rarely have a genuine economic link.

### How it changes the math
The full universe has `m = 1,225` pairs. Restricting to same-sector pairs reduces this to `m = 174`.

**The critical impact is on BH-FDR.** The BH critical value for rank `k` is:
```
BH_critical_k = (k / m) × q
```

When `m = 1,225`: BH critical for rank 1 at q=0.05 is `(1/1225) × 0.05 = 0.0000408`
When `m = 174`: BH critical for rank 1 at q=0.05 is `(1/174) × 0.05 = 0.000287`

The same-sector threshold is **7× more lenient**. A p-value that was too large to survive BH with 1,225 tests might survive with only 174 tests. This is mathematically legitimate — fewer tests means fewer opportunities for false positives, so the penalty should be lighter.

### What stays the same
Everything else is identical: the same data, the same `coint()` parameters, the same half-life computation, the same hedge ratio filter. The only change is which pairs enter the test set.

### What the code does
1. Filters `scan_df` to keep only `within_sector == True` pairs
2. Extracts their raw p-values
3. Re-applies BH-FDR at q=0.05 and q=0.10 using the smaller `m = 174`
4. For any BH survivors, computes half-life and checks downstream filters (HL in [5,60], hedge ratio > 0)
5. Reports how many pairs pass each filter stage

### Interpretation
If the same-sector analysis produces exploratory candidates, it suggests the main result's null finding was partly driven by the large number of irrelevant cross-sector tests inflating the BH penalty. This would be a legitimate methodological consideration for future work.

---

## Alternative Spec 2: Bidirectional Engle-Granger

### The problem with unidirectional EG
The Engle-Granger cointegration test has a well-known limitation: it's **asymmetric**. When you regress `A ~ B`, OLS finds the line that minimizes vertical distances (residuals along the A-axis). When you regress `B ~ A`, it minimizes vertical distances along the B-axis. These are different lines, producing different residuals, and therefore different ADF test statistics and p-values.

Visually: imagine a scatterplot of A vs B. OLS(A~B) finds the best-fit line that minimizes errors measured vertically. OLS(B~A) finds the best-fit line that minimizes errors measured horizontally. Unless the correlation is perfect (R²=1), these are different lines.

### What the code does
For the top 15 "near-miss" pairs (smallest raw p-values, excluding GOOG-GOOGL):

1. **Forward test:** `coint(A, B)` — the same direction used in the main run
2. **Reverse test:** `coint(B, A)` — the opposite direction  
3. Record both p-values and identify which direction gives the smaller (more significant) p-value
4. Take `min_pval = min(forward_p, reverse_p)` — the most favorable result

### The key question
Does using `min_pval` instead of the forward-only p-value change any BH-FDR outcome?

```python
BH_rank1_critical = 1/1225 × 0.05 = 0.0000408
```

If even the best-case `min_pval` for any near-miss pair still exceeds 0.0000408, then the alphabetical ordering convention was not the reason for the null result.

### Important caveat
A properly designed bidirectional test would actually **double** the test count (testing both `coint(A,B)` and `coint(B,A)` for every pair), which would make BH-FDR **stricter** (critical value for rank 1 would be `1/2450 × 0.05`). The code's approach of using `min_pval` without doubling tests is already a generous/optimistic scenario.

---

## Alternative Spec 3: Daily-Close Frequency

### The motivation
An influential argument in econometrics (articulated by Dave Giles and others) is that for unit root and cointegration tests, **time span matters more than sample size**. Our 5-minute data gives ~19,000 observations per pair, but it's all within a single year. The high-frequency sampling may actually hurt the test because:

1. **Microstructure noise:** At 5-minute frequency, prices are contaminated by bid-ask bounce, temporary order imbalances, and high-frequency trading activity. This noise adds variance to the spread that isn't related to the fundamental cointegration relationship.

2. **Overfitting of lag structure:** The ADF test's `autolag='aic'` might select long lag structures to absorb high-frequency autocorrelation patterns that are noise, not signal. This can distort the test statistic.

3. **Anti-power effect:** In theory, more data should increase statistical power. But if the extra data is mostly noise, it can actually reduce the signal-to-noise ratio of the test.

### What the code does
1. **Resample the 5-minute panel to daily close:**
   ```python
   daily_panel = panel.resample('1D').last().dropna()
   ```
   This takes the last 5-minute observation of each trading day as the daily close, producing ~250 observations per pair (one per trading day in 2022).

2. **Select the top 20 near-miss pairs** from the main run

3. **Re-run `coint()` on daily data** for each pair:
   ```python
   coint(log_a_daily, log_b_daily, trend='c', autolag='aic')
   ```
   Note: `maxlag` is left at the default (which will be much smaller than 30 since the series is only ~250 observations long).

4. **Compare daily p-values against 5-minute p-values:** Does the same pair show stronger cointegration evidence at daily frequency?

5. **Re-compute half-life on daily data:** Since the data is already daily, the OU regression directly gives half-life in trading days (no need to divide by 77 bars/day as with 5-minute data):
   ```
   half_life_days = -ln(2) / λ_daily
   ```

### What to watch for
- If many pairs have **lower** p-values at daily frequency, it suggests the 5-minute data was too noisy for cointegration detection
- If p-values are similar or higher at daily frequency, the null result is robust across frequencies

### The BH caveat
The code applies BH-FDR to just the 20-pair daily subset. Since `m = 20` is tiny, the BH threshold is extremely lenient (rank 1 critical = 1/20 × 0.05 = 0.0025). Results from this small-sample BH are not comparable to the main run's BH on 1,225 tests. A fair comparison would require testing all 1,225 pairs at daily frequency, which this notebook does not do (it's an exploratory check, not a full alternative pipeline).

---

## Summary Comparison Table

The code produces a final table comparing all specifications:

| Specification | Pairs Tested | Raw p < 0.05 | BH q=0.05 | BH q=0.10 | Pass All | Status |
|--------------|-------------|-------------|-----------|-----------|----------|--------|
| **OFFICIAL MAIN** | 1,225 | ~5% | 0 | 1 | 0 | AUDITED FINAL |
| Alt 1: Same-sector | 174 | varies | varies | varies | varies | APPENDIX |
| Alt 2: Bidir EG | 15 | varies | 0 | 0 | 0 | APPENDIX (sensitivity) |
| Alt 3: Daily close | 20 | varies | varies | varies | 0 | APPENDIX (exploratory) |

---

## Final Robustness Conclusion

The notebook's conclusion addresses four points:

1. **The official result stands:** The audited full-run pipeline found 0 approved pairs. No alternative specification produced a result that would warrant changing this conclusion.

2. **Each alternative tested a different design choice:** reduced universe (Alt 1), regression direction (Alt 2), and data frequency (Alt 3). None exposed a fundamental flaw.

3. **Any exploratory candidates are clearly labeled** as such and not promoted to the official result.

4. **Changing the result would require revising the approved methodology** — which is a deliberate design decision, not something done retroactively after seeing results. This protects against the cardinal sin of quantitative research: data snooping (changing your methodology to get the result you want).
