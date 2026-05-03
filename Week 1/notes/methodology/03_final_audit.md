# Notebook 03 — Final Red-Team Audit: Methodology

## Purpose

When a quantitative pipeline produces a dramatic result (zero approved pairs out of 1,225 tested), the natural question is: **did we make a mistake?** This notebook is a formal, independent audit — often called "red-teaming" in quantitative finance — that systematically verifies every component of the pipeline to confirm the zero-pair result is genuine, not a bug.

**Critical constraint:** This notebook is strictly **read-only**. It does NOT change any parameters, thresholds, or methodological choices. It only re-checks the existing work.

---

## Part 1: Manual 5-Pair Re-Check (Independent Recomputation)

### What the code does
Five pairs are selected to cover a diverse range of scenarios:
- `GOOG-GOOGL` — same company (Alphabet Class A vs Class C shares), the only BH-FDR survivor
- `COP-CVX` — within-sector (Energy), both oil companies
- `BAC-JPM` — within-sector (Financials), both large banks
- `HD-MS` — cross-sector pair
- `LOW-MS` — cross-sector pair

For each pair, the code **independently recomputes everything from scratch**:
1. Extracts raw log-price vectors from the panel
2. Runs `coint()` with the exact same parameters (`trend='c'`, `autolag='aic'`, `maxlag=30`)
3. Runs OLS regression to get the hedge ratio
4. Constructs the spread and computes the OU half-life
5. Counts zero crossings of the demeaned spread

### The comparison algorithm
For each of the three key values (t-statistic, p-value, hedge ratio), the code compares the recomputed value against the stored value from Notebook 02:

```python
tstat_match = abs(stored_tstat - recomp_tstat) < 1e-4
pval_match  = abs(stored_pval  - recomp_pval)  < 1e-4
hr_match    = abs(stored_hr    - recomp_hr)     < 1e-4
```

**Why 1e-4 tolerance?** Floating-point arithmetic is not perfectly reproducible across different execution environments (different NumPy versions, different CPU instruction sets, different memory layouts). A tolerance of 0.0001 is tight enough to catch any genuine bug (wrong column order, wrong regression convention) while accommodating harmless floating-point rounding differences.

**What this proves:** If all 5 pairs match within tolerance, it confirms:
- The `coint()` function was called correctly (right series order, right parameters)
- The OLS convention is consistent (A ~ const + B, not B ~ const + A)
- The stored results weren't corrupted during the save/load cycle
- The same statsmodels algorithms produce the same results when re-run

---

## Part 2: Top-10 Smallest Raw P-Values Audit

### What the code does
This section picks the 10 pairs with the lowest raw p-values and manually traces why each one was rejected by BH-FDR.

### The BH critical value calculation (done by hand in the code)
For each pair at rank `k` out of `m = 1,225` valid tests:

```
BH_critical_k(q=0.05) = (k / 1225) × 0.05
BH_critical_k(q=0.10) = (k / 1225) × 0.10
```

The code explicitly computes these and compares them against each pair's raw p-value.

### The key finding
- **Rank 1 (GOOG-GOOGL):** Raw p-value ≈ 0.000049. BH critical at q=0.10 is (1/1225)×0.10 ≈ 0.000082. Since 0.000049 < 0.000082, this pair **passes** BH-FDR at q=0.10.
- **Rank 2:** Raw p-value jumps to approximately **20× larger** than Rank 1. The BH critical for rank 2 is (2/1225)×0.10 ≈ 0.000163, but the actual p-value far exceeds this.

**Why is the gap so large?** GOOG and GOOGL are literally the same company (Alphabet Inc.) with two share classes. Their prices are mechanically linked by arbitrage. Every other pair consists of genuinely different companies, and in 2022's volatile market, none maintained a stable long-run equilibrium.

### What this proves
The BH-FDR procedure is working correctly — it's not a bug that's accidentally rejecting valid pairs. The test statistics genuinely show that no pair (except the trivial same-company GOOG-GOOGL) has statistically significant cointegration after multiple-testing correction.

---

## Part 3: Full Rejection Audit Table

### What the code does
Loops through **all 1,225 valid pairs** and, for each one, determines exactly where in the pipeline it was rejected. This produces a complete audit trail.

### The rejection classification system

For each pair, the code assigns one of these statuses:

| Status | Meaning |
|--------|---------|
| `rejected_bhfdr` | Failed the BH-FDR multiple testing correction (raw p-value too large relative to its rank) |
| `rejected_halflife` | Passed BH-FDR but the spread's half-life was outside the acceptable range, or the spread was not mean-reverting (λ ≥ 0) |
| `rejected_hedgeratio` | Passed BH-FDR and half-life, but hedge ratio was ≤ 0 |
| `rejected_economic` | Passed all statistical filters but failed the economic logic screen (cross-sector with no documented linkage) |
| `near_miss` | Would have passed all filters under the relaxed q=0.10 regime |
| `approved` | Passed everything (none in this dataset) |

### Computing diagnostics for all pairs
Unlike Notebook 02 (which only computed half-life for BH-passing pairs), this audit notebook recomputes half-life and zero crossings for **every** pair. This allows the audit to identify whether rejected pairs had reasonable spread dynamics that were hidden by the BH filter, or whether they were genuinely poor candidates.

---

## Part 4: Final Audit Conclusion

### The formal verdict
The code prints a structured audit report covering six areas:

1. **Timestamp handling** — Panel covers the full date range in US/Eastern timezone, index is unique and monotonic
2. **Pair ordering consistency** — All 5 audited pairs confirmed that coint() and OLS use the same A-B alphabetical convention
3. **P-value extraction** — MacKinnon N=2 critical values are correctly applied by statsmodels
4. **BH-FDR correction** — Mathematical verification that the BH procedure is implemented correctly
5. **Half-life calculation** — GOOG-GOOGL's half-life of ~0.47 days correctly triggers the [5,60] day filter
6. **Rejection logging** — The proportion of pairs with raw p < 0.05 (~5%) is consistent with what random chance would produce under the null hypothesis

### The conclusion
The audit found **no evidence of implementation error**. The zero-approved-pairs result is interpreted as a valid empirical finding driven by:
- The 2022 market regime (Fed tightening, inflation, sector rotation) weakening mean-reversion dynamics
- The strict but defensible multi-layer filtering design
- The BH-FDR correction being the decisive rejection stage — no pair even reached the economic logic filter
