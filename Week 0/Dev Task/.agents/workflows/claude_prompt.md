# System Prompt & Context for Claude: 1987 Market Break Analysis

## Goal
Refactor `analyze_crash.py` to identify the October 1987 "market break" based on structural regime shifts rather than mere volatility.

## Constraints
1. **NO VOLATILITY SPIKE LOGIC:** You must explicitly remove the existing 2-sigma volatility logic. 
2. **DATA LIMITATIONS:** You only have `SP500_Futures` minute-level prices. You do NOT have cash index, volume, or bid-ask arrays.
3. **GAP-AWARENESS (CRITICAL):** Review `.agents/workflows/data_description.md`. You must NOT forward-fill `Halted` periods or compute returns across overnight gaps. Doing so creates artificial 0-variance streams that corrupt the changepoint mathematics.

## Execution Blueprint (from ChatGPT Deep Research)
1. **Pre-Processing Framework:** Calculate log returns strictly on adjacent trading minutes. Treat "Halted" blocks as either missing (Version A) or add them explicitly as a staleness metric (Version B).
2. **Feature Engineering - Multivariate Dysfunction:** Construct a clean multivariate series across time composed of:
   - *Staleness:* e.g., rolling instances of exactly 0% returns.
   - *Serial Dependence:* e.g., rolling 15m AR(1) coefficient or Auto-correlation.
   - *Jump-Share:* e.g., proportion of Realized Variance attributable to discontinuities (bipower variation).
3. **Changepoint Detection (Primary):** Apply the offline PELT algorithm (using Python's `ruptures` library or similar) across this multivariate feature set to find the **most robust single changepoint timestamp**.
4. **Outputs:** 
   - Print a robustness sensitivity table comparing the break timestamp across different rolling window sizes (e.g. 15m vs 30m).
   - Generate **exactly ONE plot** tracking the price with a vertical red line at the break timestamp, alongside the key feature (e.g. AR1) that triggered the break. No duplicate zoomed plots.
