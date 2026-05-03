---
description: Execute the Week 1 Cointegration and Pairs Trading pure Python pipeline, maintaining strict methodology rules against spurious correlations and prioritizing accurate timestamp alignment.
---

# Execution Workflow: Week 1 Cointegration & Pairs Trading (Jupiter Notebook)

This workflow defines the strict sequential steps for the pair-trading selection, minimizing overhead while upholding rigorous mathematical and economic proofs.

## Constraints (Do Not Do)
* **Do NOT arbitrarily interpolate or forward-fill missing data.**
* **Do NOT test individual I(1) stationarity on every single asset upfront.**
* **Do NOT run rolling regressions (moving betas).**
* **Do NOT generate charts for all 500 pairs.**
* **Do NOT write any execution or backtesting logic.**

## Phase 1: Deep Research & Methodology Definition
1. Enforce the Engle-Granger method: regress the pair, extract the residual, and run the ADF test **on the residual** (never on raw price).
2. Apply pragmatic I(1) assumptions, residual stationarity threshold, p-values, and hedge ratio sanity defined in `notes/methodology/spurious_correlation_rules.md`.

## Phase 2: Data Audit & Preparation
1. Load 1-minute stock files via Python data pipeline script.
2. Align timestamps perfectly on a common grid.
3. Drop sparse assets with excessive missingness or stale quotes.
4. **Critical:** Avoid interpolation unless explicitly justified. Use highly restricted forward-fill *only* if the gap length is trivial.
5. Export to `data/intermediate/aligned_1min_prices.parquet`.

## Phase 3: Prototype Coding (Optional Smoke Test)
1. Run a rapid syntax smoke-test using the `1987_crash_market_data.csv` in a standalone `.py` script to ensure the `statsmodels` syntax and DataFrame operations run smoothly.

## Phase 4: Universe Definition & Pair Generation
1. Choose the ticker universe from the available 1-min data.
2. Define a strict rule to generate ~500 candidate pairs. 
3. Exclude assets that are too sparse or short-lived.
4. Output a formalized list/dataframe of the ~500 candidate pairs.

## Phase 5: Full-Scale Pair Scan
1. Run the full Engle-Granger steps (regress -> residual -> run ADF on residual) on the candidate pairs via Python.
2. Output all raw statistics to `outputs/pair_scan_results/raw_adf_stats.csv`.

## Phase 6: Spurious-Correlation Audit & Filtering
1. **Statistical Filter:** Reject pairs whose residual failed ADF or whose hedge ratio is not sane. Perform pragmatic I(1) diagnostic testing *only* on the shortlisted pairs.
2. **Economic Logic Filter:** Prioritize same sector/industry -> substitute products/shared exposure. If the pair cannot be explained in one clear economic sentence, reject it unconditionally.
3. Export validated pairs to `outputs/final_outputs/approved_pairs.csv`.

## Phase 7: Final Python Outputs & Report Assembly
1. Python execution script must output a raw results table displaying **ALL** candidate pairs and their stats (universe before filter) as CSV or text formatting.
2. Python execution script must output a finalized results table of the **APPROVED** pairs (universe after filter).
3. Generate and save `.png` spread charts, residuals, and rolling z-scores strictly for the final approved winners.
4. Synthesize findings into the Pairs Selection Report.