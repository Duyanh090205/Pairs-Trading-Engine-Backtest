# Week 1 — Pair Discovery & Universe Construction

**Theme:** The Foundation.

## Objective

Build a clean, statistically rigorous data pipeline and identify candidate cointegrated pairs from S&P 500 equities for downstream signal generation.

## Deliverable

A **Pairs Selection Report** containing a screened universe, cointegration scan results, and a fully validated log-price panel ready for signal engine consumption.

## Scope

- Transform raw 1-minute OHLCV equity data into a clean, aligned, log-transformed 5-minute panel.
- Apply strict survivorship, liquidity, and data-quality filters to construct a tradable universe.
- Run an all-pairs Engle-Granger cointegration scan with Benjamini-Hochberg FDR correction.
- Compute Ornstein-Uhlenbeck half-lives to filter for actionable mean-reversion speeds.
- Produce audit-ready output artifacts (Parquet files + metadata) for Week 2.

## Data

- **Source:** 1-minute OHLCV CSVs, organized by monthly folders (`01`–`12`), one file per ticker per day (`{TICKER}_{DATE}.csv`).
- **Timestamps:** Nanosecond UTC integers → converted to `US/Eastern` (DST-aware).
- **Session Filter:** 09:35–15:55 ET (excludes opening/closing auction noise).
- **Resampling:** Aggregated to 5-minute bars using last observation.

## Method

### Data Engineering
1. **Universe Discovery:** Ticker must appear in all 12 months (continuous data rule).
2. **Hard Screens:** Median price ≥ $5, avg daily dollar volume ≥ $1M, completeness ≥ 90%, zero-return fraction < 50%.
3. **Outlier Treatment:** Z-score > 10 on minute-level returns → flagged, forward-filled. Tickers with > 1% outlier fraction dropped entirely.
4. **Log Transform & Alignment:** `ln(close)` → resample to 5-min → strict inner join across all tickers → perfectly rectangular T × N matrix with zero NaN.

### Cointegration Scan
1. **Engle-Granger two-step** (OLS hedge ratio + ADF on residuals) across all C(N,2) pairs.
2. **BH-FDR** multiple-testing correction at q = 0.05.
3. **OU Half-Life** filter: [5, 60] trading days.
4. **Economic Logic Filter:** Positive hedge ratio + same GICS sector (later relaxed in Week 4).

### Key Finding
Zero pairs survived the full filter funnel over 12 months of 2022 data — a critical empirical result that motivated the methodological upgrades in subsequent weeks (Johansen, Kalman, PCA).

## Directory Structure

```
Week 1/
├── notebooks/           # Data profiling, cointegration scan, audits
├── scripts/             # Report generation
├── notes/methodology/   # Step-by-step methodology documentation
├── outputs/             # Generated figures and tables
└── data/                # Processed Parquet artifacts (gitignored)
```
