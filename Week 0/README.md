# Week 0 — 1987 Black Monday Backtest & Market Break Detection

**Theme:** The Crash.

## Objective

Build an institutional-grade, multivariate changepoint detection system capable of identifying structural market breaks, using the October 19, 1987 Black Monday crash as the primary case study. This project serves as the foundational exercise in market microstructure analysis before the pairs trading pipeline begins in Week 1.

## Deliverable

A complete **Market Break Detection Report** containing:
1. A Jupyter Notebook (`week0_submission.ipynb`) with end-to-end analysis and visualizations.
2. A multivariate PELT-based detection engine that pinpoints the exact minute of market structural failure.
3. Strategic memos and autopsy reports evaluating market behavior during extreme volatility.
4. Generalization tests across multiple historical crises (2008 Financial Crisis, 2020 COVID Crash).

## Scope

The project is divided into three parallel workstreams:

| Workstream | Focus | Key Output |
|------------|-------|------------|
| **Dev Task** | Core detection engine — multivariate PELT changepoint detection on S&P 500 Futures microstructure data | `analyze_crash.py`, detection at **09:42 AM** (8 min into crash) |
| **Strategist Task** | Strategic analysis — portfolio insurance evaluation, regime shift memos, and PDF report generation | Strategic memos, autopsy reports |
| **Vibe Coding Task** | Experimental analysis — backtest visualizations, automated post-mortem crash reports | Interactive notebooks, diagnostic charts |

## Data

| Property | Value |
|----------|-------|
| **Asset** | S&P 500 Futures |
| **Frequency** | 1-minute intervals |
| **Period** | October 16–21, 1987 (6 trading days) |
| **Rows** | ~1,440 minutes |
| **Trading Hours** | 9:30 AM – 4:00 PM EST |
| **Known Halts** | Oct 19: 60 min (12:00–13:00), Oct 20: 90 min (~11:30–13:00) |

Additional datasets for generalization testing: 2008 Financial Crisis, 2020 COVID Crash (sourced via yfinance).

## Method

### Feature Engineering (Rolling Windows: 15-min and 30-min)

| Feature | Calculation | Economic Meaning |
|---------|-------------|------------------|
| **Staleness** | Fraction of NaN + zero-return bars | Price discovery failure, system overload |
| **Autocorrelation** | Lag-1 autocorrelation of log returns | Serial dependence, bid-ask bounce from friction |
| **Jump-Share** | Fraction of moves exceeding 2σ | Discontinuous repricing, liquidity collapse |

### Changepoint Detection
- **Algorithm:** PELT (Pruned Exact Linear Time) with RBF kernel on standardized features.
- **Input:** All 3 dysfunction signals simultaneously (multivariate detection).
- **Penalty:** 5 (tuned for early detection without false positives).
- **Gap-Aware Returns:** Log returns computed only between consecutive trading minutes — halts skipped (not forward-filled), session boundaries respected.

### Key Result
- **Detection:** October 19, 1987 at **09:42 AM EST** — 8 minutes into the crash.
- **Convergence:** Both the 15-min window (09:43) and 30-min window (09:42) agree within 1 minute → high confidence.
- **Feature Signature at 09:42:** Staleness elevated (1.54), autocorrelation negative (−0.45, bid-ask friction), jump-share emerging (0.05).

### Historical Context
- Largest single-day percentage decline in S&P 500 history (−22.6%).
- Triggered immediate circuit breakers and led to the SEC Brady Task Force investigation.

## Directory Structure

```
Week 0/
├── week0_submission.ipynb      # Primary end-to-end analysis notebook
├── Dev Task/
│   ├── analyze_crash.py        # Core PELT detection engine
│   ├── analyze_crash.ipynb     # Interactive analysis notebook
│   ├── data/                   # 1987 market data CSVs
│   ├── outputs/                # Charts, reports
│   └── research/               # Background research
├── Strategist Task/
│   ├── strategist_memo.ipynb   # Strategic analysis notebook
│   ├── Code/                   # Memo generation scripts
│   ├── Portfoilo Insurance Report/
│   └── outputs/                # PDF memos
├── Vibe Coding Task/
│   ├── blackmonday_backtest.ipynb  # Experimental backtest
│   ├── src/                    # Source modules
│   └── outputs/                # Diagnostic charts
├── Readings/                   # Reference literature
├── data/                       # Shared datasets (gitignored)
└── outputs/                    # Shared output artifacts
```
