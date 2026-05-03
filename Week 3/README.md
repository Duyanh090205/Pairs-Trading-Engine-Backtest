# Week 3 — Verified Backtest Engine & Data Integrity Audit

**Theme:** The Verification.

## Objective

Build a verified backtest engine with institutional-grade PnL mechanics, and construct a systematic framework for detecting look-ahead bias through deliberate data corruption experiments.

## Deliverables

1. **Deliverable 1 — "Creating the Bad Data":** Four distinct look-ahead bias injection methods (future-close substitution, timestamp backdating, spread-level injection, full-dataset normalization leak), each swept across 5 contamination levels (10%–50%) to produce 20 flawed datasets. The goal is to empirically measure how each type of bias inflates strategy Sharpe.

2. **Deliverable 2 — Verified Backtest Log:** A 7-section audit log including trade logs, timestamp verification proof, Sharpe sensitivity tables, sizing comparison, threshold sensitivity, and negative control results — serving as a reproducible audit trail.

## Scope

- Implement a custom vectorized backtest engine (pandas + Numba) inheriting the Week 2 signal logic.
- Run the engine under two parallel sizing regimes: static OLS β and monthly Kalman β rebalance.
- Compute daily mark-to-market PnL with 60 bps round-trip transaction costs (split 30 bps entry + 30 bps exit).
- Perform Sharpe sensitivity analysis across all 20 flawed datasets + 1 clean baseline.
- Execute engine-level vs. data-level bias comparison.
- Validate with negative control pairs (CVNA/ISRG, INTC/JPM).

## Data

- **Source:** Same 1-minute OHLCV data, session 09:30–15:59 ET.
- **Pairs:** CMS/DUK (primary, EG t = −5.268), DOW/LYB (secondary), CVNA/ISRG and INTC/JPM (negative controls).
- **Parameters Inherited from Week 2:** OLS β = 1.0487, α = −0.6956, Kalman β ≈ 0.76–0.78, rolling window = 680 bars, Z = 2.0, warmup = 30 bars, execution lag = 1 bar.

## Method

### Engine Architecture
- **Signal:** OLS Z-score with Week 2 state machine, Z = 2.0 fixed entry, zero-crossing exit.
- **Sizing Version A:** Static OLS β = 1.0487 (fixed throughout Jul–Dec 2022).
- **Sizing Version B:** Kalman β monthly rebalance (uses end-of-prior-month Kalman estimate).
- **PnL:** Daily mark-to-market. `Equity[d] = Equity[d−1] × (1 + R_d)`, starting from 1.0.
- **Sharpe:** `mean(daily_return) / std(daily_return) × √252`.
- **Timestamp Verification:** Assert `exec_ts > signal_ts` on 100% of trades.

### Four Look-Ahead Bias Methods
| Method | Mechanism | Realistic? | Detection Difficulty |
|--------|-----------|-----------|---------------------|
| H1 — Future Close | Replace `close[t]` with `close[t+1]` for k% rows | High | Low |
| H2 — Timestamp Backdating | Subtract 60s from `window_start` for k% rows | Medium | High |
| H3 — Spread Injection | Write `S(t+1)` into `spread_biased` at row t | High | Medium |
| H4 — Normalization Leak | Normalize k% tickers using full-year mean/std | Very High | Very High |

### Sensitivity Analyses
- **CP6a:** 20-cell Sharpe table (4 methods × 5 k-values).
- **CP6b:** Engine-level injection (`position[t] = position[t]` removes execution lag).
- **CP6c:** OLS sizing vs. Kalman sizing comparison.
- **CP6d:** Z = 2.0 vs Z = 2.57 threshold sensitivity.
- **CP6e:** Negative control sanity check.

## Directory Structure

```
Week 3/
├── scripts/
│   ├── deliverable1.py    # Flawed dataset generation
│   ├── deliverable2.py    # Full backtest + sensitivity sweeps
│   ├── engine.py          # Core vectorized backtest engine
│   ├── utils.py           # Data loading, spread, Z-score utilities
│   └── verify_params.py   # Parameter verification
├── notebooks/             # Analysis notebooks
├── logs/                  # Backtest audit logs
├── data/                  # Clean + 20 flawed datasets (gitignored)
├── Week3_Final_Plan_v3.md
└── Week3_Workflow_Checkpoint.md
```
