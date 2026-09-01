# Pairs Trading Engine & Backtest

A statistical arbitrage pipeline built end to end, May-July 2026 — and the evidence that it
does not work. **The headline result is negative.** Screening the S&P 500 through an
all-pairs cointegration scan with Benjamini-Hochberg correction, **zero pairs survived** the
full filter funnel over twelve months of 2022 data.

The pipeline is the point: raw data ingestion, pair discovery, signal generation, a backtest
engine with look-ahead bias injected on purpose to measure how much it inflates Sharpe,
microstructure-aware friction modeling, and a live paper-trading deployment. It was built to
be capable of finding an edge, then used honestly enough to report that it found none.

## Quickstart

**The scan cannot be re-run from this repository.** The raw 1-minute price panel is
several gigabytes and is not committed, so the cointegration screen has no input here.
Saying so is cheaper than shipping a command that fails.

What is committed, and what to read instead:

| Where | What it shows |
|---|---|
| [`Week 1/Week1_Pairs_Selection_Report.ipynb`](Week%201/Week1_Pairs_Selection_Report.ipynb) | The pair-discovery notebook with outputs saved — the filter funnel, and the zero survivors at the end of it |
| [`Week 3/`](Week%203/) | The look-ahead bias study: the injection scripts and run logs |
| [`Week 4/results/`](Week%204/results/) | The 45-fold monthly walk-forward validation output |
| [`Week 6/`](Week%206/) | The live paper-trading engine, cost model and audits |

Every week folder has its own README describing what was built and what it found.

---

## Pipeline Architecture

```
Week 1                Week 2              Week 3              Week 4                Week 5               Week 6
The Foundation  →  The Signal  →  The Verification  →  The Defense  →  The Friction  →  Going Live
─────────────     ───────────     ────────────────     ───────────     ─────────────     ──────────
Universe &         Z-Score         Backtest Engine      Multi-Regime     Microstructure    Paper Trading
Pair Discovery     Signal Engine   + Bias Detection     Walk-Forward     Cost Model        Deployment
                                                        Defense
```

---

## Weekly Progression

### [Week 1 — Pair Discovery & Universe Construction](Week%201/)
Build a clean, log-transformed 5-minute price panel from raw 1-minute OHLCV data. Screen the S&P 500 universe through liquidity/quality filters, run an all-pairs Engle-Granger cointegration scan with BH-FDR correction, and identify candidate pairs.

**Key finding:** Zero pairs survived the full filter funnel over 12 months of 2022 data — motivating methodological upgrades in later weeks.

---

### [Week 2 — Z-Score Signal Engine](Week%202/)
Translate cointegrated spreads into actionable trading signals. Build a Numba-compiled state machine for path-dependent position tracking, implement rolling Z-score normalization with half-life-derived windows, and introduce a Rolling Kalman Filter for dynamic hedge ratio estimation.

**Key finding:** Static OLS β drifts by 25.5% over 12 months, proving dynamic rebalancing is a mathematical necessity.

---

### [Week 3 — Verified Backtest Engine & Data Integrity](Week%203/)
Construct a verified backtest engine with daily mark-to-market PnL, and systematically inject four types of look-ahead bias into 20 flawed datasets to empirically measure how data corruption inflates Sharpe ratios. Validate with negative control pairs.

**Key finding:** Full-dataset normalization leak (H4) is the most dangerous bias — moderate Sharpe inflation but nearly impossible to detect from the data file alone.

---

### [Week 4 — Multi-Regime Strategy Defense](Week%204/)
Upgrade the entire pipeline to production grade: Johansen cointegration, PCA hedge ratios, 2D Kalman Filter with auto-selected δ, and a 45-fold monthly walk-forward validation spanning 2022–2026. Defend the strategy across Bear and Bull regimes with OAT sensitivity analysis and overfitting diagnostics.

**Key deliverable:** A 12-section Strategy Whitepaper presented to the AI Investment Committee.

---

### [Week 5 — Microstructure Reality (Net-of-Fees)](Week%205/)
Replace the static 60 bps cost assumption with a three-component dynamic friction model: empirical half-spread + instability-scaled market impact + borrow cost, calibrated from a 213M-row limit order book dataset. Answer the core question: does alpha survive real-world friction?

**Key deliverable:** A Net-of-Fees Performance Report with side-by-side Gross vs. Static vs. Dynamic cost comparisons.

---

### [Week 6 — Cloud Deployment (Paper Trading)](Week%206/)
Deploy the strategy into a live paper-trading environment with real-time Kalman state updates, WebSocket resilience testing (intentional 5-minute API disconnect), and a Drift Monitor comparing live execution costs to the Week 5 model's predictions.

**Status:** Architecture specification phase.

---

## Technical Stack

- **Language:** Python 3.x
- **Core Libraries:** NumPy, pandas, SciPy, statsmodels
- **Performance:** Numba `@njit` for stateful execution loops
- **Data Format:** Apache Parquet
- **Statistical Methods:** Johansen cointegration, PCA, Kalman Filter, Ornstein-Uhlenbeck, BH-FDR, Deflated Sharpe Ratio, PBO

---

## Repository Structure

```
Quant Program/
├── Week 1/     # Pair Discovery & Universe Construction
├── Week 2/     # Z-Score Signal Engine
├── Week 3/     # Verified Backtest Engine & Data Integrity
├── Week 4/     # Multi-Regime Strategy Defense
├── Week 5/     # Microstructure Reality (Net-of-Fees)
├── Week 6/     # Cloud Deployment (Paper Trading)
├── Readings/   # Reference literature
└── .gitignore
```

> **Note:** Large datasets (`data/`, `Big Dataset/`, `*.parquet`) are excluded from version control via `.gitignore`.
