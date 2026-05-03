# Week 4 — Multi-Regime Strategy Defense (The Thesis)

**Theme:** The Defense.

## Objective

Defend the cointegration pairs trading strategy before an AI Investment Committee by proving it generalizes across both Bear (2022) and Bull (2023–2026) market regimes. Produce a comprehensive Strategy Whitepaper.

## Deliverable

A **Strategy Whitepaper** covering the cointegration thesis, signal logic, and verified backtest results across multiple market regimes — structured as 12 reporting sections with regime-conditional analysis, overfitting diagnostics, and sensitivity grids.

## Scope

- Implement a 5-phase production pipeline: Data Gateway → Cointegration Discovery → Signal Generation & Execution → Backtest & Validation → Multi-Regime Defense.
- Upgrade from Engle-Granger to **Johansen cointegration test** (symmetric, multivariate).
- Replace OLS hedge ratios with **PCA (secondary eigenvector)** for formation and **2D Kalman Filter** for execution.
- Run a **45-fold monthly rolling walk-forward** validation (6-month formation, 1-month trading).
- Execute **One-At-a-Time (OAT) sensitivity analysis** across 9 parameters (~27 additional runs × 45 folds).
- Validate with negative controls, latency stress tests, and overfitting diagnostics (DSR + PBO).

## Data

- **Source:** 1-minute OHLCV, S&P 500 universe.
- **Date Range:** January 3, 2022 – March 19, 2026 (~4.2 years).
- **Regimes:** Late Bear 2022 (Folds 1–6), Early Bull 2023 (7–18), Mid Bull 2024 (19–30), Late Bull 2025–Q1 2026 (31–45).

## Method

### Phase 0 — Data Quality Gateway
Standalone vectorized module with timestamp normalization, session filtering, Z-score outlier treatment, and hard assertions (monotonicity, OHLC validity, no duplicates).

### Phase 1 — Cointegration Discovery
- **Universe Screens:** Median price ≥ $5, ADV ≥ $1M, completeness ≥ 90%, zero-return < 50%.
- **Hedge Ratio:** PCA secondary eigenvector (Avellaneda-Lee convention).
- **Cointegration:** Johansen trace + max eigenvalue test.
- **Multiple Testing:** BH-FDR at q = 0.05.
- **OU Half-Life:** [1, 10] trading days (recalibrated from Week 1's [5, 60] for intraday execution).

### Phase 2 — Signal Generation & Execution
- **Kalman Filter:** 2D state `[α, β]` with auto-selected δ via multi-criterion optimization (kurtosis + half-life + ACF78).
- **Spread:** Computed from Kalman **prior** state (genuine OOS residual).
- **State Machine:** Z = ±2.0 entry, zero-crossing exit, Numba @njit.
- **Position Sizing:** Dollar-normalized, threshold rebalance at 10% β drift (with hysteresis dead band).

### Phase 3 — Backtest & Validation
- **Walk-Forward:** 45 monthly folds, 6-month formation + 1-month trading + embargo gap.
- **Transaction Costs:** 60 bps round-trip (30 entry + 30 exit) + 50 bps/year borrow cost.
- **Negative Control:** CVNA/ISRG (empirical) + synthetic random walks, with bootstrap threshold.
- **Latency Stress:** {t+1, t+2, t+5, t+10} bar delays + random latency.

### Phase 4 — Multi-Regime Defense
- **Regime Partition:** Sharpe distributions per regime (Bear/Bull breakdown).
- **Pair Persistence:** Decay curve of 2022-identified pairs maintaining cointegration through 2026.
- **Volume Stratification:** Hussein-inspired tertile analysis (within-tertile pair formation).
- **Overfitting:** Deflated Sharpe Ratio + Probability of Backtest Overfitting.
- **OAT Sensitivity:** 9 parameters × ~3 values each + 3 targeted interaction combos.

## Directory Structure

```
Week 4/
├── src/
│   ├── phase0_data_gateway/   # Data quality gateway
│   ├── phase1_cointegration/  # Pair discovery pipeline
│   ├── phase2_execution/      # Kalman signal + sizing engine
│   ├── phase3_backtest/       # Walk-forward orchestrator
│   ├── phase4_defense/        # Regime analysis + diagnostics
│   └── utils/                 # Shared utilities
├── docs/
│   ├── methodology/           # Strategy whitepaper methodology
│   └── pipeline/              # Phase-by-phase pipeline specs
├── scripts/                   # Standalone execution scripts
├── tests/                     # Unit tests
├── results/                   # Output artifacts
├── run_final_pipeline.py      # Master pipeline orchestrator
└── data/                      # Validated data artifacts (gitignored)
```
