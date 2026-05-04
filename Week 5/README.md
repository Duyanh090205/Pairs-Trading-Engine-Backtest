# Week 5 — Microstructure Reality (Net-of-Fees Backtest)

**Theme:** The Friction.

## Objective

Determine whether the Week 4 strategy's alpha survives when the static 60 bps cost assumption is replaced with empirical bid-ask spreads that widen during spread instability. Does alpha survive — or was it an illusion eaten by friction?

## Deliverable

A **Net-of-Fees Performance Report** presenting side-by-side comparisons across three cost regimes (Gross, Static 60 bps, Dynamic), with cost decomposition waterfalls, regime-conditional analysis, kill-zone identification, and overfitting diagnostics on net returns.

## Scope

This week **measures** friction on the existing strategy. We do **NOT** modify the strategy itself (no sizing changes, no new entry/exit gates, no regime suppression). Same signals, same positions — different cost model.

The pipeline is structured as four plans:
- **Plan 0 — Data Gateway:** Orderbook ingestion, spread extraction, rolling instability computation.
- **Plan 1 — Cost Model:** Three-component dynamic cost function (spread + impact + borrow).
- **Plan 2 — Cost Application:** Per-trade post-processing pass over Week 4's trade log (no re-execution).
- **Plan 3 — Validation & Report:** 11-section performance report with diagnostics.

## Data

| Property | Value |
|---|---|
| **File** | `orderbook.parquet` |
| **Total Rows** | 212,975,144 |
| **Frequency** | 1-minute (dominant; some 2–5 min gaps) |
| **Tickers** | 504 (2022-01) → 526 (2026-03) |
| **Date Range** | 2022-01-03 09:00 → 2026-03-19 23:59 |
| **Structure** | 3-level limit order book (L1/L2/L3 bid/ask price + size) |
| **L1 Spread** | Universe median ~10 bps; SPY ~4.7 bps; max > 150 bps (stress) |

**Known Limitation:** `bid_sz == ask_sz` at all levels (symmetric/synthetic LOB). No order-flow imbalance signals can be derived.

## Method

### Three-Component Dynamic Cost Model

$$C_{total}(t) = C_{spread}(t) + C_{impact}(t) + C_{borrow}(t)$$

| Component | Formula | Description |
|-----------|---------|-------------|
| **Spread Cost** | `half_spread_l1_bps(t)` | Empirical, varies per ticker per bar |
| **Market Impact** | `κ × spread_std_1d(t)` | Spread-instability-scaled; κ pre-assigned by liquidity tier |
| **Borrow Cost** | `(rate / 10,000) / 365 × short_notional` | Daily accrual on short leg (50 bps/yr default; calendar-day convention) |

### Impact Coefficient (κ) Tiers
| Tier | Median L1 Spread | κ |
|------|-----------------|---|
| Tight (< 8 bps) | SPY, AAPL, MSFT | 0.3 |
| Medium (8–20 bps) | Most S&P 500 | 0.5 |
| Wide (> 20 bps) | VTRS, T, UBER | 0.8 |

### Report Sections
1. Executive Summary (verdict)
2. Empirical Spread Profile (median, p95, p99 by tier and regime)
3. Slippage Model Specification
4. The Before/After Table (Sharpe Gross vs. Net across all three regimes)
5. Cost Waterfall (where gross alpha goes)
6. Regime-Conditional Costs (Bear vs. Bull)
7. Spread-Vol Correlation (empirical proof spreads widen with volatility)
8. Kill Zone + Intraday Seasonality (time-of-day net alpha heatmap)
9. Negative Control under Dynamic Costs
10. Overfitting Diagnostics on Net Returns (DSR/PBO)
11. OAT Sensitivity (κ, borrow rate, spread level)
12. Honest Verdict

## Directory Structure

```
Week 5/
├── src/
│   ├── plan0_gateway/      # Orderbook ingestion, spread extraction, rolling stats
│   ├── plan1_cost_model/   # Three-component cost model + interface contract
│   ├── plan2_backtester/   # Per-trade cost application over Week 4 trade log
│   └── plan3_validation/   # 11 validation/report modules
├── workflows/              # Pipeline specs, methodology docs
├── reports/                # Generated performance reports
├── run_pipeline.py         # Master pipeline orchestrator
└── data/                   # Microstructure artifacts (gitignored)
```
