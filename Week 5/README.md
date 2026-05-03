# Microstructure Test on Backtest Engine

**Theme:** The Friction.

## Objective
The Week 4 strategy showed profitability under a flat 60 bps cost assumption. When we replace that with empirical bid-ask spreads that widen during spread instability, does alpha survive — or was it an illusion eaten by friction?

## Deliverable
A **"Net-of-Fees Performance Report"** that measures friction on the existing pairs trading strategy by replacing flat cost assumptions with empirical bid-ask spreads derived from order book data.

## Scope Boundary
This project **measures** friction on the existing strategy. We do **NOT** modify the strategy itself (no sizing changes, no new entry/exit gates, no regime suppression). The system uses the exact same signals and same positions, but implements a dynamic, microstructure-aware cost model.

## Data Profile
- **File:** `orderbook.parquet` (213M rows)
- **Frequency:** 1-minute (dominant; some 2–5 min gaps)
- **Tickers:** 504 (2022-01) → 526 (2026-03)
- **Date Range:** 2022-01-03 09:00 to 2026-03-19 23:59
- **Structure:** 3-level limit order book (`L1/L2/L3` bid/ask price + size)
