# Week 6 — Cloud Deployment (Paper Trading)

**Theme:** Going Live.

## Objective

Deploy the quantitative pairs trading strategy into a live paper-trading environment, ensuring mathematical continuity from the backtest and demonstrating robust infrastructure resilience under real-world conditions.

## Deliverable

A **Live Trading Dashboard** (web-hosted or terminal UI) displaying real-time P&L, open positions, API health, and a Drift Monitor comparing live execution costs to Week 5's dynamic model predictions. The system must demonstrate graceful handling of an intentional 5-minute API disconnect.

## Scope

This week integrates everything built in Weeks 4 and 5 into a live execution layer:
- **From Week 4:** Core strategy (Kalman-filtered cointegration signals, walk-forward architecture, threshold rebalances).
- **From Week 5:** Dynamic cost model (empirical spread + instability-scaled impact + borrow). The live engine computes the "Drift" — the delta between actual paper trading execution cost and predicted cost.

## Data

- **Live Source:** Real-time 1-minute pricing + L1 order book data via brokerage API (Alpaca or Interactive Brokers Paper Trading).
- **Session:** 09:30–15:59 ET (matching Weeks 4–5 execution window).
- **Kalman State:** Initialized from the latest monthly fold's formation parameters, updated bar-by-bar in real time.

## Method

### 1. API Authentication & WebSocket Module
- Secure brokerage connection (Alpaca / Interactive Brokers Paper Trading).
- Live 1-minute OHLCV + L1 order book data ingestion.
- Simultaneous order routing for both pair legs.

### 2. Resilience & Reconnection Handler
- **Disconnect Simulation:** Intentional 5-minute API disconnect.
- **Fault Tolerance:** Detect dropped WebSocket, pause state updates, queue auto-reconnect.
- **State Preservation:** Recover Kalman state `θ̂(t)` and position inventory without crashes or duplicate trades.

### 3. Live Execution Engine
- Run the existing Numba `@njit` engine on real-time data (not static Parquet files).
- Strict 09:30–15:59 ET session adherence with end-of-day flattening rules.

### 4. Drift Monitor
- Real-time cost decomposition: realized execution prices vs. theoretical mid-prices.
- `Drift = Realized Trade Cost − Predicted Week 5 Dynamic Cost`.
- Anomaly alerting if live slippage consistently exceeds model predictions.

### 5. Live Trading Dashboard
- Live P&L (Gross vs. Net-of-Fees).
- Open pair positions & current Kalman Z-scores.
- API connection status (heartbeat / disconnect events).
- Real-time Drift visualization (live slippage vs. backtest prediction).

## Directory Structure

```
Week 6/
└── pipeline_week6.md   # Architecture specification (implementation pending)
```

> **Status:** Architecture and specification phase. Implementation pending completion of Weeks 4–5 outputs.
