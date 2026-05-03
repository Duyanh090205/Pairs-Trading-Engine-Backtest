# Week 6: The Engine — The "Cloud" Deployment (Paper Trading)

## Objective
**Theme:** Going Live.
The goal is to deploy the quantitative pairs trading strategy into a live paper-trading environment, ensuring mathematical continuity from the backtest and demonstrating robust infrastructure resilience.

## Scope & Integration Context
The Week 6 deployment is the culmination of the prior phases, acting as the live execution layer for the models built in Weeks 4 and 5.

* **From Week 4 (The Signal Engine):**
  * **Core Strategy:** Cointegration pairs trading using a 2D Kalman filter for signal generation (spread state `[α,β]`). 
  * **Architecture:** We are executing the out-of-sample trading window of the latest monthly fold in a rolling walk-forward architecture. The live engine must ingest 1-minute OHLCV data, update the Kalman state `θ̂(t)` bar-by-bar, and execute threshold rebalances using the frozen parameters (`δ`, `R`, Kalman init) calibrated during the formation window.
* **From Week 5 (The Friction Model):**
  * **Drift Baseline:** Week 5 established a dynamic cost model comprising empirical Spread Cost (half-spread), Market Impact (instability-scaled `κ × spread_std_1d`), and Borrow Cost. 
  * **Strategist Role:** The live deployment must compute the "Drift"—the delta between the actual paper trading execution cost (realized slippage) and the predicted cost from the Week 5 dynamic model.

## Detailed Elements Needed to Deliver

### 1. API Authentication & WebSocket Module (Dev Role)
* **Broker Integration:** Secure connection to a Brokerage API (e.g., Alpaca or Interactive Brokers Paper Trading).
* **Data Stream:** Ingestion of live 1-minute pricing and real-time L1 order book data to feed the Kalman state machine and calculate dynamic execution costs.
* **Order Routing:** Logic to submit live market or limit orders for both legs of the pair simultaneously.

### 2. Resilience & Reconnection Handler (AI Audit Target)
* **Disconnect Simulation:** The system will be subjected to an intentional 5-minute API disconnect.
* **Fault Tolerance:** The bot must detect the dropped WebSocket/REST connection, pause state updates, and queue a graceful auto-reconnect.
* **State Preservation:** Upon reconnection, the bot must recover the Kalman state `θ̂(t)` and active position inventory without crashing or making duplicate trades.

### 3. Live Execution Engine
* **Integration:** Running the existing Numba `@njit` engine logic on real-time data instead of static parquet files.
* **Execution Alignment:** Strict adherence to the 09:30–15:59 ET trading session, including end-of-day flattening (EOS) rules if applicable.

### 4. Drift Monitor (Strategist Role)
* **Real-time Cost Decomposition:** Continuously tracking the realized execution prices against the theoretical mid-prices.
* **Comparison:** Live calculation of `Drift = Realized Trade Cost - Predicted Week 5 Dynamic Cost`.
* **Alerting:** Flagging anomalies if the live realized slippage consistently exceeds the dynamic model's predictions, indicating structural execution leakage.

### 5. Live Trading Dashboard
* **Visual Interface:** A web app or detailed terminal UI displaying the system's live operational health.
* **Metrics Displayed:**
  * Live P&L (Gross vs. Net-of-Fees)
  * Open Pair Positions & current Kalman Z-scores
  * API Connection Status (Heartbeat / Disconnect events)
  * Real-time Drift (Live Slippage vs. Backtest Prediction)

### 6. Final Deliverable format
* **Submission:** A Live Trading Dashboard URL (if web-hosted) + a link to a screenshot (Imgur, cloud storage, etc.).
* **Grading Criteria:** The AI will evaluate the visual health of the dashboard from the screenshot. The bot must strictly demonstrate handling an API disconnect and reconnect gracefully without crashing.
