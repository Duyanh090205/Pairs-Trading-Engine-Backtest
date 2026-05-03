# Project Context: 1987 Crash Backtest

You are an expert quantitative developer. Your task is to build a backtesting simulation evaluating a "buy the dip" trading strategy during the 1987 stock market crash.

## 1. Goal
Simulate the strategy: "Buy when the price drops 5%" using minute-level market data from the 1987 crash (`data/1987_crash_market_data.csv`).
Determine if the bot effectively goes "bankrupt" before the day ends, output a visual autopsy of the crash vs. portfolio value, and propose a stop-loss mechanism.

## 2. Folder Structure Strict Requirements
Please create and adhere strictly to this structure:
```text
project_root/
├── data/
│   └── 1987_crash_market_data.csv    (Assume this exists; do not create)
├── src/
│   ├── main.py                       (Entry point to run the whole backtest)
│   ├── data_loader.py                (Reads and cleans the CSV data)
│   ├── backtest.py                   (Execution and portfolio math)
│   └── visualizer.py                 (Generates the required plot)
├── outputs/
│   ├── crash_autopsy.png             (Final chart output)
│   └── autopsy_report.txt            (Final text report output)
└── requirements.txt                  (List dependencies like pandas, matplotlib)
```

## 3. Implementation Steps
1. **Data Ingestion (`src/data_loader.py`)**: 
   - Load the CSV. Parse timestamps and sort the data chronologically by minute.
2. **Strategy Engine (`src/backtest.py`)**:
   - Loop through the data tick-by-tick (or use vectorized pandas if it does not introduce look-ahead bias).
   - Trigger a buy order if `current_price <= reference_price * 0.95`.
3. **Bankruptcy Evaluation**:
   - Track portfolio value (Cash + Holdings * current_price) at every minute.
   - Flag the exact timestamp if the portfolio value reaches $0 or goes negative (evaluating margin/bankruptcy).
4. **Stop-Loss Calculation**:
   - Retrospectively calculate a specific trailing or fixed percentage stop-loss that would have preserved capital, keeping the portfolio from fully going bankrupt.
5. **Visualization (`src/visualizer.py`)**:
   - Generate a single readable plot showcasing the Market Price on one axis (with prominent markers for every "Buy" execution) and the Portfolio Value on a secondary axis (showing its decline).
   - Export to `outputs/crash_autopsy.png`.
6. **Delivery (`src/main.py`)**:
   - Orchestrate these components so running `python src/main.py` performs the entire pipeline, saves the graph, and writes `outputs/autopsy_report.txt` containing the initial/final capital, bankruptcy status, and the Stop Loss proposal.

## 4. Key Assumptions (Apply these constraints directly)
- **Initial Capital**: $100,000.
- **Reference Price**: Calculate the 5% drop relative to the maximum price achieved so far that day (a rolling all-time high for the day).
- **Position Sizing**: The bot deploys exactly 25% of its *remaining cash* on every valid 5% dip trigger.
- **Bankruptcy Definition**: If portfolio value drops below a 90% drawdown (i.e. <$10,000), consider it functionally "bankrupt".
- **Friction**: Assume $0 trading fees and zero slippage for MVP simplicity.

## 5. Quality Control Checklist (Verify before finalizing)
- [ ] Ensure the strategy does not use future data to make trading decisions (no look-ahead bias).
- [ ] Check that `crash_autopsy.png` successfully saves to the `outputs/` folder.
- [ ] Ensure `outputs/autopsy_report.txt` clearly states "Bankrupt: Yes/No", the time of death (if any), and outlines the 1-2 sentence Stop Loss Proposal.
- [ ] Prove that running `python src/main.py` executes start-to-finish without user input.
