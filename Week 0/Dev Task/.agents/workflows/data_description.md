# 1987 Crash Market Data - Dataset Context & Anomalies

## 1. File Overview
- **File Location:** `data/1987_crash_market_data.csv`
- **Granularity:** 1-minute intervals.
- **Coverage:** October 16, 1987 to October 21, 1987.
- **Target Asset:** `SP500_Futures` (This is the exclusive focus of the current analysis task).

## 2. The "Halted" Anomaly (CRITICAL)
In the `SP500_Futures` column, there are roughly 150 rows where the numerical price is replaced by the string text `"Halted"`. 

### When do these occur?
- **October 19, 1987:** Exactly 60 minutes of consecutive "Halted" entries (representing the initial circuit breaker/trading halt implemented due to extreme order imbalances).
- **October 20, 1987:** Exactly 90 minutes of consecutive "Halted" entries.

### Why this is structurally dangerous for coding:
If an analyst or script naively handles these "Halted" strings by coercing them to `NaN` and then blindly applying **forward-fill (`ffill()`)**, it will artificially create 60–90 minutes of **perfectly flat prices (0% returns)**. 
- When calculating rolling metrics (like standard deviation, limits, or autocorrelation), these artificial blocks of 0-variance will fundamentally corrupt the math. Dividing by near-zero variance will cause formulas to output `NaN`, `infinity`, or wildly skewed statistical distributions exactly when the system resumes.

### The Required Handling Protocol:
1. Identify the specific indices of the "Halted" periods and mark them.
2. Ensure that any rolling-window calculations (such as Autocorrelation, Hurst Exponent, or other microstructure signals) **bridge over** these halts or pause calculations entirely during these intervals.
3. Do not allow 0% forward-filled artifact data to feed into historical baseline computations.

## 3. Session Boundaries & Gaps
- The dataset spans mathematically continuous 24-hour clocks, but active trading does not occur overnight or on weekends.
- **Overnight Gaps:** Calculating timeseries momentum or rolling changes across an overnight boundary (e.g., from 4:00 PM Friday to 9:00 AM Monday) will produce a single massive artificial spike reflecting the overnight gap rather than an intraday anomaly.
- **Constraint:** All rolling computations must rigidly reset at start-of-session boundaries. 

## 4. Current Task Limitations
- **No Cash Data:** We do not possess `SP500_Cash` (meaning direct "Basis Spread" calculations are impossible).
- **No Liquidity Data:** We do not possess `Volume`, Bid/Ask spreads, or Depth metrics.
- **Mandate:** Any microstructure breakdown indicator must be computable purely via the raw `SP500_Futures` sequence.
