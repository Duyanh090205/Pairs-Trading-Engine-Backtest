# 1987 Market Crash Analysis Plan

> [!NOTE]
> **Status:** COMPLETED. All acceptance criteria met. Script and outputs are finalized.

## 1. Practical Project Plan
**Objective:** Develop a Python script using pandas and matplotlib/plotly to process `1987_crash_market_data.csv`, plot the S&P 500 futures minute-by-minute data, and algorithmically (or visually) flag the exact moment the market "broke" (e.g., largest percentage drop, highest volatility spike, or specific known historical timestamp).
**Tools:** Python 3.x, `pandas` for data manipulation, `matplotlib` or `plotly` for visualization, Claude Code for iterative implementation.
**Scope:** Data ingestion, cleaning (handling "Halted" states), transformation, critical moment anomaly detection, and final visualization highlighting the break point.
**Next Step:** Antigravity to execute a skill search using chat-skill-look up before starting implementation.

## 2. Simple File Structure
```text
project_root/
│
├── data/
│   └── 1987_crash_market_data.csv    # Source data
│
├── analyze_crash.py                  # Single code file for data loading, analysis, and visualization
├── requirements.txt                  # Dependencies: pandas, matplotlib, plotly
├── README.md                         # Task summary for Claude
├── assumptions_to_verify.md          # Strict assumption verification list for Claude
├── failure_points.md                 # Failure points for Claude to watch out for
└── output/
    └── crash_plot.png                # Output visualization file
```

## 3. Checklist of Assumptions
*Moved to `assumptions_to_verify.md` for Claude Code.*

## 4. Step-by-Step Implementation Order
1. **Setup & Ingestion:** Write data loading logic in `analyze_crash.py` to read the CSV into a pandas DataFrame.
2. **Data Cleaning:** Convert `Timestamp` to datetime. Clean numeric columns and missing values.
3. **Anomaly Detection (The "Break"):** Compute rolling metrics on `SP500_Futures`. Find the timestamp corresponding to the minimum return or maximum volatility.
4. **Visualization:** Plot the cleaned `SP500_Futures` price series. Add a distinct marker/annotation at the identified "break" timestamp.
5. **Output:** Save the finalized chart to the `output/` directory as `crash_plot.png`.

## 5. Acceptance Criteria
- [x] Script runs end-to-end via `python main.py` without throwing exceptions.
- [x] `SP500_Futures` "Halted" and invalid data points are gracefully handled and don't break the plot.
- [x] The generated plot clearly shows the minute-by-minute trajectory of the S&P 500 Futures.
- [x] A vertical line or distinct annotation specifically highlights the exact timestamp the algorithms identified as the market break.
- [x] The identified timestamp aligns intuitively with the largest structural drop visualized.

## 6. Common Failure Points
*Moved to `failure_points.md` for Claude Code.*
