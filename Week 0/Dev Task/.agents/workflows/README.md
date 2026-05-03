# Task Summary

> [!NOTE]
> **Status:** COMPLETED. The analysis script is implemented in `analyze_crash.py` and outputs are available in the `outputs/` directory.

Develop a Python script to process `1987_crash_market_data.csv`, plot the S&P 500 futures minute-by-minute data, and algorithmically (or visually) flag the exact moment the market "broke". The solution should reside in a single Python file (`analyze_crash.py`) and output the resulting chart to an `output/` directory.

## Assumption Verification Checklist
**Claude Code MUST explicitly verify all the assumptions below from the CSV data BEFORE writing any code:**

- [ ] **Data Types:** Verify the exact format of the `Timestamp` column so it can be successfully parsed into datetime objects.
- [ ] **Missing/Anomalous Data:** Identify the presence and format of "Halted" string values (or similar anomalies) across all relevant columns (DOW_Futures, SP500_Futures, etc.).
- [ ] **Timezone/Market Hours:** Determine if the data spans 24 hours or just market hours. This dictates whether overnight gaps exist and how they affect the logic.
- [ ] **Data Granularity:** Verify if there is exactly one row per minute, and detect if there are any unexpectedly large gaps in the timespan.
