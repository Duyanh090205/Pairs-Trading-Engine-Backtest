# Notebook 01 — Data Profiling & Preparation: Methodology

## Purpose

This notebook transforms raw 1-minute OHLC (Open-High-Low-Close) equity data into a clean, aligned, log-transformed 5-minute panel ready for statistical cointegration testing. Think of it as building the "foundation" — if the data is dirty, every statistical test downstream will produce garbage results.

---

## Step 1: Configuration & Parameter Setup

### What the code does
The script defines all key parameters up-front as Python constants. Nothing is hard-coded deep inside functions — every threshold lives at the top of the file so it can be audited and changed in one place.

### Key parameters and why they were chosen

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SESSION_START` | 09:35 ET | The US stock market opens at 09:30, but the first 5 minutes are dominated by the "opening auction" — a chaotic period where market makers match overnight orders. Prices during this window are noisy and unreliable for mean-reversion analysis. |
| `SESSION_END` | 15:55 ET | The market closes at 16:00, but the last 5 minutes see a "closing auction" effect where index-tracking funds rebalance. Same problem — artificial price pressure, not real supply-demand. |
| `RESAMPLE_FREQ` | 5 minutes | Raw 1-minute bars contain too much microstructure noise (bid-ask bounce, stale quotes). Aggregating to 5-minute bars smooths this noise while still capturing intraday dynamics. |
| `OUTLIER_ZSCORE_THRESHOLD` | 10 | A return more than 10 standard deviations from the mean is almost certainly a data error (bad print, stock split not adjusted). Normal market moves rarely exceed 5-6 sigma even during crashes. |
| `UNIVERSE_CAP` | 50 | Limits the final universe to the 50 most liquid stocks. This keeps the number of pairs manageable (C(50,2) = 1,225) and ensures every stock has enough trading volume to actually execute a pairs trade. |

---

## Step 2: Universe Discovery

### What the code does
The function `discover_tickers()` scans 12 monthly folders (01 through 12) of CSV files. Each file is named like `AAPL_2022-01-03.csv`. The function extracts the ticker symbol from each filename.

### The "12-month rule"
A ticker must appear in **all 12 months** to be included. If a stock was listed in March but delisted in September, or only started trading in July, it gets excluded entirely.

**Why this matters:** Cointegration testing requires a continuous, unbroken time series. If Stock A has data for January–December but Stock B only has March–November, their "aligned" series would only cover 9 months. Worse, the missing periods might coincide with important market events, biasing the test results. The 12-month rule eliminates this problem at the source.

### Algorithm
```
for each month folder (01..12):
    extract all ticker names from filenames
    store as a set

candidate_universe = intersection of all 12 sets
```
This set intersection ensures only tickers present in every single month survive.

---

## Step 3: Data Loading

### What the code does
For each candidate ticker, `load_ticker_data()` reads all its daily CSV files across all 12 months and concatenates them into one DataFrame.

### Timestamp handling (critical detail)
The raw CSV files store timestamps as **nanosecond UTC integers** in the `window_start` column. The code converts these through a precise pipeline:

1. `pd.to_datetime(window_start, unit='ns', utc=True)` — parse nanoseconds into UTC datetime
2. `.dt.tz_convert('US/Eastern')` — convert to Eastern Time (automatically handles EST/EDT daylight saving transitions)
3. Set as the DataFrame index and sort chronologically

**Why UTC → ET matters:** The session filter (09:35–15:55) is defined in Eastern Time. If you filtered on UTC timestamps, you'd accidentally include pre-market or post-market data during parts of the year when the EST/EDT offset changes.

### Duplicate handling
If two rows have the same timestamp for the same ticker, the code keeps the **last** one and drops earlier duplicates. This follows the principle that later data corrections supersede earlier ones.

### Session filtering
The function `filter_session()` applies a simple mask:
```python
mask = (times >= 09:35) & (times <= 15:55)
```
Any data point outside this window (pre-market, opening auction, closing auction, after-hours) is discarded.

---

## Step 4: Per-Ticker Quality Profiling & Screening

### What the code does
For every loaded ticker, `compute_ticker_profile()` calculates a set of quality metrics. Then four hard screening filters are applied sequentially.

### Quality metrics computed

| Metric | How it's calculated | What it measures |
|--------|-------------------|-----------------|
| `completeness_pct` | `actual_minutes / (trading_days × 381) × 100` | Data coverage. A perfect ticker during session hours has 381 one-minute bars per day (from 09:35 to 15:55 inclusive). If a ticker only has 300 bars on most days, something is wrong. |
| `median_close` | Median of all close prices | Central tendency of the stock price. Used to filter out penny stocks. |
| `avg_daily_dollar_volume` | Average of (daily sum of `close × volume`) | Liquidity measure. A stock trading $50M/day is very liquid; a stock trading $100K/day would have massive slippage in a real pairs trade. |
| `zero_return_pct` | Percentage of consecutive minutes where `close` didn't change | Staleness detector. If a stock's price is unchanged for 50%+ of its bars, it's likely illiquid — the "price" is just a stale quote, not real trading. |

### The four hard screens

Each ticker must pass ALL four tests. Failing any one means immediate exclusion.

1. **Median price ≥ $5.00** — Eliminates penny stocks. These have extreme percentage volatility, wide bid-ask spreads, and are often subject to manipulation. They would poison cointegration tests.

2. **Average daily dollar volume ≥ $1,000,000** — Ensures the stock is liquid enough to actually trade. A pairs strategy requires buying one stock and shorting another simultaneously. If either side is illiquid, execution costs destroy any statistical edge.

3. **Completeness ≥ 90%** — Ensures the data is sufficiently dense. A ticker missing 20% of its expected bars might have data feed problems or frequent trading halts, both of which would create artificial gaps in the spread.

4. **Zero-return fraction < 50%** — Catches "zombie tickers" — stocks that technically have data but aren't actually trading. If the price doesn't move for half the day, the stock is too illiquid for pairs trading.

### Universe cap
If more than 50 tickers pass all screens, the code sorts by `avg_daily_dollar_volume` descending and keeps only the top 50. This ensures the final universe contains the most tradable stocks.

---

## Step 5: Outlier Treatment (Z-Score Based)

### The problem
Raw minute-level data sometimes contains erroneous prices — a stock might show a close of $0.01 due to a bad print, or $9,999 due to a data feed glitch. Even one such outlier can wreck a cointegration test.

### The algorithm

For each surviving ticker:

1. **Compute minute-level returns:**
   ```
   R_t = (P_t - P_{t-1}) / P_{t-1}
   ```
   But skip returns that cross day boundaries (the first bar of each day has no meaningful "previous bar" since overnight gaps are structural, not outliers).

2. **Compute the z-score for each return:**
   ```
   z_t = |R_t - mean(R)| / std(R)
   ```
   where `mean(R)` and `std(R)` are computed over the entire sample for that ticker.

3. **Flag outliers:** Any bar where `z_t > 10` is flagged. A 10-sigma return is so extreme that it's virtually impossible under any reasonable model — it's almost certainly a data error.

4. **Patch flagged prices:** Set the close price to `NaN`, then forward-fill with a limit of 1 bar. This means `P_t = P_{t-1}` (the previous price is carried forward). The limit of 1 prevents long stretches of fake flat prices.

5. **Safety valve:** If any ticker has more than 1.0% of its bars flagged as outliers, the entire ticker is **removed** from the universe. Rationale: if a stock has hundreds of bad prices, the data quality is fundamentally unreliable and patching won't fix it.

### Global budget check
After processing all tickers, the code checks whether total outliers across the entire universe exceed 0.5% of all data points. If so, it prints a warning — this would suggest a systematic data quality issue rather than isolated glitches.

---

## Step 6: Log Transform & Resampling

### Log transform
Every surviving close price is converted to its natural logarithm:
```
L_t = ln(P_t)
```

**Why log prices?** Two critical reasons:

1. **Scale invariance:** A $1 move on a $10 stock (10%) and a $1 move on a $500 stock (0.2%) are fundamentally different. Log-transforming makes price movements proportional. The first difference of log prices (`ln(P_t) - ln(P_{t-1})`) approximately equals the percentage return.

2. **Cointegration interpretation:** When we later find that `ln(P_A) - β × ln(P_B)` is stationary, we're saying the two stocks move together in *percentage terms*, which is economically meaningful. Without logs, we'd be testing whether dollar-value differences are stationary, which is meaningless when comparing a $50 stock to a $500 stock.

### Resampling to 5-minute bars
After log-transforming, the 1-minute log prices are resampled to 5-minute frequency using the **last** observation in each 5-minute window:
```python
log_5min = df['log_close'].resample('5min').last()
```
The "last" method is chosen because it represents the most recent transaction price within each window — the best estimate of the "true" price at that moment.

### Time alignment (inner join)
The final and most consequential step: all tickers' 5-minute log-price series are merged into a single DataFrame (a panel), and any row containing even a single NaN is dropped.

```python
log_price_panel = pd.DataFrame(resampled).dropna()
```

This produces a **perfectly rectangular matrix** with dimensions `T × N` (timestamps × tickers) and zero missing values. Every ticker has a value at every timestamp.

**Why this is non-negotiable:** The cointegration test in Notebook 02 compares two columns of this matrix. If they had different lengths or misaligned timestamps, the OLS regression would be meaningless. The inner join guarantees perfect alignment at the cost of dropping some timestamps where any ticker had missing data.

---

## Step 7: Output Artifacts

The code saves three Parquet files:

| File | Contents | Purpose |
|------|----------|---------|
| `log_prices_5min.parquet` | The main `T × N` panel of aligned 5-minute log prices | Primary input for Notebook 02 |
| `universe_metadata.parquet` | Per-ticker quality metrics (completeness, dollar volume, outlier counts, etc.) | Audit trail for the screening process |
| `universe_completeness.parquet` | Summary counts at each filtering stage (how many tickers at each step) | Reproducibility documentation |

---

## Step 8: Validation

Before finishing, the code reloads the saved Parquet file and runs six assertions:

1. **Shape check** — reloaded data matches the in-memory panel
2. **No duplicate timestamps** — index is unique
3. **No NaN values** — panel is fully populated
4. **All columns are float64** — no mixed types
5. **Monotonic index** — timestamps are in chronological order
6. **Log-price sanity** — exponentiated values (`exp(log_price)`) fall in a reasonable range ($1–$10,000), confirming no extreme data corruption survived
