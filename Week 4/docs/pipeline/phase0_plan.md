# Phase 0 Plan — Data Quality Gateway

## Objective
Build a standalone, vectorized data gateway that all downstream phases must read from. Raw CSVs in → validated parquet out. No joins here.

## Input
`data/minute_ohlc_flatfiles/{TICKER}_{YYYY-MM-DD}.csv`  
Schema: `ticker,volume,open,close,high,low,window_start,transactions`  
`window_start` = nanosecond UTC integer

## Reuse from Prior Weeks
| Code | Source | Adaptation needed |
|---|---|---|
| CSV loading loop | `Week 1/notebooks/01_data_profiling.py` | None |
| UTC→ET conversion | `Week 1/notebooks/01_data_profiling.py` | `pd.to_datetime(window_start, unit='ns', utc=True).dt.tz_convert('US/Eastern')` |
| Outlier detection \|Z\|>10 | `Week 1/notebooks/01_data_profiling.py` | Use log returns not pct_change; rolling 1-day window |
| OHLC assertions | `Week 3/scripts/utils.py` | Extend with volume-zero session check |

## File to Create
`src/phase0_data_gateway/gateway.py`

## Implementation Steps

### Step 1 — Load & Convert Timestamps
```python
# Adapt from Week 1 Notebook 01
df['timestamp_et'] = (
    pd.to_datetime(df['window_start'], unit='ns', utc=True)
    .dt.tz_convert('US/Eastern')
)
```
One file per ticker per day. Concatenate all days for a ticker into a single DataFrame sorted by timestamp.

### Step 2 — Session Filter (after timezone conversion)
```python
SESSION_P1 = ('09:35', '15:55')   # Phase 1 output
SESSION_P2 = ('09:30', '15:59')   # Phase 2 output
```
Apply AFTER timezone conversion, not before.

### Step 3 — Outlier Treatment on 1-min Returns (BEFORE resample)
```python
return_t = np.log(close_t / close_t.shift(1))
rolling_mean = return_t.rolling(window=390, min_periods=1).mean()   # 390 = 1 day
rolling_std  = return_t.rolling(window=390, min_periods=1).std()
Z_t = (return_t - rolling_mean) / rolling_std

bad_mask = Z_t.abs() > 10
close_t[bad_mask] = np.nan
close_t = close_t.ffill(limit=1)

outlier_fraction = bad_mask.sum() / len(return_t)
if outlier_fraction > 0.01:
    drop ticker entirely
```

### Step 4 — Resample (after outlier treatment)
```python
# Phase 1 output
df_5min_p1 = df.resample('5min').agg({'close': 'last', 'volume': 'sum', ...})

# Phase 2 output
df_1min_p2 = df   # raw 1-min, no resample
```
Apply session filter BEFORE resample so out-of-session bars are excluded.

### Step 5 — Hard Assertions (fail-fast, raise ValueError on violation)
Adapt `Week 3/scripts/utils.py` assertions:
1. Monotonic timestamps per ticker
2. No duplicate `(ticker, timestamp)` rows
3. OHLC valid: `O ∈ [L, H]`, `C ∈ [L, H]`, `H ≥ L`
4. Non-negative price and volume
5. Volume not all-zero within a session (unless market-halt day — cross-reference cross-ticker freeze flag)

### Step 6 — Bad-Data Flags (log only, do NOT drop rows)
Write to `meta_flags.parquet`:
- **Stale price:** ≥10 consecutive identical close values per ticker
- **Intra-session gap:** bar that should exist within session but is missing
- **Cross-ticker freeze:** ≥30% of tickers have identical close at same timestamp → log as market-halt day
- **Volume-price coherence:** `|vol Z| > 10` AND `|return Z| < 1` → suspect tick

### Step 7 — Write Parquet Output
```python
# Phase 1: log-prices + volume, 5-min, 09:35–15:55 ET
out['log_close'] = np.log(out['close'])
out.to_parquet('data/validated/5min_phase1.parquet')

# Phase 2: OHLCV, 1-min, 09:30–15:59 ET
out.to_parquet('data/validated/1min_phase2.parquet')

# Flags
flags_df.to_parquet('data/validated/meta_flags.parquet')
```

## Skills to Invoke
- After implementing `gateway.py`, run **`/simplify`** to review for vectorization quality and redundant logic.
- After Phase 0 outputs are confirmed, run **`/init`** to generate a CLAUDE.md documenting the repo structure (so future sessions know what's built).

## Smoke Test (before hard stop)
1. Run gateway on 5 tickers (AAPL, MSFT, JPM, XOM, CVNA) for 2022-01-03 to 2022-01-07
2. Assert 3 parquet files created in `data/validated/`
3. Spot-check 1 ticker: verify timestamps are in ET, session is 09:35–15:55, no bars outside session
4. Verify outlier fraction < 1% for AAPL (expect 0 outliers for liquid names)
5. Check OHLC assertion passes (no failures expected for AAPL)

---
## ⛔ HARD STOP — Review Before Phase 1

**Before proceeding to Phase 1, verify:**
- [ ] `data/validated/5min_phase1.parquet` exists and has correct shape
- [ ] `data/validated/1min_phase2.parquet` exists and has correct shape
- [ ] `data/validated/meta_flags.parquet` exists
- [ ] Timestamps are US/Eastern (not UTC) in parquet
- [ ] No bars outside session boundaries exist in either output
- [ ] Outlier treatment logs: at least 1 ticker dropped if fraction > 1% (or confirm none qualify)
- [ ] All 5 hard assertions pass without raising
- [ ] `/simplify` has been run and suggestions addressed
- [ ] `/init` has been run to generate CLAUDE.md

**Signal to proceed:** User explicitly types "Phase 0 approved, proceed to Phase 1"
