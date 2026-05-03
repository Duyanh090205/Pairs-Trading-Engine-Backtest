# Implementation Checklist — Pairs Trading Cointegration Notebook

Source of truth: `.agents/workflows/cointegration_methodology_spec.md`

---

## 1. Notebook Build Order

Two notebooks, built sequentially. Notebook 01 must complete before 02 can start.

### Notebook 01: `01_data_profiling.py` — Data Ingestion, Cleaning, Universe Selection

**Build in this order:**
1. Raw data loader (parse CSVs → single DataFrame per ticker)
2. Timestamp conversion (nanosecond UTC → ET datetime)
3. Trading hours filter (keep 9:35–15:55 ET)
4. Per-ticker profiling (price stats, volume stats, completeness, staleness)
5. Universe screening (apply hard thresholds, output surviving ticker list)
6. Outlier treatment on surviving tickers
7. Log transform
8. Resample to 5-min bars
9. Save cleaned DataFrame to `data/intermediate/`
10. Data audit summary table + diagnostic plots

### Notebook 02: `02_cointegration_scan.py` — Pair Generation, Testing, Filtering, Report

**Build in this order:**
1. Load cleaned 5-min log-price data from `data/intermediate/`
2. Sector mapping dict + pair generation (all C(N,2))
3. Cointegration scan loop (coint() + OLS per pair)
4. BH-FDR correction on all raw p-values
5. Half-life computation for BH-passing pairs
6. Hard filters (sequential)
7. Economic logic annotation for survivors
8. Ranking (composite or simplified fallback)
9. Output tables (full scan, filtered, top pairs, rejection summary)
10. Report plots (p-value histogram, funnel, spread charts)

---

## 2. Deterministic Implementation Checklist

### Phase A: Data Loading (Notebook 01)

**Objective:** Load all 1-min CSVs into a single per-ticker time series of close prices.

**Inputs:** `data/{MM}/TICKER_YYYY-MM-DD.csv` files (131K+ files, 12 month folders)

**Outputs:** Dict or DataFrame mapping `{ticker: Series of close prices with ET DatetimeIndex}`

**Key implementation details:**
- CSV schema: `ticker,volume,open,close,high,low,window_start,transactions`
- `window_start` is nanoseconds since Unix epoch (UTC). Convert: `pd.to_datetime(window_start, unit='ns', utc=True).tz_convert('US/Eastern')`
- Use `close` column as the price
- Also retain `volume` column — needed for dollar volume screening
- Load strategy: iterate month folders, glob `*.csv`, read and concat per ticker

**Universe completeness rule:**
- Primary rule: use tickers present in ALL 12 months (317 tickers). This ensures full-year coverage for cointegration testing
- **Must report impact:** Log and print: (a) total unique tickers across all months, (b) how many tickers were excluded solely for not appearing in all 12 months, (c) list the excluded tickers by name
- **Fallback:** If after ALL screening steps (Phase C) the universe drops below 30 tickers, relax to "present in at least 90% of trading days" (≈227 of 252 days). Document this relaxation in the notebook

**Edge cases:**
- Duplicate timestamps within a single ticker-day file: drop duplicates, keep last
- Files may contain pre/post-market rows: filter AFTER timestamp conversion

**Test before moving on:**
- Pick 3 tickers (AAPL, XOM, a small one). Verify row count ≈ 390 rows/day × 252 days ≈ 98K. Verify first timestamp is 2022-01-03 ~9:30 ET. Verify last is 2022-12-30 ~16:00 ET

---

### Phase B: Time Filtering (Notebook 01)

**Objective:** Restrict to regular trading hours minus auction periods.

**Inputs:** Raw per-ticker DataFrames from Phase A

**Outputs:** Same DataFrames filtered to 9:35–15:55 ET only

**Key implementation details:**
- After converting to ET, filter: `(time >= 09:35) & (time <= 15:55)`
- This drops pre-market, after-hours, first 5 min of open, last 5 min before close
- Expected: ~370 usable minutes per day (down from ~390)

**Edge cases:**
- Half-days (day before Thanksgiving, Christmas Eve, etc.): market closes early (1:00 PM ET). The time filter still works — it just produces fewer rows those days. Do NOT special-case these
- Some rows may fall exactly on boundaries (9:35:00, 15:55:00): use `>=` and `<=` (inclusive)

**Test before moving on:**
- For AAPL, verify no timestamps before 9:35 or after 15:55 ET. Count minutes per day — should be ~370 on full days

---

### Phase C: Universe Screening (Notebook 01)

**Objective:** Reduce 317 tickers to ~30-50 that pass quality filters.

**Inputs:** Time-filtered per-ticker DataFrames

**Outputs:** List of surviving tickers + Data Audit Summary table

**Screening rules (apply in order):**

| # | Filter | Threshold | How to compute |
|---|--------|-----------|----------------|
| 1 | Median close price | >= $5 | `ticker_series.median()` |
| 2 | Avg daily dollar volume | >= $1M | Per-day: `sum(volume * close)`, then `mean()` across days |
| 3 | Data completeness | >= 90% | `actual_minutes / expected_minutes` where expected ≈ 370 × trading_days |
| 4 | Zero-return minutes | < 50% | `(minute_returns == 0).mean()` |

**CRITICAL DATA DISCOVERY:** 317 tickers present all year. C(317,2) = 50,086 pairs — far too many. The universe screening must aggressively reduce this. With the $1M daily volume filter, expect roughly 150-200 survivors. C(200,2) = 19,900 — still too many.

**Solution (methodology-consistent):** After quality screening, if N > 50, apply a **deliberate universe cap**: select the top 50 tickers by average daily dollar volume. This gives C(50,2) = 1,225 pairs — manageable. If N <= 50 after screening, use all survivors.

*Justification:* The methodology says "target ~30-50 liquid, well-known stocks." Selecting top-50 by liquidity is deterministic, not heuristic, and directly implements the methodology's intent. Capping at 50 is a computational constraint, not a statistical one.

**Edge cases:**
- Tickers with zero volume on many days: caught by zero-return filter + dollar volume filter
- ETFs in the universe (XLE, XLF, SPY, etc.): keep them — they are valid for pairs trading
- Single-digit price stocks that dip below $5 intraday but have median >= $5: keep (median filter handles this)

**Test before moving on:**
- Print the Data Audit Summary table. Verify familiar liquid tickers (AAPL, MSFT, AMZN, XOM) survive. Verify penny stocks and illiquid tickers are removed. Count survivors — should be 30-50 range

---

### Phase D: Outlier Treatment + Log Transform + Resample (Notebook 01)

**Objective:** Clean prices, take logs, downsample to 5-min bars.

**Inputs:** Time-filtered data for surviving tickers only

**Outputs:** DataFrame with columns = tickers, index = 5-min ET timestamps, values = log(close)

**Implementation order:**
1. Compute minute returns per ticker: `ret = close.pct_change()`
2. Flag |z| > 10σ as outliers: `z = (ret - ret.mean()) / ret.std(); mask = abs(z) > 10`
3. Replace flagged close prices with NaN, then ffill (max 1 bar)
4. Apply log: `log_close = np.log(close)`
5. Resample to 5-min: `log_close.resample('5min').last()` — take last value in each 5-min bucket
6. Drop any remaining NaN rows (should be minimal after ffill)

**Outlier transparency requirements (MANDATORY):**
- For each ticker, log: `n_outliers_flagged`, `pct_outliers` (= n_outliers / n_total_minutes × 100)
- Print a summary table: ticker, n_outliers_flagged, pct_outliers
- **Hard rule:** If any ticker has pct_outliers > 1%, REMOVE that ticker from the universe entirely rather than silently patching hundreds of points. Document the removal
- The outlier treatment must NEVER run silently — the notebook must contain a visible cell showing exactly how many prices were modified per ticker
- After treatment, assert: total modified points across all tickers < 0.5% of total dataset. If exceeded, flag for investigation

**Edge cases:**
- First minute of each day has no return (NaN from pct_change): skip in z-score calc
- A ticker might have entire days missing: these naturally become NaN rows in the 5-min resample. Forward-fill handles this across the resample but NOT across days — if an entire day is missing, leave NaN
- After resample, expected: ~74 bars/day × 252 days ≈ 18,648 rows per ticker
- Log of zero or negative: impossible after $5 filter, but assert `(close > 0).all()` before log

**Test before moving on:**
- Verify DataFrame shape: ~18,648 rows × N_tickers columns
- Verify no NaN in log-price columns (or < 0.1%)
- Spot-check: `np.exp(log_close)` should approximately match original close prices
- Verify outlier summary table shows < 1% modification rate for all surviving tickers

---

### Phase E: Save Intermediate Output (Notebook 01)

**Objective:** Persist cleaned data so Notebook 02 can load it without re-running Notebook 01.

**Inputs:** 5-min log-price DataFrame

**Outputs:**
- `data/intermediate/log_prices_5min.parquet` (the main data artifact)
- `data/intermediate/universe_metadata.parquet` (ticker list, sectors, screening stats)

**Key details:**
- Use parquet format (preserves dtypes, faster than CSV, smaller)
- Also save the surviving ticker list and the Data Audit Summary as a separate file for reference

**Test before moving on:**
- Reload the parquet and verify shape, dtypes, first/last timestamps match

---

### Phase F: Pair Generation (Notebook 02)

**Objective:** Generate all C(N,2) pairs, tag with sector info.

**Inputs:** Surviving ticker list + GICS sector mapping dict

**Outputs:** DataFrame of pair metadata

**Key implementation details:**
- Hardcoded sector dict in notebook — map each surviving ticker to its GICS sector
- Use `itertools.combinations(tickers, 2)` to generate all pairs
- For each pair: assign `within_sector = (sector_A == sector_B)`
- Assign `pair_id = f"{ticker_a}-{ticker_b}"` (alphabetical order, consistent)

**Sector dict requirements:**
- Define in a single, clearly labeled cell at the top of Notebook 02 (not scattered across cells)
- Must cover ALL possible surviving tickers — build for the full 317-ticker pool, not just a guess at who will survive
- Structure:
```
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer Discretionary',
    'XOM': 'Energy', 'CVX': 'Energy', 'JPM': 'Financials', ...
}
```
- **Hard assertion before pair generation:** `assert set(surviving_tickers).issubset(set(SECTOR_MAP.keys()))` — if any ticker is unmapped, the notebook stops with a clear error, never silently generates pairs with missing sectors

**Edge cases:**
- Tickers not in the dict: this is a bug. The assert above catches it. Fix by adding the missing ticker to the dict
- ETFs: map to their primary sector (XLE → Energy, XLF → Financials, SPY → Broad Market)

**Test before moving on:**
- Verify pair count = C(N,2). If N=50, expect 1,225 pairs
- Verify within_sector flag is correct for known pairs (XOM-CVX should be True)
- Verify no duplicate pairs, no self-pairs

---

### Phase G: Cointegration Scan Loop (Notebook 02)

**Objective:** Run coint() + OLS on every pair. This is the computational core.

**Inputs:** 5-min log-price DataFrame + pair metadata DataFrame

**Outputs:** Scan results DataFrame with one row per pair

**Per-pair workflow:**
```
For pair (A, B):
  1. Inner join log_A and log_B on timestamp index → aligned DataFrame
  2. If aligned length < 5000 (< ~3 months of 5-min bars): skip pair, record as "insufficient data"
  3. coint_t, pvalue, crit_values = coint(aligned_A, aligned_B, trend='c', autolag='aic', maxlag=30)
  4. model = OLS(aligned_A, add_constant(aligned_B)).fit()
  5. hedge_ratio = model.params[1]  (the slope, not the intercept)
  6. spread = aligned_A - hedge_ratio * aligned_B
  7. Store: pair_id, ticker_a, ticker_b, sector_a, sector_b, within_sector,
           coint_tstat, raw_pval, hedge_ratio, n_obs, spread (keep in memory for later use)
```

**Performance concern:** 1,225 pairs × coint() call each. Each coint() on ~18K rows with maxlag=30 takes ~0.1-0.5s. Total: ~2-10 minutes. Acceptable.

**Edge cases:**
- `coint()` may raise exceptions on degenerate data (constant series, too few obs): wrap in try/except, record as failed
- `pvalue` may return exactly 0.0 or 1.0: these are valid, store as-is
- `hedge_ratio` can be negative: store as-is, filter later
- Pairs where inner join loses > 20% of data: skip per methodology (record reason)

**Test before moving on:**
- Run on 5 known pairs first (XOM-CVX, AAPL-MSFT, JPM-BAC, a cross-sector pair, a likely-spurious pair)
- Verify XOM-CVX has low p-value, positive hedge ratio
- Verify cross-sector pair has high p-value
- Verify coint_tstat sign is negative (as expected for stationary residuals)

---

### Phase H: BH-FDR Correction (Notebook 02)

**Objective:** Apply Benjamini-Hochberg FDR to all raw p-values.

**Inputs:** Array of raw p-values from scan (one per pair)

**Outputs:** Array of adjusted p-values + boolean reject array

**Implementation:**
```
reject, pvals_adj, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
```

**Edge cases:**
- Pairs that were skipped (insufficient data) should NOT be included in the p-value array. Only include pairs that produced a valid p-value
- If all p-values > 0.05: reject array is all False. This is valid — report it
- NaN p-values: exclude from correction, then mark as non-rejected

**Test before moving on:**
- Verify `sum(reject)` is reasonable (expect 5-50 pairs for 1,225 tested)
- Verify adjusted p-values are >= raw p-values (BH only inflates)
- Verify the most significant raw p-value is still significant after adjustment

---

### Phase I: Half-Life Computation (Notebook 02)

**Objective:** Compute half-life of mean reversion for all BH-passing pairs.

**Inputs:** Spread series for each pair that passed BH-FDR

**Outputs:** Half-life in trading days for each pair

**Implementation:**
```
For each passing pair's spread S:
  1. spread_lag = S.shift(1)
  2. spread_diff = S.diff()
  3. Drop first row (NaN)
  4. OLS: spread_diff ~ spread_lag (with constant)
  5. lambda = slope coefficient (must be negative for mean reversion)
  6. half_life_bars = -ln(2) / lambda
  7. half_life_days = half_life_bars / 74  (74 five-min bars per trading day)
```

**Edge cases:**
- λ >= 0: spread is NOT mean-reverting. Set half_life = Inf or NaN. This pair will fail the half-life filter
- λ very close to 0: half-life will be enormous. Same outcome — filtered out
- λ very negative: half-life will be < 1 day. Store as-is, filter will catch it if < 5 days

**Test before moving on:**
- For a known good pair (XOM-CVX if it passed), verify half-life is in single-digit to double-digit days range
- Verify no negative half-lives (mathematically impossible if λ < 0)

---

### Phase J: Hard Filters (Notebook 02)

**Objective:** Apply the 4 hard filters sequentially, tracking how many pairs drop at each stage.

**Inputs:** Full scan results with BH-adjusted p-values and half-lives

**Outputs:** Filtered pairs DataFrame + rejection funnel counts

**Filter sequence:**
```
Stage 0: Total pairs tested = N_total
Stage 1: BH-FDR reject = True                         → N_after_fdr
Stage 2: Half-life in [5, 60] trading days             → N_after_hl
Stage 3: Hedge ratio β > 0                             → N_after_hr
Stage 4: Economic logic (1-sentence rationale exists)  → N_final

Funnel: [N_total, N_after_fdr, N_after_hl, N_after_hr, N_final]
```

**Fallback logic — precise trigger and labeling rules:**

Fallback is triggered AT MOST ONCE per filter. Once triggered, do not cascade further relaxations on the same filter.

```
FALLBACK_USED = False

# Check after Stage 3 (β > 0):
if N_after_hr < 10:
    Re-run Stage 2 with half-life [3, 90] instead of [5, 60]
    FALLBACK_USED = True
    # Re-apply Stage 3 on the new set

# Check again after re-application:
if N_after_hr < 5 AND FALLBACK_USED:
    Re-run Stage 1 with q=0.10 instead of 0.05
    # Re-apply Stages 2 and 3 from scratch with relaxed FDR
```

**Labeling requirement:** The notebook MUST contain a clearly visible markdown cell or print statement:
- If no fallback used: `"MAIN RESULT: All filters applied at primary thresholds (BH q=0.05, HL=[5,60])."`
- If fallback used: `"SENSITIVITY RELAXATION APPLIED: [describe which threshold was relaxed and why]. Results below reflect relaxed thresholds and should be interpreted with additional caution."`
- The Approved Pairs table (Table 3) must include a column `filter_regime` with value `"primary"` or `"relaxed"` so every row is unambiguous

**Economic logic (Stage 4):**
- For each pair surviving Stage 3: check if `within_sector == True`
- If within-sector: auto-assign Tier 1 or Tier 2 rationale based on sector
- If cross-sector: examine manually. If no economic link → reject
- Store rationale as a string column: e.g., "Same sector (Energy), both integrated oil majors"

**Edge cases:**
- All pairs filtered out: valid result. Report: "No pairs survived rigorous filtering"
- Economic logic requires human judgment: pre-build a rationale lookup for common within-sector pairs. For cross-sector survivors, the coding agent should flag them for manual review

**Test before moving on:**
- Verify funnel is monotonically decreasing
- Verify no pair with BH-adjusted p > 0.05 survived
- Verify no pair with half-life outside [5, 60] survived (unless fallback triggered)
- Verify no pair with β <= 0 survived

---

### Phase K: Ranking + Final Output (Notebook 02)

**Objective:** Rank surviving pairs, produce all required tables and plots.

**Inputs:** Filtered pairs DataFrame

**Outputs:** All 5 tables + all plots

**Ranking implementation:**
- Primary: use simplified fallback from methodology — sort by BH-adjusted p-value ascending, then by abs(half_life - 20) ascending
- If > 20 pairs survive and composite scoring is worth it: percentile-rank each metric, weighted sum, sort descending
- Select top 10-20

**Tables to generate:** See Section 3 below for exact schemas.

**Plots to generate:** See Section 4 below for exact specs.

---

## 3. Required Tables and Schemas

### Table 1: Data Audit Summary (Notebook 01 output)

**Part A: Universe completeness report (printed before screening)**

| Metric | Value |
|--------|-------|
| `total_unique_tickers_all_months` | int — total distinct tickers seen across any month |
| `tickers_in_all_12_months` | int — tickers present in every month |
| `tickers_excluded_by_12mo_rule` | int — how many dropped solely for not being in all 12 months |
| `excluded_ticker_list` | list of str — names of excluded tickers (for inspection) |

**Part B: Per-ticker screening table**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Stock symbol |
| `n_raw_minutes` | int | Total minute bars before filtering |
| `n_filtered_minutes` | int | After time filter (9:35-15:55) |
| `n_5min_bars` | int | After resampling |
| `median_close` | float | Median close price across full year |
| `avg_daily_dollar_volume` | float | Mean of daily (sum of volume × close) |
| `completeness_pct` | float | filtered_minutes / expected_minutes × 100 |
| `zero_return_pct` | float | % of minute returns that are exactly 0 |
| `n_outliers_flagged` | int | Number of minute returns with |z| > 10σ |
| `pct_outliers` | float | n_outliers_flagged / n_filtered_minutes × 100 |
| `passed_screening` | bool | True if ticker survived all filters |
| `rejection_reason` | str | Which filter failed (null if passed) |

### Table 2: Full Scan Results (Notebook 02 output, all tested pairs)

| Column | Type | Description |
|--------|------|-------------|
| `pair_id` | str | "TICKERA-TICKERB" alphabetical |
| `ticker_a` | str | First ticker (alphabetical) |
| `ticker_b` | str | Second ticker |
| `sector_a` | str | GICS sector of ticker_a |
| `sector_b` | str | GICS sector of ticker_b |
| `within_sector` | bool | sector_a == sector_b |
| `n_aligned_obs` | int | Number of aligned 5-min observations |
| `coint_tstat` | float | Engle-Granger test statistic from coint() |
| `raw_pval` | float | Raw p-value from coint() |
| `bh_adj_pval` | float | BH-FDR adjusted p-value |
| `bh_reject` | bool | True if BH rejects null (pair is cointegrated) |
| `hedge_ratio` | float | OLS slope β |
| `intercept` | float | OLS intercept α |

### Table 3: Approved Pairs (after all hard filters)

All columns from Table 2, plus:

| Column | Type | Description |
|--------|------|-------------|
| `half_life_days` | float | OU half-life in trading days |
| `hurst` | float | Hurst exponent of spread (optional, NaN if not computed) |
| `zero_crossings` | int | Demeaned spread zero-crossings (optional, NaN if not computed) |
| `economic_tier` | str | "Tier 1", "Tier 2", "Tier 3" |
| `economic_rationale` | str | 1-sentence explanation |
| `filter_regime` | str | "primary" or "relaxed" — indicates whether fallback thresholds were used |
| `composite_score` | float | Ranking score (or NaN if simplified ranking used) |
| `rank` | int | Final rank (1 = best) |

### Table 4: Rejected Pairs Summary (funnel + examples)

| Column | Type | Description |
|--------|------|-------------|
| `filter_stage` | str | "BH-FDR", "Half-life", "Hedge ratio", "Economic logic" |
| `pairs_entering` | int | Number of pairs entering this stage |
| `pairs_rejected` | int | Number rejected at this stage |
| `pairs_remaining` | int | Number surviving this stage |
| `example_rejected_pair` | str | One example pair_id rejected here |
| `example_reason` | str | Why it was rejected |

---

## 4. Required Plots

### Must-have plots (8 total minimum):

**Plot 1: P-value distribution histogram**
- X: raw p-values (0 to 1, 50 bins)
- Y: count
- Annotation: vertical line at 0.05, count of pairs below 0.05 before and after BH correction
- Purpose: show the uniform-under-null distribution + spike near 0 for true pairs

**Plot 2: Rejection funnel bar chart**
- X: filter stages (All Tested → BH-FDR → Half-life → β>0 → Economic Logic)
- Y: number of surviving pairs
- Purpose: visualize how many pairs drop at each stage

**Plot 3 (per top pair, ~10 pairs = ~10 plots): Spread time series**
- X: datetime
- Y: spread value (log_A - β·log_B)
- Overlay: horizontal lines at mean, ±1σ, ±2σ
- Title: "Spread: TICKER_A – β·TICKER_B (β = X.XX, p = X.XXX, HL = X days)"

**Plot 4 (per top pair): Normalized log price overlay**
- X: datetime
- Y: normalized log prices (subtract initial value so both start at 0)
- Two lines: ticker_A and ticker_B
- Purpose: visual confirmation the prices move together

### Nice-to-have plots (drop if behind schedule):

**Plot 5: Half-life distribution histogram**
- X: half-life in trading days (for BH-passing pairs)
- Y: count
- Vertical lines at 5 and 60 (filter bounds)

**Plot 6: Spread histogram with normal overlay (per top pair)**
- X: spread values
- Y: density
- Overlay: fitted normal PDF

**Plot 7: Composite score bar chart (top 20)**
- X: pair_id
- Y: composite score
- Color: by sector

---

## 5. Prototype Definition of Done

The prototype runs end-to-end on a **reduced dataset** to validate the pipeline before scaling.

### Prototype scope:
- **Notebook 01:** Load data for only 10 hand-picked tickers (e.g., AAPL, MSFT, AMZN, XOM, CVX, JPM, BAC, V, MA, META). Skip universe screening — these 10 are known liquid
- **Notebook 02:** Test all C(10,2) = 45 pairs

### Prototype is DONE when:
- [ ] 10-ticker data loads, time-filters, log-transforms, resamples to 5-min without errors
- [ ] 5-min log-price DataFrame saved to parquet and reloadable
- [ ] Sector dict covers all 10 tickers
- [ ] 45 pairs generated, all 45 have coint() results (no crashes)
- [ ] BH-FDR runs on 45 p-values, produces adjusted p-values
- [ ] Half-life computes for all BH-passing pairs without errors
- [ ] Hard filters produce a non-empty (or empty-but-reported) approved list
- [ ] At least 1 output table renders correctly
- [ ] At least 1 spread plot renders correctly for a top pair
- [ ] XOM-CVX appears in results and is inspectable (sanity candidate — useful for eyeballing, but failure alone does NOT prove the pipeline is wrong; 2022 market conditions and 5-min sampling may produce unexpected results for any specific pair)
- [ ] Runtime for all 45 pairs < 30 seconds

### Sanity checks vs. correctness tests (important distinction):
- **Correctness tests (hard fail):** coint() returns 3-tuple without error; BH-adjusted p-values >= raw p-values; half-life is positive when λ < 0; no NaN in required columns; funnel counts are monotonically decreasing
- **Sanity checks (soft, investigate only):** XOM-CVX having a low p-value; at least some pairs rejecting BH null; half-lives in plausible range. If these fail, investigate but do not auto-conclude the code is broken

### What the prototype does NOT need:
- Full 317-ticker load
- Economic rationale annotations
- All 5 tables complete
- All plots polished
- Composite scoring (just sort by p-value)

---

## 6. Full Notebook Definition of Done

### Notebook 01 is DONE when:
- [ ] All 12 months of CSVs loaded for tickers present in all months
- [ ] Universe completeness report printed (Part A of Table 1)
- [ ] Time filter applied (9:35-15:55 ET), pre/post market excluded
- [ ] Universe screening applied: price >= $5, volume >= $1M, completeness >= 90%, zero-returns < 50%
- [ ] If N survivors > 50: top-50 by dollar volume selected
- [ ] Outliers flagged and treated (|z| > 10σ), with per-ticker transparency table printed
- [ ] Any ticker with > 1% outlier rate removed instead of patched
- [ ] Log prices computed, no NaN/Inf
- [ ] Resampled to 5-min bars
- [ ] Data Audit Summary table (Table 1 Part B) printed in notebook
- [ ] `data/intermediate/log_prices_5min.parquet` saved
- [ ] Universe metadata saved

### Notebook 02 is DONE when:
- [ ] Cleaned data loaded from parquet
- [ ] Sector mapping dict covers ALL surviving tickers (assert, no KeyError)
- [ ] All C(N,2) pairs generated and tagged
- [ ] coint() + OLS run on all pairs without unhandled exceptions
- [ ] BH-FDR correction applied to all valid p-values
- [ ] Half-life computed for all BH-passing pairs
- [ ] 4 hard filters applied sequentially with funnel counts recorded
- [ ] Fallback triggered if needed, clearly labeled as "SENSITIVITY RELAXATION"
- [ ] filter_regime column populated ("primary" or "relaxed") in Table 3
- [ ] Economic rationale assigned to all surviving pairs
- [ ] Table 2 (Full Scan Results) exists and is complete
- [ ] Table 3 (Approved Pairs) exists with economic_rationale and filter_regime columns populated
- [ ] Table 4 (Rejection Summary) exists with funnel counts + examples
- [ ] Plot 1 (p-value histogram) rendered
- [ ] Plot 2 (rejection funnel) rendered
- [ ] Plot 3 (spread charts for top pairs) rendered — at least top 5
- [ ] Plot 4 (normalized price overlay for top pairs) rendered — at least top 5
- [ ] Notebook contains markdown cells documenting: assumptions, methodology summary, and limitations
- [ ] No cell raises an unhandled exception on Restart & Run All

---

## 7. Top Implementation Risks

### Risk 1: Universe too large → too many pairs (HIGHEST RISK)
**What:** 317 tickers × screening survivors could be 100-200+, giving 5K-20K pairs. coint() on 20K pairs with 18K rows each = hours of runtime.
**Mitigation:** Cap universe at 50 tickers (top by dollar volume). C(50,2) = 1,225 pairs. At ~0.3s per coint() call = ~6 minutes. Acceptable.
**Detection:** Check N_survivors immediately after screening. If > 50, apply cap before proceeding.

### Risk 2: `coint()` crashes on edge-case data
**What:** Degenerate series (constant price, too few obs, collinear) can cause LinAlgError or convergence failure inside coint().
**Mitigation:** Wrap every coint() call in try/except. Record failed pairs with reason. Do not let one failure stop the scan.
**Detection:** After scan, check `n_failed_pairs`. If > 5% of total, investigate.

### Risk 3: BH-FDR rejects everything (zero pairs survive)
**What:** If no pair has raw p < 0.05, or BH adjustment pushes all above 0.05.
**Mitigation:** Fallback: relax to q=0.10. If still zero: report honestly. Also: verify coint() is being called correctly (this would be a coding bug, not a data issue).
**Detection:** Check `sum(reject)` immediately after multipletests().

### Risk 4: Half-life computation returns nonsensical values
**What:** λ from OLS is positive (spread is explosive, not mean-reverting) or very close to zero (half-life = thousands of days).
**Mitigation:** For λ >= 0: set half_life = NaN, pair fails filter. For λ close to 0: half_life will be huge, pair fails [5, 60] filter naturally.
**Detection:** Histogram of half-lives for all BH-passing pairs. If all are > 100, suspect a units error (bars vs days conversion).

### Risk 5: Timestamp/timezone bugs
**What:** `window_start` is UTC nanoseconds. If not converted to ET correctly, the 9:35-15:55 filter will cut wrong rows. US Eastern has DST transitions (March, November 2022).
**Mitigation:** Use `pd.to_datetime(ns, unit='ns', utc=True).tz_convert('US/Eastern')`. Pandas handles DST automatically. Verify with known market open time.
**Detection:** After time filter, check that the earliest time on any day is ~9:35 ET and latest is ~15:55 ET. Spot-check a day in January (EST) and a day in July (EDT).

### Risk 6: Memory pressure from loading 131K CSVs
**What:** Loading all CSVs into memory at once could exceed RAM.
**Mitigation:** Load and process one ticker at a time (concatenate all days for that ticker, filter, then store only the close+volume columns). Discard raw data after processing each ticker. Or: only load the 50 selected tickers.
**Detection:** Monitor memory during load. If > 8GB before reaching 50% of tickers, switch to selective loading.

### Risk 7: Parquet save/load loses DatetimeIndex timezone
**What:** Some parquet engines strip timezone info on save, causing the reload to have UTC or naive timestamps.
**Mitigation:** After reload, verify `.index.tz` is `US/Eastern`. If stripped, re-localize.
**Detection:** Assert `loaded_df.index.tz is not None` immediately after reload.
