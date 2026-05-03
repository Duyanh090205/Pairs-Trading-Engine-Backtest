# Pairs Trading / Cointegration Notebook — Methodology Plan

## Context

Build a notebook that scans ~500 asset pairs for mathematical cointegration using Engle-Granger + ADF, rejects spurious correlations, and produces a Pairs Selection Report. Data: 1-minute OHLC flat files for 2022 (131K+ CSVs across 12 months, ~50+ tickers). This plan synthesizes 10 deep research documents into one actionable methodology spec.

---

# Phase 1: Research Synthesis

## File-by-File Summary

### File 1: "Spurious correlation, cointegration, and the mathematics of pairs selection" (root)
**Key conclusions:**
- Pearson correlation on non-stationary prices is mathematically meaningless — Phillips (1986) proved t-stats diverge as sqrt(T)
- Engle-Granger two-step: OLS for hedge ratio, then ADF on residuals with **MacKinnon N=2 critical values** (not standard ADF)
- EG critical values at 5%: **-3.34** (vs standard -2.86) — a 0.55 gap that causes massive false positives if ignored
- 500 stocks = 124,750 pairs; at alpha=0.05, expect **~6,237 false positives** purely by chance
- Benjamini-Hochberg FDR preferred over Bonferroni (too conservative)
- Half-life sweet spot: **5-60 days** for daily data
- 10-step framework: ADF on levels → EG residual test → Johansen → OOS validation → Economic plausibility → BH-FDR → Rolling correlation → Granger causality → BIC → Structural breaks
- **Economic plausibility is the most important filter; OOS validation is the most powerful statistical safeguard**

### File 2: "Statistical Foundations" (subfolder 1)
**Key conclusions (Vietnamese):**
- Always use `statsmodels.tsa.stattools.coint()` — it auto-applies MacKinnon N=2 and `regression='n'` correctly
- If manually running OLS + `adfuller()`, must use `mackinnonp(stat, N=2)` for correct p-values
- Multiple testing: BH-FDR via `multipletests(pvals, method='fdr_bh', alpha=0.05)`
- For minute data: cap maxlag at 20-30 (default formula gives huge lags)
- Consider downsampling to 5-min bars to reduce microstructure noise
- 8 common beginner mistakes listed, most critical: using standard ADF tables instead of MacKinnon N=2
- Correlation is a pre-filter only, not a selection criterion
- ~40% of in-sample cointegrated pairs fail out-of-sample (QuantRocket)
- **Log prices are standard** — must use consistently

### File 3: "Minute Data Problems" (subfolder 2)
**Key conclusions (Vietnamese):**
- **5 critical minute-data problems:** bid-ask bounce, stale quotes, illiquidity spikes, intraday U-shape volatility, spurious cointegration from massive sample size
- 98,000 obs/year gives ADF power to detect microstructure noise (stationary) instead of economic cointegration → spurious rejection
- **Solution: resample to 5-minute bars** — Liu et al. (2017) used 5-min, Sharpe 3.9; Hansen & Lunde confirm noise mostly eliminated at 5-min
- Exclude opening 5-10 min (auction effects) and closing 5 min → usable window: **9:35-15:55 ET** (~370 min/day)
- Alignment: inner join as baseline; limited ffill (max 5 bars) if >10-15% sample loss
- Data cleaning thresholds: price >=$5, daily dollar volume >=$1M, completeness >=95%, zero-return minutes <50%
- Outliers: remove |z|>10sigma (data errors), winsorize at 0.1/99.9 percentile
- **Log prices default** for equities — variance stabilization, academic standard
- Dave Giles: **time span matters more than sample size** for unit root tests; need >=10x expected half-life
- Recommended workflow: 11 steps from raw ingestion through validation

### File 4: "Spurious Correlation in Financial Time Series" (subfolder 3, Vietnamese PDF)
Covered same ground as the markdown companion — Granger-Newbold, Phillips divergence, multiple testing. Reinforced the four mechanisms creating fake correlation: common macro trend, multiple testing, regime-specific correlation, look-ahead bias.

### File 5: "Spurious Correlation & Economic Logic" (subfolder 3, English markdown)
**Key conclusions:**
- **Composite scoring system**: Statistical tier (max 50) + Economic tier (max 50) = 100 points
- Hard filters before scoring: EG p-value < 0.10, Hurst < 0.50, half-life 5-250 days, economic logic >= 15/50, liquidity (ADV > 100K shares, spread < 50 bps)
- Red flags: Critical (auto-reject) / High-severity (2+ = reject) / Medium (investigate)
- 1-sentence rejection rule: if you can't explain the pair in one economic sentence, reject it
- Detailed economic checklist: Industry alignment (8pts), Supply chain (7pts), Shared drivers (7pts), Regulatory/currency (10pts), Product market (5pts), Geographic overlap (6pts), Precedent (7pts)
- Real examples with full scoring: XOM/CVX = 87/100 (strong accept), DAL/UAL (conditional), AAPL/XOM (reject)
- Decision matrix: >=75 accept, 60-74 conditional, 40-59 weak, <40 reject

### File 6: "ADF and p-value on residuals" (subfolder 4, Vietnamese PDF)
Reinforced MacKinnon critical values, super-consistency of OLS in cointegrating regressions, proper test procedure.

### File 7: "Pairs trading methodology guide" (subfolder 4, English markdown)
**Key conclusions:**
- Detailed pass/fail framework with specific thresholds
- Hard filters: >=252 trading days, ADF adjusted p < 0.05 after BH-FDR, Hurst < 0.5, beta > 0 for same-sector, ADF passes in >=2 of 3 sub-periods
- **Composite ranking weights**: ADF stat (25%), half-life proximity to 15-25 days (20%), spread Sharpe (15%), zero crossings (15%), hedge ratio stability (10%), adjusted p-value (10%), Hurst (5%)
- Correlation gets **zero weight** in ranking — only pre-filter role
- Top 10-20 pairs for final report
- 7 invalidating errors documented
- Epistemic honesty requirements for report language

### File 8: "Spurious Correlation Rules" (methodology folder)
**Key conclusions:**
- Two-layer filter: Statistical first, Economic second
- Pragmatic I(1) check: don't over-engineer upfront; check I(1) only for shortlisted assets
- 1-sentence rejection rule confirmed

### File 9: "Pairs Selection Report Outline" (outputs folder)
Template: Executive Summary → Data & Methodology → Scan Results → Selected Pair Deep-Dives → Conclusion

## Cross-File Conflicts & Resolutions

| Topic | Conflict | Resolution |
|-------|----------|------------|
| **Half-life range** | File 1: 5-60 days. File 5: 5-250 days. File 7: 5-60 preferred, cite Avellaneda κ>8.4 (~30 day ceiling) | **Use 5-60 trading days as hard filter for minute data.** 250 days is for daily data with longer horizons; we have minute data for 2022 only |
| **Minimum data length** | File 7: >=252 trading days. Our data: ~252 days of 2022 minute data | **Use full 2022 dataset.** Split into ~8 months formation + ~4 months OOS if doing validation |
| **I(1) check timing** | File 1: check both series I(1) first (step 1). File 8: pragmatic — check I(1) only for shortlisted pairs | **Pragmatic approach: run ADF on individual series as a diagnostic for shortlisted pairs, not as a gate for all 500+.** Most equity prices are I(1) by nature; confirm for final candidates |
| **Johansen test** | File 1: run as step 3 confirmation. File 7: not mentioned in hard filters. File 5: EG+Johansen disagreement is a medium flag | **Use Engle-Granger as primary. Johansen as optional confirmation for top candidates only.** Keeps notebook simpler |
| **Structural breaks** | Files 1,5: include CUSUM/Chow/Bai-Perron. File 8: not mentioned | **Skip formal structural break tests.** With only 2022 data, structural break detection has limited value. Flag it as a limitation |
| **Scoring system** | File 5: elaborate 100-point composite. File 7: percentile-rank weighted composite | **Use File 7's percentile-rank approach for statistical scoring, simplified.** File 5's 100-point system is too heavy for a class project — adapt the economic logic as a qualitative pass/fail rather than a 50-point rubric |
| **Resampling** | File 3: 5-min default. File 2: consider 5-min | **Resample to 5-minute bars.** Unanimous recommendation from research |
| **Correlation pre-filter** | File 2: >0.7-0.9. File 7: >0.5-0.7. File 4: mentioned as option | **Do NOT pre-filter by correlation.** Test all C(N,2) pairs. Correlation pre-filter was considered but dropped in favor of testing everything and letting BH-FDR + economic logic do the filtering. Stronger for the report |

---

# Phase 2: Recommended Methodology Decisions

## 1. Main Statistical Method
**Engle-Granger two-step cointegration test**, implemented via `statsmodels.tsa.stattools.coint()`.

*Why:* Unanimous recommendation across all research files. Simpler than Johansen for bivariate pairs. `coint()` handles MacKinnon N=2 critical values automatically, avoiding the single most common implementation error.

## 2. Exact Role of Engle-Granger
- **Step 1:** OLS regression `log(P_A) = α + β·log(P_B) + ε` → produces hedge ratio β and residuals ε
- **Step 2:** ADF test on residuals ε with MacKinnon N=2 critical values → tests H₀: no cointegration

The hedge ratio β from step 1 defines the spread. The p-value from step 2 is the cointegration evidence.

## 3. What ADF Should Be Run On
- **Primary (cointegration test):** ADF on OLS residuals via `coint()` — this IS the cointegration test
- **Diagnostic (for shortlisted pairs only):** ADF on individual log price series to confirm they are I(1)
- **Do NOT** run ADF on raw prices and interpret it as a cointegration result

## 4. Raw Prices vs Log Prices
**Use log prices.**

*Why:* (a) Variance stabilization — $200 stock and $20 stock become comparable; (b) `log(P_A) - β·log(P_B)` stationary means the price ratio mean-reverts, which has economic meaning; (c) Academic standard (Vidyamurthy 2004, Avellaneda & Lee 2010, Liu et al. 2017); (d) All research files recommend it unanimously.

Apply `np.log()` after confirming all prices > 0 (enforced by the >=$5 price filter).

## 5. How to Clean and Align Minute-Level Data

### Preprocessing pipeline (in order):
1. **Load & parse:** Read CSVs, convert `window_start` (nanoseconds UTC) to ET datetime
2. **Filter trading hours:** Keep only 9:30-16:00 ET, then trim to **9:35-15:55 ET** (exclude opening/closing auctions)
3. **Universe screening:**
   - Median price >= $5
   - Average daily dollar volume >= $1M (volume × close)
   - Data completeness >= 90% of expected minutes
   - Zero-return minutes < 50%
4. **Outlier treatment:** Flag minute returns with |z| > 10σ as data errors → replace with NaN → forward-fill (max 1 bar)
5. **Log transform:** `log_price = np.log(close_price)`
6. **Resample to 5-minute bars:** Take last close price in each 5-min window. This reduces microstructure noise (bid-ask bounce, stale quotes, Epps effect) and prevents ADF from overpowering on noise
7. **Pairwise alignment:** Inner join on timestamp. If >15% sample loss, apply limited ffill (max 1 bar = 5 min) then re-join. If still >20% loss, skip that pair

## 6. How to Define the Asset Universe
- Profile all unique tickers in the data
- Apply the screening filters above
- Manually assign GICS sectors to surviving tickers (hardcoded dict in notebook)
- Target: ~30-50 liquid, well-known stocks that span multiple sectors

## 7. How to Generate ~500 Candidate Pairs

**Deterministic algorithm — no heuristics:**

```
STEP 1: After universe screening, get list of N surviving tickers
STEP 2: Assign each ticker a GICS sector via hardcoded dict
STEP 3: Generate ALL C(N,2) unique pairs (every ticker paired with every other)
STEP 4: Tag each pair as "within-sector" or "cross-sector"
STEP 5: Test ALL generated pairs with coint() — do NOT pre-filter by correlation
STEP 6: The ~500 target is descriptive, not prescriptive — test however many pairs exist
```

**Why test all pairs, not just within-sector?**
- Sector tag is used AFTER statistical testing, in the economic logic filter
- Cross-sector pairs that pass cointegration are almost certainly spurious → they get rejected at the economic logic step, which is the point — you SHOW the rejection
- This is stronger for the report: "we tested X pairs including cross-sector; cross-sector pairs were rejected for lacking economic rationale"

**If N = 32 tickers → 496 pairs. If N = 50 → 1,225 pairs.** Both are fine. Report whatever the data gives you.

## 8. How to Define Pass/Fail Rules

### Hard filters (must pass ALL):
| # | Filter | Threshold | Fallback |
|---|--------|-----------|----------|
| 1 | BH-FDR adjusted p-value | < 0.05 | Relax to 0.10 if <5 pairs survive |
| 2 | Half-life | 5–60 trading days | Relax to 3–90 if <10 pairs survive |
| 3 | Hedge ratio β | > 0 | No relaxation |
| 4 | Economic logic | 1-sentence rationale exists | No relaxation — this is the anti-spurious wall |

Hurst exponent: computed and displayed, but NOT a gate.

### Automatic disqualification:
- Hedge ratio flips sign during formation period
- Either stock has avg daily volume < $1M
- No economic rationale whatsoever (cross-sector with zero linkage)

### Fallback if too few pairs survive:
| Trigger | Action |
|---------|--------|
| Fewer than 10 pairs survive all filters | Relax half-life range from [5, 60] to [3, 90] days |
| Still fewer than 5 pairs | Relax BH-FDR threshold from q=0.05 to q=0.10, clearly document this |
| Still fewer than 3 pairs | Report whatever survives honestly. Finding few pairs is a VALID result |

**What NOT to do as a fallback:**
- Do NOT drop BH-FDR correction entirely
- Do NOT drop the economic logic requirement
- Do NOT switch to correlation-based selection

## 9. How to Define Ranking Rules
**Percentile-rank composite scoring** on surviving pairs:

| Metric | Direction | Weight |
|--------|-----------|--------|
| ADF test statistic (more negative = better) | Higher abs value = better | 30% |
| Half-life proximity to sweet spot (~15-25 days equiv.) | Closer = better | 25% |
| Zero crossings of spread per unit time | More = better | 20% |
| BH-adjusted p-value | Lower = better | 15% |
| Hurst exponent | Lower = better | 10% |

Select **top 10-20 pairs** for the final report.

**Simplified fallback ranking:** If composite scoring adds too much complexity, sort by BH-adjusted p-value ascending, then by half-life proximity to 20 days. This is sufficient.

## 10. Spurious-Correlation Rejection Rules
Three-layer defense:

1. **Statistical:** BH-FDR correction at q=0.05 across all tested pairs
2. **Economic:** 1-sentence rule — if you cannot state in one sentence why this pair should mean-revert, reject it regardless of p-value
3. **Diagnostic:** For top candidates, verify both individual series are plausibly I(1) via ADF on levels (should fail to reject) and on differences (should reject)

## 11. Economic Logic for a Valid Pair
Simplified from File 5's 50-point rubric into a practical tier system:

- **Tier 1 (Strong):** Same GICS sub-industry + direct competitors (XOM/CVX, V/MA)
- **Tier 2 (Adequate):** Same sector + shared cost/revenue drivers (DAL/UAL via jet fuel)
- **Tier 3 (Marginal):** Related sectors with clear economic linkage (gold miner/gold ETF)
- **Reject:** Cross-sector with no identifiable linkage (AAPL/XOM)

For the report: annotate each selected pair with its economic rationale in 1-2 sentences.

## 12. Whether and How to Use Half-Life
**Yes, include half-life as a hard filter and ranking input.**

- Estimate via OU process discretization: regress ΔS on S_{t-1}, get λ, compute `half_life = -ln(2)/λ`
- Filter range: 5-60 trading days (on 5-min bars: 1 day ≈ 74 bars, so 370-4440 bars)
- Ranking: score proximity to 15-25 day sweet spot
- Half-life is practical tradability — ADF says *whether* spread reverts, half-life says *how fast*

## 13. Multiple Testing Correction
**Benjamini-Hochberg FDR at q = 0.05.**

```python
from statsmodels.stats.multitest import multipletests
reject, pvals_adj, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
```

*Why not Bonferroni:* Too conservative for pairs screening — Harlacher (2016) showed it eliminates even truly cointegrated pairs. BH controls the *proportion* of false discoveries, not the probability of *any* false discovery, which is the right framing for a screening exercise.

Report both raw and adjusted p-values in the output table.

---

# Phase 3: Final Implementation Spec

## Objective
Scan ~500 equity pairs from 2022 minute-level data for mathematical cointegration. Produce a Pairs Selection Report that identifies the top 10-20 cointegrated pairs, proves each pair's statistical validity, and explicitly rejects spurious correlations with economic reasoning.

## Datasets and Their Roles

| Dataset | Role |
|---------|------|
| 1-minute OHLC flat files (2022, 12 months) | Primary dataset. Resampled to 5-min for cointegration testing |
| 1987_crash_market_data.csv | NOT used. Skipped entirely to keep notebook focused |

## Preprocessing Rules (Notebook 01: Data Profiling)

1. **Load:** Parse `window_start` (nanoseconds UTC) → ET datetime. Use `close` as the price field
2. **Filter hours:** Keep 9:35-15:55 ET only (drop pre/post-market, first 5 min, last 5 min)
3. **Screen universe:**
   - Median close >= $5
   - Avg daily dollar volume >= $1M
   - Minute-level completeness >= 90%
   - Zero-return minute fraction < 50%
4. **Outliers:** Remove minute returns |z| > 10σ (replace price with NaN, ffill max 1 bar)
5. **Log transform:** `log_close = np.log(close)`
6. **Resample:** Aggregate to 5-minute bars (last close per 5-min window)
7. **Output:** Cleaned, aligned 5-min log-price DataFrame for all tickers. Save to `data/intermediate/`

## Pair-Generation Rules (Notebook 02: Cointegration Scan)

1. Assign GICS sectors to all surviving tickers (hardcoded dict in notebook)
2. Generate ALL C(N,2) unique pairs from N surviving tickers
3. Tag each pair as "within-sector" or "cross-sector"
4. Test ALL pairs — do not pre-filter
5. Record pair metadata: ticker_A, ticker_B, sector, within_sector flag, pair_id

## Division of Labor: `coint()` vs OLS

These are NOT alternatives. They are used together, for different outputs:

| Function | What it does | What you get from it |
|----------|-------------|---------------------|
| `statsmodels.tsa.stattools.coint(y, x, trend='c', maxlag=30)` | Runs the full Engle-Granger two-step internally (OLS + ADF on residuals with MacKinnon N=2) | `coint_tstat`, `pvalue`, `crit_values` — the cointegration verdict |
| `statsmodels.regression.linear_model.OLS(y, sm.add_constant(x)).fit()` | Separate OLS regression you run yourself | `beta` (hedge ratio), `resid` (spread = y - β·x) — the spread you need for half-life, plots, and trading |

**Workflow per pair:**
```
1. coint(log_A, log_B) → get p-value (is this pair cointegrated?)
2. OLS(log_A ~ log_B)  → get hedge ratio β and residuals (what does the spread look like?)
3. Use residuals from step 2 to compute half-life, zero-crossings, plots
```

## Cointegration Testing Workflow

For each candidate pair (log_price_A, log_price_B):

```
Step 1: Pairwise inner join on 5-min timestamps
Step 2: Run coint(log_A, log_B, trend='c', autolag='aic', maxlag=30)
        → returns (coint_t, pvalue, crit_values)
Step 3: Run OLS log_A = α + β·log_B + ε → store hedge ratio β and residuals
Step 4: Compute spread = log_A - β·log_B
Step 5: Compute half-life via OU regression on spread
Step 6: Compute Hurst exponent of spread (optional diagnostic)
Step 7: Count zero-crossings of demeaned spread (nice-to-have)
Step 8: Store all results in a DataFrame
```

Cap `maxlag=30` to control computation time on 5-min data.

## Filtering Rules (applied sequentially)

```
Filter 1: BH-FDR correction on all raw p-values → keep pairs where reject=True (q=0.05)
Filter 2: Half-life in [5, 60] trading days
Filter 3: Hedge ratio β > 0
Filter 4: Economic logic — 1-sentence rationale must exist for each surviving pair
          Reject any pair that lacks identifiable economic linkage
```

## Ranking Rules

For pairs surviving all filters:
1. Convert each metric to percentile rank (0-100) across the surviving pool
2. Compute weighted composite: ADF_stat(30%) + half_life_score(25%) + zero_crossings(20%) + adj_pval(15%) + hurst(10%)
3. Sort descending by composite score
4. Select top 10-20 pairs for deep-dive report

**Simplified fallback:** Sort by BH-adjusted p-value ascending, then by half-life proximity to 20 days.

## Must-Have vs. Optional Components

### MUST-HAVE (assignment fails without these)
1. **Data loading & cleaning** — parse minute CSVs, filter hours, screen universe, resample to 5-min
2. **Engle-Granger cointegration test via `coint()`** on all pairs with correct MacKinnon N=2 critical values
3. **BH-FDR multiple testing correction** — this is the proof you didn't just data-mine
4. **Half-life computation** for surviving pairs — proves tradability, not just statistical significance
5. **Economic logic annotation** — 1-sentence rationale per selected pair rejecting spurious results
6. **Pairs Selection Report output** — ranked table of top pairs with stats + economic rationale
7. **Rejection evidence** — show pairs that were rejected and why (funnel + examples)
8. **Key plots**: (a) p-value distribution, (b) spread charts for top pairs with mean/sigma bands, (c) rejection funnel

### NICE-TO-HAVE (include if time permits, drop without penalty)
- Hurst exponent
- Zero-crossings count
- Composite percentile-rank scoring (can simplify to sorting by p-value + half-life)
- Spread histogram with normal overlay
- Sub-sample stability check
- Hedge ratio stability analysis (rolling β CV)
- Formal I(1) confirmation on individual series

### EXPLICITLY OUT OF SCOPE
- Johansen test, Kalman filter, structural break tests, Granger causality, rolling cointegration windows
- 1987 crash data, backtesting, PnL simulation, transaction cost modeling

## Required Output Tables

### Table 1: Universe Summary
- Total tickers loaded, tickers after screening, pairs generated, pairs tested

### Table 2: Full Scan Results (all pairs)
Columns: `pair_id, ticker_a, ticker_b, sector, within_sector, coint_tstat, raw_pval, bh_adj_pval, bh_reject, hedge_ratio, half_life_days, hurst, zero_crossings, composite_score, rank`

### Table 3: Filtered Pairs (after all hard filters)
Same columns as Table 2, plus `economic_rationale` (1-sentence string)

### Table 4: Top 10-20 Selected Pairs
Same columns plus detailed annotations

### Table 5: Rejection Summary
- How many pairs rejected at each filter stage (funnel chart data)
- Examples of spurious pairs rejected and why

## Required Plots

1. **P-value distribution histogram** — raw p-values across all pairs (expect uniform under null + spike near 0 for true pairs)
2. **Rejection funnel** — bar chart showing pairs remaining after each filter stage
3. **For each top-10 pair:**
   - (a) Normalized log price overlay (both series on same chart)
   - (b) Spread time series with ±1σ, ±2σ bands and mean line
   - (c) Spread distribution histogram with normal overlay
4. **Half-life distribution** — histogram across all passing pairs
5. **Composite score distribution** — histogram or ranked bar chart of top 20

## Assumptions (state explicitly in notebook)

- Prices in the flat files are split-adjusted and dividend-adjusted (or: we assume they are; the data source documentation does not confirm adjustments)
- All tickers are US equities trading in USD during regular hours
- We use 2022 data only — cointegration relationships may not persist beyond this window
- No transaction costs, short-selling costs, or market impact are modeled
- Sector assignments are based on primary business as of 2022

## Things Explicitly NOT To Do

- Do NOT run ADF on raw prices and call it a cointegration test
- Do NOT use standard ADF critical values for Engle-Granger residuals (use `coint()` which handles this)
- Do NOT use Pearson correlation on price levels as evidence of a relationship
- Do NOT use Bonferroni correction (too conservative — use BH-FDR)
- Do NOT test at 1-minute frequency (microstructure noise dominates; resample to 5-min)
- Do NOT include pre/post-market data
- Do NOT skip multiple testing correction
- Do NOT claim cointegration "proves" a pair is profitable — use hedged language ("provides evidence consistent with")
- Do NOT run Johansen as the primary test (Engle-Granger is simpler and sufficient for bivariate)
- Do NOT build an elaborate 100-point scoring rubric — keep it practical for a class project
- Do NOT attempt Kalman filter hedge ratio estimation (overkill for this scope)
- Do NOT attempt formal structural break tests (insufficient data span for meaningful results)

---

# Resolved Decisions

1. **Resampling:** 5-minute bars only. No 1-min comparison.
2. **Universe size:** ~500 pairs is a soft target — whatever screening produces is fine.
3. **OOS validation:** Use full 2022 year for formation. Document lack of OOS as a known limitation.
4. **1987 crash data:** Skip entirely. Notebook focuses solely on 2022 pairs analysis.
5. **Sector mapping:** Hardcoded Python dictionary in the notebook. No external CSV.
6. **Hurst exponent:** Optional diagnostic, not a hard filter. Computed and displayed but does not gate.
7. **Pair generation:** Deterministic — all C(N,2) pairs, no correlation pre-filter. Tag within/cross-sector for economic logic step.
8. **Fallback:** Tiered relaxation (half-life first, then FDR threshold). Never drop BH-FDR or economic logic. Few surviving pairs is a valid result.
