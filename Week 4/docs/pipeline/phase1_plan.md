# Phase 1 Plan — Cointegration Discovery

## Objective
Per-fold cointegration engine. Takes a 6-month formation window slice of 5-min log-prices and outputs surviving pairs with hedge ratio metadata. Called ~45 times by the Phase 4 orchestrator.

## Input
- `data/validated/5min_phase1.parquet` (built by Phase 0)
- `formation_start`, `formation_end` timestamps (from orchestrator)

## Output (per fold)
DataFrame with columns: `[ticker_A, ticker_B, alpha_PCA, beta_PCA, half_life_days, R_measurement_noise, johansen_pval]`

## Reuse from Prior Weeks
| Code | Source | Adaptation needed |
|---|---|---|
| BH-FDR correction | `Week 1/notebooks/02_cointegration_scan.py` | Direct reuse: `multipletests(pvals, alpha=0.05, method='fdr_bh')` |
| OU half-life AR(1) regression | `Week 1/notebooks/02_cointegration_scan.py` | Change divisor: 77 → 78 bars/day; change range: [5,60d] → [1,10d] |
| Universe hard screens | `Week 1/notebooks/01_data_profiling.py` | Direct reuse (same thresholds) |

## New Code Required
| Component | Why not in prior weeks |
|---|---|
| Johansen cointegration test | Week 1 used Engle-Granger (asymmetric, double-test bias) |
| PCA hedge ratio (secondary eigenvector) | Week 1 used OLS (asymmetric attenuation bias) |
| Pairwise inner join with 80% min-overlap | Week 1 used universe-wide rectangular inner join |

## File to Create
`src/phase1_cointegration/discovery.py`

## Implementation Steps

### Step 1 — Load Formation Window Slice
```python
df = pd.read_parquet('data/validated/5min_phase1.parquet')
df_fold = df[(df.index >= formation_start) & (df.index <= formation_end)]
```

### Step 2 — Universe Hard Screens (per ticker)
Adapt from `Week 1/notebooks/01_data_profiling.py`:
```python
# Median close >= $5
# ADV_$ = mean(close × volume) >= $1M
# Completeness >= 90%: len(ticker_bars) / expected_bars >= 0.90
# Zero-return fraction < 50%
```
Log: actual survivor count + which filter is binding for this fold.

### Step 3 — All-Pairs Enumeration
```python
from itertools import combinations
pairs = list(combinations(survivor_tickers, 2))
# No sector filter, no volume bucket filter — pure combinatorial
```

### Step 4 — Pairwise Inner Join with Min-Overlap
For each pair (A, B):
```python
df_pair = df_A.join(df_B, how='inner', lsuffix='_a', rsuffix='_b')
overlap_ratio = len(df_pair) / expected_bars_in_formation_window
if overlap_ratio < 0.80:
    continue   # skip pair
```
This is pairwise — NOT universe-wide rectangular join.

### Step 5 — PCA Hedge Ratio (NEW — replaces OLS from Week 1)
```python
X = np.column_stack([
    np.log(df_pair['close_a']) - np.log(df_pair['close_a']).mean(),
    np.log(df_pair['close_b']) - np.log(df_pair['close_b']).mean()
])   # shape (T, 2), centered

Cov = X.T @ X / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(Cov)   # sorted ascending

# Sort DESCENDING (eigh gives ascending)
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# Secondary eigenvector = cointegrating direction (Avellaneda-Lee)
v2 = eigenvectors[:, 1]
beta_PCA  = -v2[0] / v2[1]
alpha_PCA = np.log(df_pair['close_a']).mean() - beta_PCA * np.log(df_pair['close_b']).mean()

# Measurement noise R for Kalman
spread_static = np.log(df_pair['close_a']) - alpha_PCA - beta_PCA * np.log(df_pair['close_b'])
R = float(spread_static.var())
```

### Step 6 — Johansen Cointegration Test (NEW — replaces EG from Week 1)
```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

log_prices = np.column_stack([
    np.log(df_pair['close_a']),
    np.log(df_pair['close_b'])
])

result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
# Trace statistic p-value
pval_trace = result.trace_stat_crit_vals[0]   # at rank 0 = at least 1 coint vector
# Use numerical p-value (statsmodels 0.14+ provides lr1 test p-values)
johansen_pval = ...   # collect for BH-FDR
```

### Step 7 — BH-FDR Multiple Testing Correction
Reuse from `Week 1/notebooks/02_cointegration_scan.py`:
```python
from statsmodels.stats.multitest import multipletests

reject, pvals_adj, _, _ = multipletests(
    all_johansen_pvals, alpha=0.05, method='fdr_bh'
)
surviving_pairs = [p for p, r in zip(all_pairs, reject) if r]
```
Apply over ALL tested pairs in this fold.

### Step 8 — OU Half-Life Filter
Adapt from `Week 1/notebooks/02_cointegration_scan.py`:
```python
spread = spread_static   # computed in Step 5
spread_lag  = spread.shift(1)
spread_diff = spread.diff()
valid = pd.concat([spread_diff, spread_lag], axis=1).dropna()
valid.columns = ['diff', 'lag']

import statsmodels.api as sm
model = sm.OLS(valid['diff'], sm.add_constant(valid['lag'])).fit()
kappa = model.params['lag']   # should be < 0 for mean-reversion

if kappa >= 0:
    skip pair   # not mean-reverting

theta = -kappa   # mean-reversion speed (per 5-min bar)
half_life_raw_bars = np.log(2) / theta
half_life_days = half_life_raw_bars / 78   # 78 5-min bars per trading day

# Keep only if [1, 10] trading days (Week 4 spec — changed from [5, 60] in Week 1)
if not (1.0 <= half_life_days <= 10.0):
    skip pair
```

### Step 9 — No Pair Cap
Return all surviving pairs. Log count. Fallback only: if runtime > threshold, cap at top-K by Johansen p-value and log "FALLBACK CAP APPLIED".

## Skills to Invoke

### If Stuck on Johansen or OU Fitting → Use `lobehub-skills-search-engine`
If `statsmodels.tsa.vector_ar.vecm.coint_johansen` behavior is unclear or the OU AR(1) regression needs a specialized implementation, use the search engine skill to find additional reference skills:
```
# Search for Johansen cointegration skill
skills search "johansen cointegration test python"

# Search for OU half-life fitting
skills search "ornstein uhlenbeck half life estimation"
```
Use `lobehub-skills-search-engine` as a fallback — don't block implementation while waiting.

### After Implementing `discovery.py` → `/simplify`
The pairwise loop over 80k–125k pairs is the hot path. Verify vectorization where possible.

### After 1-Fold Smoke Test → `/review`
Confirm implementation against pipeline spec §1.1–1.7.

## Smoke Test (before hard stop)
1. Run `discovery.py` on Fold 1 (formation window: 2022-07-01 to 2022-12-31) with 10 tickers
2. Assert output DataFrame has all required columns: `[ticker_A, ticker_B, alpha_PCA, beta_PCA, half_life_days, R_measurement_noise, johansen_pval]`
3. Verify `beta_PCA` is from secondary eigenvector (not OLS) by checking it differs from `np.linalg.lstsq` result
4. Verify `half_life_days` is in [1, 10] for all surviving pairs
5. Check BH-FDR was applied: number of surviving pairs < number tested at raw p<0.05
6. Log survivor count and binding filter for Fold 1 — document in results/logs/fold01_universe.txt

---
## ⛔ HARD STOP — Review Before Phase 2

**Before proceeding to Phase 2, verify:**
- [ ] `discovery.py` runs without error on at least 1 full fold (6-month window)
- [ ] Output columns all present and numerically sane (no NaN, no inf)
- [ ] `beta_PCA` computed from secondary eigenvector (confirmed by manual spot-check vs OLS)
- [ ] `half_life_days` all in [1, 10] for surviving pairs
- [ ] BH-FDR applied — surviving pairs < raw p<0.05 count
- [ ] Survivor count logged (expect ~400–500 tickers in universe, pair count varies by fold)
- [ ] `/simplify` run and hot-path (pairwise loop) reviewed for vectorization
- [ ] `/review` run and implementation confirmed against pipeline spec §1.1–1.7

**Signal to proceed:** User explicitly types "Phase 1 approved, proceed to Phase 2"
