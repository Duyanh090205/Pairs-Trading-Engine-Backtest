# Vectorized pair trading Z-score pipelines at scale

The key to a performant 500-asset pair trading pipeline is **never computing all 124,750 pairs at once**. A cascading filter on daily-resampled data—correlation screening, then cointegration testing—reduces the universe to 10–50 tradeable pairs before any minute-level Z-score computation begins. For the computation itself, Bottleneck's `move_mean`/`move_std` on NumPy arrays deliver **60–300× speedups** over naive approaches, while chunked processing with joblib parallelization keeps memory under control. The entire architecture must enforce a strict walk-forward boundary: pairs are selected and hedge ratios estimated only on in-sample data, with rolling Z-scores computed exclusively from past observations.

This report synthesizes best practices across vectorized computation, pair selection, spread construction, walk-forward design, storage, project structure, and the critical pitfalls that destroy pair trading backtests.

---

## The computation is infeasible without aggressive pre-filtering

The arithmetic is unforgiving. With 500 assets at 1-minute resolution, one year of data produces a price matrix of shape **(98,280 × 500)**, consuming ~393 MB in float64. Computing spreads for all 124,750 pairs creates a **(98,280 × 124,750)** array—roughly **98 GB in float64**. Add rolling mean, rolling standard deviation, and Z-score arrays, and the naive all-at-once approach demands ~400 GB of RAM. The 3D broadcasting approach (`prices[:, :, np.newaxis] - prices[:, np.newaxis, :]`) is even worse at **~196 TB**—completely infeasible.

The solution is a multi-stage filtering pipeline that progressively narrows the pair universe using increasingly expensive tests, all on **daily-resampled data** before touching minute bars:

- **Stage 1 — Heuristic filters** (instant): Restrict to same-sector/industry pairs, similar market cap. Reduces ~125K pairs to ~5,000–15,000.
- **Stage 2 — Correlation screening** (sub-second): Compute the full 500×500 correlation matrix via `pd.DataFrame.corr()` on daily returns. Filter pairs above ρ > 0.8. Typically yields ~1,000–3,000 surviving pairs.
- **Stage 3 — Cointegration testing** (~1–3 minutes parallelized): Run `statsmodels.tsa.stattools.coint()` on surviving pairs. Apply **Benjamini-Hochberg FDR correction** at α=0.05 to control false discoveries. Yields ~100–500 pairs.
- **Stage 4 — Quality filters** (seconds): Hurst exponent < 0.5, half-life between 5–120 days, sufficient zero-crossings. Final result: **10–50 tradeable pairs**.

At 50 pairs on minute data, the spread matrix is just (98,280 × 50)—roughly **39 MB** in float64. Rolling Z-scores fit trivially in RAM.

The correlation matrix itself can be computed via optimized matrix multiplication rather than the pandas convenience method:

```python
returns = np.diff(np.log(prices_np), axis=0)
xv = returns - returns.mean(axis=0)
xvss = (xv * xv).sum(axis=0)
corr_matrix = np.matmul(xv.T, xv) / np.sqrt(np.outer(xvss, xvss))
```

This leverages BLAS-optimized routines and completes in under a second for 500 assets. The multiple testing problem is severe here: testing 124,750 pairs at α=0.05 produces ~6,237 false positives by chance. Benjamini-Hochberg FDR correction is the recommended approach—it controls the *proportion* of false discoveries rather than the family-wise error rate, making it far less conservative than Bonferroni while remaining statistically rigorous.

---

## Log prices and Kalman-filtered hedge ratios produce the most stationary spreads

Spread construction requires two decisions: price transformation and hedge ratio estimation. **Log prices are the default recommendation.** The spread `log(P_A) − β·log(P_B)` has better statistical properties—log returns more closely approximate normality, and practitioners consistently find it easier to identify stationary spreads in log space. The tradeoff is that log-price spreads require periodic rebalancing to maintain the dollar-value ratio, whereas raw-price spreads maintain fixed share counts. For a walk-forward pipeline with regular re-estimation, the rebalancing cost is negligible.

For hedge ratio estimation, three methods are common, with clear performance differences:

**Rolling OLS** via `statsmodels.regression.rolling.RollingOLS` is the simplest approach. Set `params_only=True` for speed. Typical windows of 60–252 days balance responsiveness and stability. The critical limitation is **asymmetry**: regressing A on B gives a different hedge ratio than B on A. Best practice is to test both orderings and select the one producing the most negative ADF test statistic.

**Total Least Squares (TLS)** via PCA resolves the asymmetry problem entirely. The hedge ratio from TLS of A vs. B equals the inverse of TLS of B vs. A. Implementation is straightforward using `sklearn.decomposition.PCA` on the two-column price matrix. Multiple practitioners now recommend TLS over OLS for this reason.

**Kalman filtering** is the strongest approach and is considered essential by several authoritative sources. It treats the hedge ratio as a latent state evolving over time via a random-walk transition model. Compared to rolling OLS, Kalman-filtered hedge ratios are dramatically more stable—rolling OLS ratios can swing from 0.6 to 1.2, while Kalman stays within 0.55–0.65 on the same data. The `pykalman` library provides a clean implementation. The key tuning parameter `delta` (typically **1e-5 to 1e-3**) controls adaptation speed. One well-regarded quantitative finance textbook concludes: "Kalman filtering is a must in pairs trading."

```python
from pykalman import KalmanFilter
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_covariance=1.0,
    transition_covariance=trans_cov)
state_means, _ = kf.filter(price_a.values)
hedge_ratios = state_means[:, 0]
```

---

## Bottleneck and chunked NumPy arrays dominate rolling Z-score performance

Once pairs are selected and spreads constructed, the Z-score computation itself is the inner loop of the pipeline. The fastest approach avoids pandas entirely for the hot path and operates on raw NumPy arrays with the **Bottleneck** library:

```python
import bottleneck as bn
rolling_mean = bn.move_mean(spreads, window=60, axis=0)
rolling_std = bn.move_std(spreads, window=60, axis=0)
z_scores = (spreads - rolling_mean) / rolling_std
```

Bottleneck's `move_mean` and `move_std` are **60–300× faster** than equivalent NumPy operations on large arrays, using Welford's numerically stable one-pass algorithm internally. Pandas itself uses Bottleneck when installed. For the pandas-native equivalent, separate `rolling().mean()` and `rolling().std()` calls are **5–20× faster** than `rolling().apply(custom_func)`, because the former delegate to optimized C code while the latter invokes a Python callback per window.

For cases where the post-filtering pair count is still too large for a single pass, **chunked processing** is essential:

```python
chunk_size = 5000  # pairs per chunk
for start in range(0, num_pairs, chunk_size):
    spreads = prices_np[:, i_idx[start:end]] - prices_np[:, j_idx[start:end]]
    z = (spreads - bn.move_mean(spreads, window=60, axis=0)) / \
         bn.move_std(spreads, window=60, axis=0)
    # process/store results, then free memory
```

Parallelization across chunks uses **joblib with memory-mapped arrays** to avoid serialization overhead. Save the price matrix to disk via `joblib.dump()`, then each worker loads it with `mmap_mode='r'` for zero-copy shared access. On an 8-core machine, this achieves near-linear speedup for the embarrassingly parallel chunk processing. **Numba JIT** with `prange` provides 100–260× speedup over pure Python for custom rolling functions and can release the GIL for true multi-threaded parallelism. **Polars** offers 3–8× speedup over pandas for rolling operations with lower peak memory, though its ecosystem is less mature for financial-specific workflows.

Using **float32 instead of float64** halves memory with minimal precision loss—float32's ~7 decimal digits are sufficient for price data and relative Z-score measures. Be cautious with very large rolling windows where floating-point accumulation errors can compound.

---

## Walk-forward design must re-select pairs at every step

The walk-forward framework is the backbone of unbiased evaluation. The recommended configuration for pair trading is a **sliding window** (not anchored/expanding) because cointegration relationships are regime-dependent and old data can poison pair selection:

- **Training window**: 6–12 months (covers enough market conditions for robust estimation)
- **Test window**: 1–6 months (the out-of-sample evaluation period)
- **Step size**: Equal to test window length (non-overlapping OOS periods)
- **Rule of thumb**: Training should be 2–4× the test length

**Pairs must be re-selected at every walk-forward step.** Fixed pairs introduce a subtle form of survivorship bias—you're implicitly conditioning on the pair relationship surviving the entire backtest. At each step, run the full selection pipeline (correlation → cointegration → quality filters) using only in-sample data, estimate hedge ratios on in-sample data, then evaluate Z-score signals on the out-of-sample window. Concatenate all OOS results for the final equity curve.

An **embargo period** between training and test windows prevents information leakage from overlapping label horizons. Set the embargo equal to the maximum of the Z-score lookback window and the hedge ratio estimation window. For a 60-minute rolling Z-score, embargo at least 60 minutes; for daily hedge ratios, embargo 1–5 trading days.

For more robust validation, **Combinatorial Purged Cross-Validation (CPCV)** from Marcos López de Prado generates multiple backtest paths from the same data. Partition observations into N sequential groups, select k as test sets across all C(N,k) combinations, and analyze the *distribution* of Sharpe ratios rather than a single point estimate. Select parameters that maximize the **10th percentile** Sharpe ratio to ensure robustness. The `timeseriescv` Python package provides a production-ready `CombPurgedKFoldCV` implementation.

```python
def walk_forward_splits(dates, train_months=12, test_months=6):
    splits, start = [], dates.min()
    while True:
        train_end = start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > dates.max(): break
        splits.append({'train': (start, train_end), 'test': (train_end, test_end)})
        start += pd.DateOffset(months=test_months)
    return splits
```

---

## Parquet for storage, Dagster for orchestration, config-driven everything

**Parquet with Snappy compression** is the clear winner for minute-level financial data storage. It achieves 5–10× compression over raw floats, supports column pruning (read only Close without loading OHLV), and enables predicate pushdown for date-range filtering at the storage level. Partition by `year/month/symbol` to match walk-forward access patterns. Use the `pyarrow` engine exclusively. **Feather** is fastest for ephemeral intermediate results. **CSV should never be used** for production minute data—it's 20× slower than alternatives. **HDF5** is discouraged for new projects due to documented corruption issues and inferior querying capabilities.

**DuckDB** serves as an excellent analytics layer on top of Parquet files, providing full SQL with window functions directly on stored data without loading into memory. **ArcticDB** (from Man Group) is purpose-built for financial time series with native append support, time-travel versioning, and S3-compatible backends—ideal for production streaming workloads.

The project should follow a modular structure with strict separation of concerns:

```
src/
├── data/        # ingest.py, clean.py, storage.py
├── pairs/       # selection.py, hedge_ratio.py, validation.py  
├── signals/     # zscore.py, features.py, filters.py
├── backtest/    # walk_forward.py, portfolio.py, metrics.py
└── utils/       # logging.py, parallel.py
```

All parameters belong in a centralized `params.yaml`—Z-score windows, entry/exit thresholds, walk-forward lengths, pair selection criteria. Use **Pydantic** for validated config loading. **DVC** handles data versioning (large Parquet files tracked via pointer files in Git), **MLflow** tracks experiment metrics (Sharpe ratios, drawdowns, pair counts per walk-forward step), and **Dagster** provides asset-centric orchestration with built-in data lineage and partitioning. For simpler setups, DVC pipelines with Make suffice.

---

## Ten pitfalls that will silently destroy your backtest

The most dangerous errors in pair trading pipelines are silent—they produce plausible-looking but inflated results.

**Lookahead bias is the #1 killer.** Three common manifestations: (1) selecting pairs using full-period cointegration then backtesting on the same period, (2) computing Z-score mean and standard deviation over the entire series rather than rolling, and (3) using bar[t] close price to generate a signal executed at bar[t] close. The fix for all three is the same: strict temporal separation. Rolling Z-scores use only past data. Signals generated from bar[t-1] trigger execution at bar[t] open via `.shift(1)`. Walk-forward ensures pair selection never sees test data.

**The multiple testing problem at 124K+ pairs is fatal without correction.** At α=0.05, expect ~6,237 spurious "cointegrated" pairs from random chance. Apply Benjamini-Hochberg FDR correction via `statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')`. Pre-filtering by sector reduces the effective test count by 10–100×, making corrections less aggressive while adding economic justification.

**Survivorship bias inflates returns by 1.6% annually** on average, and can inflate Sharpe ratios by up to 0.5 points. A survivorship-biased dataset going back 10 years is missing ~75% of stocks that actually traded. Use bias-free data sources (Norgate, CRSP, QuantRocket EDI). When one leg of a pair delists, force-close the position at the last available price and include that loss.

**Transaction costs dominate at minute frequency.** Research shows that pairs trading profits frequently lose statistical significance after accounting for just 20 bps of market impact. One study found gross returns of 249% but net losses of -40% to -1,138% after costs on 5-minute data. Model realistic friction: commission + half bid-ask spread × 2 legs × 2 (entry + exit) = typically 4× half-spread per round trip.

**Stale hedge ratios cause Z-score drift.** A static hedge ratio becomes increasingly wrong as the relationship evolves, generating false signals. Kalman filtering provides continuous adaptation. At minimum, re-estimate via rolling OLS daily.

**Cointegration breakdown goes undetected without monitoring.** Use rolling ADF tests, CUSUM charts, or Bayesian online changepoint detection. Set stop-losses at |Z| > 4 and time-based exits at 2× the estimated half-life. Never assume mean reversion is permanent.

**Correlation does not imply cointegration.** Two series can have ρ=0.998 yet fail cointegration testing (p=0.258). Always test cointegration explicitly. Additionally, two stocks may appear cointegrated only because both are cointegrated with the market—include SPY as a control variable.

**Overnight gaps contaminate intraday Z-scores.** The 3:59 PM → 9:30 AM jump can spike the spread due to overnight news. Either reset Z-score calculations at market open or exclude cross-day bars from rolling windows.

---

## Conclusion

Building this pipeline is fundamentally a **data reduction problem**, not a computation problem. The architectural insight is that the 500-asset universe should touch minute-level data only after being filtered to dozens of pairs through cheap daily-frequency tests. The recommended technology stack—Bottleneck for rolling statistics, joblib with memory-mapped arrays for parallelism, Parquet for storage, Kalman filtering for hedge ratios, and Benjamini-Hochberg for multiple testing correction—addresses each bottleneck with the most efficient available tool.

The most underappreciated design decision is **re-selecting pairs at every walk-forward step**. Fixed pair lists are a form of lookahead that produces smooth equity curves from relationships that wouldn't have been identifiable in real time. Combined with proper embargo periods and CPCV validation, this walk-forward discipline is the difference between a backtest that matches live performance and one that silently overstates returns by 30–50%.

Half-life of mean reversion deserves special attention as a unifying metric: it determines the Z-score lookback window, sets time-based stop-losses, and serves as a pair quality filter. Pairs with half-lives between **5–50 bars** occupy the sweet spot—fast enough to be tradeable, slow enough to not be measurement noise. Computing it is trivial (fit AR(1) to spread, `half_life = -ln(2)/ln(θ)`), and it should be a first-class citizen in the pair selection pipeline alongside cointegration p-values and Hurst exponents.