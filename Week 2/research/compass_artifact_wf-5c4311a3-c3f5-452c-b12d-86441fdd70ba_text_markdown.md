# High-performance rolling Z-scores for pair trading in Pandas

**The single most impactful design choice for a fast pair trading pipeline is data shape: a wide price matrix (tickers as columns, dates as index) combined with bottleneck's C-compiled rolling functions on extracted NumPy arrays delivers 10–100× speedups over naive Pandas patterns.** Most performance failures in quant Pandas code trace not to algorithmic complexity but to structural mistakes — row-level Python loops, object-dtype columns, and long-format DataFrames that force groupby overhead. This guide covers every layer of the performance stack: data layout, vectorization strategy, library selection, universe reduction, caching, and correctness testing.

---

## 1. Every bar-level calculation must be vectorized

Four operations form the hot path of any pair trading pipeline, and all four must execute as vectorized array operations — never as Python-level loops over rows.

**Spread computation** (`spread = price_A - beta * price_B`) is simple column arithmetic that Pandas delegates directly to NumPy's compiled C ufuncs. Vectorized spread calculation on a 100,000-row DataFrame completes in microseconds; a row-by-row loop with `iterrows` takes seconds — a **~1,300× difference** in measured benchmarks. **Rolling mean and rolling standard deviation** (`spread.rolling(window).mean()` and `.std()`) use Pandas' internal Cython accumulator algorithms that run in O(N) time with a single pass through the data. The built-in rolling mean benchmarks at roughly **258 µs** on a typical series, whereas `rolling().apply(custom_mean_func)` falls back to Python dispatch per window and clocks **5.31 ms** — a 20× penalty for doing the same math. **Z-score computation** (`(spread - rolling_mean) / rolling_std`) is element-wise arithmetic on three aligned arrays, dispatched entirely to NumPy ufuncs with zero interpreter overhead per element.

The reason vectorization matters this much is Pandas indexing overhead. Accessing a single element in a Pandas Series triggers roughly 100 Python function calls for index alignment, type checking, and NaN handling. NumPy array indexing goes straight to C extensions — benchmarked at **100×+ faster** for scalar access. When this overhead multiplies across hundreds of thousands of rows, the cumulative cost dominates runtime.

### What is acceptable to loop over

Not every loop is a performance sin. The key distinction is **frequency**: operations that run once or a handful of times per trading day can afford Python-level iteration; operations that touch every row cannot.

Loops are acceptable for **pair selection** (iterating over N×(N-1)/2 candidate pairs to run cointegration tests — each iteration calls a vectorized statistical test on the full price series), **parameter tuning** (grid search over window sizes and Z-score thresholds, where the inner computation is fully vectorized), and **hedge ratio estimation** (per-pair OLS regression, where statsmodels is itself vectorized internally). In live trading, processing one bar at a time in an event loop is architecturally necessary, but the calculation within each bar should reference pre-computed vectorized results rather than recomputing rolling statistics from scratch.

Operations that must **never** be looped include row-level spread calculation, per-bar rolling mean or standard deviation recomputation, any use of `iterrows()` or `apply(axis=1)` for Z-score logic, and per-row position sizing. Position signals should use `np.where(zscore > threshold, -1, np.where(zscore < -threshold, 1, 0))` — fully vectorized boolean logic, not a conditional inside a for-loop.

---

## 2. Wide format is the only correct data shape

Three common layouts exist for multi-asset financial data, and they are not equally suited to vectorized rolling calculations.

**Long format** (stacked rows with a ticker column) is natural for database storage and tidy-data visualization, but it is the worst choice for pair trading computation. Computing rolling statistics requires `groupby('ticker').rolling()`, which adds Python-level group-iteration overhead. Cross-ticker spread computation demands a pivot or merge before any math can happen. **Multi-index DataFrames** improve on long format by making group-level operations cleaner, but still require group iteration internally and add indexing overhead compared to simple column access.

**The wide price matrix** — tickers as columns, a DatetimeIndex as the row index — is optimal for three reasons. First, a single `df.rolling(window).mean()` call computes the rolling mean for every ticker simultaneously in one C-level pass through the underlying 2D NumPy array. With 500 tickers, long format requires 500 separate rolling computations with Python overhead per group; wide format does it in one operation. Second, spread computation across many pairs is a single broadcasting operation: construct two DataFrames of leg-A and leg-B prices (each with shape T × num_pairs), multiply leg-B by a vector of hedge ratios, and subtract. All pair spreads materialize in one vectorized expression. Third, a wide DataFrame with homogeneous float64 columns stores its data in a single contiguous NumPy array block, giving CPU cache locality that benefits sequential rolling operations.

### Memory layout matters at the NumPy level

Pandas DataFrames default to **C-contiguous** (row-major) memory layout, meaning all ticker prices for the same date are adjacent in memory. Rolling operations scan down columns (along the time axis), which means jumping across rows — potentially non-sequential memory access. For pure NumPy or Numba inner loops, converting to **F-contiguous** (column-major) layout with `np.asfortranarray()` aligns consecutive time points for the same ticker adjacently in memory, which can yield measurable improvements. Benchmarks show contiguous arrays run **2.3× faster** in Numba JIT-compiled functions versus strided non-contiguous arrays (88.5 µs vs 201 µs on 100,000 elements). For standard Pandas rolling, the built-in accumulator algorithms minimize cache misses well enough that contiguity rarely becomes the bottleneck — but when dropping to raw NumPy, it matters.

On `.values` vs `.to_numpy()`: for homogeneous float64 DataFrames (the standard case in pair trading), both return a view of the same underlying ndarray with no measurable performance difference. Use `.to_numpy()` by default for explicit dtype guarantees. If your DataFrame ever contains nullable integer or string columns, `.values` can silently return an object array, destroying all performance.

---

## 3. Bottleneck is the fastest single-threaded path for rolling statistics

The performance hierarchy for rolling mean and standard deviation, from fastest to slowest, reveals large gaps between approaches.

**Bottleneck's `move_mean` and `move_std`** operating on raw NumPy arrays are the fastest single-threaded option. Written in pure C and compiled via Cython, they use O(N) online algorithms — one pass through the data, updating running statistics incrementally. Official benchmarks show **60–182× speedups** over equivalent NumPy operations on large arrays, and a practical **3–10× speedup** over Pandas' built-in rolling (which is itself Cython-optimized but carries Pandas overhead for index alignment and type checking). The `move_std` function uses **Welford's one-pass algorithm**, which is numerically stable when the mean is large relative to the standard deviation — important for price-level spreads.

Critical caveats with bottleneck: the **default ddof is 0** (population standard deviation), while Pandas defaults to **ddof=1** (sample). You must explicitly pass `ddof=1` to match Pandas output. Bottleneck also **cannot handle Inf values** — when Inf enters a window, the output becomes NaN and stays NaN for all subsequent values. Always sanitize your spread series before passing it to bottleneck. NaN handling is configurable via `min_count`: setting `min_count=window` (the default) means any NaN in the window produces NaN output, while `min_count=1` computes with whatever non-NaN values are available.

**Pandas' built-in `rolling().mean()` and `.std()`** use their own Cython implementation (not bottleneck, even when bottleneck is installed as an optional dependency). These are solid O(N) algorithms that benchmark at roughly **258 µs** for rolling std on a typical series. For many pipelines, this is fast enough — the overhead versus bottleneck only becomes significant at scale (many pairs, high-frequency data).

**NumPy stride tricks** (`np.lib.stride_tricks.sliding_window_view`) create a memory-efficient view with an extra dimension representing the rolling window, then aggregate with `.mean(axis=-1)`. This approach is **O(N×W)** — each window element is touched during aggregation. NumPy's own documentation warns that for window size 100, this can be **100× slower** than specialized O(N) algorithms. It benchmarks 2–5× faster than Pandas for tiny windows (2–10), but the advantage disappears or reverses for typical pair trading windows of 20–60 bars. Avoid this in production.

**Numba JIT compilation** excels at custom rolling functions that bottleneck doesn't provide. Pandas' `rolling().apply()` with `engine='numba'` is **~20× faster** than the default Cython engine (188 ms vs 3.92 s on 1M rows). For standard rolling mean and std, though, Numba offers no advantage over bottleneck's pre-compiled C. Reserve Numba for rolling correlation, rolling regression, Kalman filter updates, or any custom windowed logic.

**Polars** offers **~2–5× speedup** over Pandas for rolling statistics, powered by a multi-threaded Rust engine with SIMD vectorization. If building a new pipeline from scratch, Polars is compelling — but verify numerical precision requirements, as subtle floating-point differences have been reported in rolling mean output.

### The optimal computation pipeline

The fastest practical path for a Z-score calculation extracts data from Pandas into NumPy arrays, uses bottleneck for rolling statistics, and wraps results back in Pandas at the end. Spread is computed as pure NumPy arithmetic (`price_a.to_numpy() - beta * price_b.to_numpy()`). Rolling mean and std use `bn.move_mean()` and `bn.move_std()` with `ddof=1`. The Z-score is a final element-wise NumPy division. This pattern avoids Pandas overhead entirely for the computation-intensive steps while preserving Pandas' convenient indexing for I/O and result inspection.

---

## 4. Seven Pandas mistakes that destroy pair trading performance

**Using `iterrows()` is the single most expensive mistake.** On a 10,000-row DataFrame, vectorized column addition takes **0.001 seconds**; `iterrows()` takes **0.74 seconds** — a 740× penalty. At 100 million rows (not unreasonable for a year of minute-bar data across many pairs), `iterrows()` never finishes. The function creates a Python Series object for every row, incurring object creation and type-checking overhead that completely dominates any useful computation. `apply(axis=1)` is better but still ~100× slower than vectorized operations, because it creates a Series from each row and performs index lookups three times per row.

**Object dtype columns** carry a **~25× arithmetic penalty** versus native float64 columns. Object dtype stores generic Python objects, requiring Python-level iteration for each operation and preventing SIMD/vectorized C routines. Each element incurs boxing/unboxing overhead: a float64 value occupies exactly 8 bytes, while a Python-wrapped float object takes 28+ bytes plus an 8-byte pointer. This silently occurs when DataFrames contain mixed types, when string columns aren't converted to `category` or `StringDtype`, or when `.values` returns an object array on a nullable-integer column.

**Excessive method chaining creates hidden intermediate copies.** A chain like `df1 + df2 + df3 + df4` allocates three intermediate DataFrames. For DataFrames over 100,000 rows, `pd.eval('df1 + df2 + df3 + df4')` evaluates the entire expression in one pass using `numexpr`. Below 10,000 rows, the `eval` overhead itself exceeds the savings.

**Using `groupby().apply()` with non-vectorized lambdas** is a common trap. A `groupby('id').apply(lambda g: custom_function(g))` call on 1M rows with 1,000 groups took **3.4 seconds** in benchmarks; the equivalent fully vectorized approach using `isin()` and boolean masking took **129 ms** — a 26× improvement. Always prefer built-in aggregation methods (`sum`, `mean`, `std`) over `apply()` with lambdas, and use `sort=False` and `observed=True` for free speedups.

**Repeated `.loc`/`.iloc` indexing in loops** wastes cycles on per-call validation. For single-cell access in unavoidable loops, use `.iat[i, j]` / `.at[label, col]` — scalar fast-path accessors with minimal overhead. Better yet, extract columns as NumPy arrays before the loop and index those directly. **Building DataFrames row by row with `df.append()`** in a loop produces O(N²) behavior from copying the entire DataFrame each iteration; collect results in a list and call `pd.concat()` once.

The overhead of DataFrame versus raw NumPy in arithmetic is stark: `pd.Series * pd.Series` benchmarks at **88.5 µs** versus `np.array * np.array` at **1.21 µs** — a 73× difference. For scalar access, Series indexing costs **168 µs** versus NumPy's **1 µs**. The pattern is clear: use Pandas for data management, drop to NumPy for computation.

---

## 5. Precompute rolling statistics once, reuse everywhere

Rolling mean and standard deviation should be **precomputed and stored** whenever they feed into multiple downstream calculations — which in pair trading they almost always do. The rolling mean appears in both the Z-score numerator and potentially in spread regime detection; the rolling std appears in the Z-score denominator and in volatility-adjusted position sizing. Computing each once and referencing the stored result is trivially cheaper than recomputing.

For **parameter sweeps** across multiple lookback windows (e.g., testing windows of 10, 20, 50, 100, 200), precompute all rolling statistics upfront in a dictionary keyed by window size. Each window's rolling mean is computed once; subsequent access is O(1) lookup. The memory cost is modest: a single rolling column for 1 million rows of float64 occupies ~8 MB. For 500 stocks × 5 windows × 2 statistics (mean, std), that totals ~40 GB — too much to cache simultaneously. The practical rule: precompute rolling statistics for the current parameter set being evaluated, discard when moving to the next.

For **live/streaming applications**, use online algorithms — Welford's method updates rolling variance in O(1) per new observation, avoiding full-window recomputation. Cache invalidation is the real danger in live systems: stale cached rolling statistics produce incorrect signals, which is worse than the cost of recomputation. Pandas' `rolling().mean()` on 1 million points takes only 5–10 ms, so recomputation is rarely the bottleneck. **Recompute when underlying data changes; cache when the same statistic feeds multiple consumers on static data.**

---

## 6. Reduce the pair universe before touching minute bars

The combinatorial explosion of candidate pairs makes pre-screening not optional but essential. For N stocks, the number of candidate pairs is N×(N-1)/2: **100 stocks produce 4,950 pairs; 500 stocks produce 124,750; 3,000 stocks produce 4.5 million.** If each pair requires a minute-level cointegration test over one year (~100,000 minute bars), the compute cost without filtering is prohibitive.

The recommended staged pipeline applies increasingly expensive filters in order of computational cost:

- **Liquidity filter** — remove stocks below a minimum average daily volume threshold (O(N), essentially free)
- **Sector or cluster grouping** — restrict pairs to within GICS sectors or ML-derived clusters using PCA + DBSCAN/OPTICS (reduces a 500-stock universe from 124,750 pairs to roughly 12,250)
- **Correlation pre-filter** — compute the full daily return correlation matrix (O(N²×T) but uses BLAS routines; completes in under 1 second for 500 stocks) and retain only pairs with |ρ| above a threshold like 0.7
- **Daily cointegration screen** — run ADF tests on daily closes for surviving pairs, keeping those with p-value < 0.05 (typically a 5% pass rate)
- **Minute-level rolling analysis** — compute spreads, rolling Z-scores, and half-life of mean reversion only for the final survivors

A concrete example: starting from 500 stocks (124,750 raw pairs), sector restriction reduces to ~12,250, correlation filtering to ~2,450, and daily cointegration screening to ~122 pairs. If minute-level analysis takes 1 second per pair, the raw approach requires **34.6 hours**; the filtered approach takes **2 minutes**. Research from Huck and Afawubo (2015) demonstrated that dropping 80% of pairs whose returns diverge most before running Johansen cointegration tests on S&P 500 constituents dramatically reduced computation with minimal loss of profitable pairs.

An important caveat: high correlation does not imply cointegration, and cointegrated pairs can have low correlation. Correlation filtering is a cheap heuristic, not a substitute for formal cointegration testing. Its value is in eliminating obviously unrelated pairs before the expensive step.

---

## 7. Testing vectorized rolling calculations for correctness

Vectorized code fails silently — an off-by-one error in a rolling window produces plausible-looking numbers that are subtly wrong. Rigorous testing requires multiple approaches working together.

**Reference implementation comparison** is the foundation. Write a deliberately slow, obviously correct loop-based implementation of each rolling statistic: iterate from index `window-1` to `len(series)`, compute `np.nanmean` of the slice `[i-window+1 : i+1]`, and store the result. Compare against the vectorized version using `np.testing.assert_allclose(fast, slow, rtol=1e-10, equal_nan=True)`. The `equal_nan=True` flag ensures NaN positions must match exactly. For DataFrame-level comparison, `pd.testing.assert_frame_equal` with `check_exact=False` and a tight `rtol` catches both value and structural discrepancies.

**Off-by-one errors** are the most common rolling window bug. The `min_periods` parameter defaults to `window` for integer windows but to 1 for offset-based windows — a silent behavioral difference that changes how many leading NaN values appear. The `center=True` parameter shifts labels to the window center but has a known issue (pandas #59252): with `min_periods=1`, the window is not symmetric at edges, causing the effective center to shift. The `closed` parameter controls endpoint inclusion and defaults differently for fixed versus offset windows. Always explicitly test that the first `window-1` values are NaN (when using default `min_periods`), and verify the exact index alignment of the output.

**NaN handling diverges between libraries.** Pandas rolling excludes NaN from calculations by default (nanmean behavior). Plain `np.mean()` propagates NaN — returning NaN if any input value is NaN. Bottleneck with default `min_count=None` returns NaN if any value in the window is NaN; with `min_count=1`, it computes from available non-NaN values. If your reference implementation uses `np.mean()` over a slice and the vectorized version uses Pandas rolling, they will disagree on windows containing NaN. Use `np.nanmean()` in the reference for a fair comparison.

**Edge cases that must be explicitly tested:**

- **Constant series** — `pd.Series([5,5,5,5,5]).rolling(3).std()` returns 0.0, and dividing by it produces Inf. Guard Z-scores with `np.where(rolling_std > epsilon, z, 0.0)`
- **Missing data gaps** — minute bars with no trades leave stale prices in the rolling window. Time-based windows (`rolling('30T')`) handle this correctly; count-based windows do not
- **Duplicate timestamps** — can produce unexpected rolling results. Always `sort_index()` before rolling
- **Very short series** — fewer data points than the window size should produce all-NaN output

**Property-based testing** with the Hypothesis library generates random inputs to find edge cases automatically. Key properties to assert: output length equals input length, rolling mean of a constant equals that constant, rolling mean is bounded by the window's min and max, and rolling Z-score over a sufficiently long series has mean approximately 0 and standard deviation approximately 1.

---

## If speed matters, do this / do not do this

### ✅ DO THIS

1. **Structure data as a wide price matrix** — tickers as columns, DatetimeIndex as rows — enabling single-call rolling operations across all tickers simultaneously
2. **Extract to NumPy arrays before computation** — use `.to_numpy()` to bypass Pandas indexing overhead (73× faster for arithmetic, 168× faster for scalar access)
3. **Use bottleneck `move_mean` / `move_std` on raw arrays** — 3–10× faster than Pandas rolling, the fastest single-threaded option for standard rolling statistics
4. **Set `ddof=1` explicitly in bottleneck** — its default is population std (ddof=0), mismatching Pandas' sample std
5. **Sanitize Inf values before bottleneck** — Inf corrupts all subsequent output permanently
6. **Compute spread, rolling mean, rolling std, and Z-score as vectorized array operations** — each is a one-liner on aligned arrays
7. **Precompute rolling statistics once and reuse** when the same statistic feeds multiple downstream calculations
8. **Apply staged universe filtering** — liquidity → sector/cluster → correlation → daily cointegration → minute-level analysis — to reduce pairs by 1,000× before expensive computation
9. **Use `np.where()` for vectorized signal generation** instead of conditional logic inside loops
10. **Use F-contiguous arrays (`np.asfortranarray`)** when dropping to NumPy/Numba for column-wise rolling operations
11. **Reserve Numba for custom rolling functions** (rolling correlation, Kalman filters) where bottleneck has no equivalent
12. **Validate against a slow loop-based reference** using `np.testing.assert_allclose` with `equal_nan=True`
13. **Test edge cases explicitly** — constant series, zero std, NaN gaps, duplicate timestamps, series shorter than window
14. **Use `.iat` / `.at` for scalar access** in unavoidable loops, never `.loc` / `.iloc` for single cells

### ❌ DO NOT DO THIS

1. **Never use `iterrows()`, `itertuples()`, or `apply(axis=1)` for bar-level calculations** — 100–740× slower than vectorized equivalents
2. **Never use long-format DataFrames for rolling calculations** — forces `groupby().rolling()` with Python-level group iteration
3. **Never allow object dtype in numeric columns** — 25× arithmetic penalty from Python-level boxing/unboxing
4. **Never use `rolling().apply(custom_func)` without `engine='numba'`** — the Cython engine calls back into Python per window (20× slower than Numba engine)
5. **Never use `np.lib.stride_tricks.sliding_window_view` for windows ≥ 20** — O(N×W) scaling makes it uncompetitive versus O(N) algorithms
6. **Never build DataFrames row by row with `append()` in a loop** — O(N²) from full-copy each iteration; collect in a list, `pd.concat()` once
7. **Never use `groupby().apply(lambda)` when a built-in aggregation exists** — built-in methods are Cython-optimized, lambdas force Python dispatch
8. **Never index `.loc`/`.iloc` repeatedly in a loop** — extract columns as arrays first, index those
9. **Never assume ddof defaults match across libraries** — Pandas uses 1, bottleneck and TA-Lib use 0, NumPy uses 0
10. **Never assume NaN handling is consistent across libraries** — Pandas rolling skips NaN, `np.mean` propagates NaN, bottleneck behavior depends on `min_count`
11. **Never skip off-by-one testing for rolling windows** — `min_periods`, `center`, and `closed` defaults differ between fixed and offset windows
12. **Never divide by rolling std without guarding against zero** — constant-price windows produce std=0 and Z-score=Inf
13. **Never run minute-level cointegration tests on an unfiltered universe** — 500 stocks × 124,750 pairs × 100K minute bars is computationally intractable without pre-screening
14. **Never use Pandas for inner computation loops** — use it for I/O and alignment, drop to NumPy/bottleneck/Numba for all math, wrap results back in Pandas at the end