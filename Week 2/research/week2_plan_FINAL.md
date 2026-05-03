# Week 2: Z-Score Signal Engine — Implementation Plan v11 (Final)

## Quick Reference

| Item | Value |
|------|-------|
| **Pairs** | Primary + secondary from Week 1 near-misses. Optional ETF pair as robustness check |
| **Data** | ohlc-2022-XX.zip, 1-min bars, `close` column, nanosecond epoch |
| **Formation** | Jan–Jun 2022 → hedge ratio, half-life, Hurst, quantile thresholds |
| **Trading** | Jul–Dec 2022 → Z-scores, signals, all evaluation |
| **Deliverable** | Signal Logic Document |
| **Est. time** | ~5.5 hours |

---

## Step 0: Pair Selection [15 min]

### Framing

Week 1 screened ~500 stocks, ran ADF with BH-FDR correction, found zero cointegrated pairs at conventional significance levels. That result stands — we do not re-run ADF or pretend otherwise.

**The pairs selected here are candidate pairs for signal-engine validation, not economically credible trading pairs.** Week 2's goal is to build and test the Z-score engine. These pairs provide realistic input data with varying spread properties. Whether any pair would be profitable to trade is a Week 3 question that requires backtest evaluation with transaction costs.

**Primary pair:** Lowest ADF p-value from Week 1 that also has economic logic (same sector, competitors, or supply-chain link). This is the pair with the strongest — though not statistically significant — evidence of mean reversion in your existing results.

**Secondary pair:** Second-lowest p-value from Week 1, preferably a different sector. Provides a comparison input with different spread characteristics.

**Robustness check (optional):** If ETF tickers (SPY/IWM, XLE/XOM) exist in the dataset, run the engine on one ETF pair as external validation on a well-documented benchmark.

```python
from pathlib import Path

data_dir = "path/to/extracted/ohlc-2022"

# Check ETF availability for robustness check
for t in ['SPY', 'IWM', 'XLE', 'XOM']:
    count = len(list(Path(data_dir).glob(f"{t}_*.csv")))
    if count > 0:
        print(f"  ✓ {t}: {count} files (available for robustness check)")

# Candidate pairs for engine validation — NOT claimed as trading pairs
pairs = {
    'primary':   ('TICKER_1A', 'TICKER_1B'),  # Week 1 lowest ADF p-value + economic logic
    'secondary': ('TICKER_2A', 'TICKER_2B'),  # Second-lowest, different sector
    # 'robustness': ('SPY', 'IWM'),            # Uncomment if available
}
```

### Record in Document

For each pair: tickers, sector, economic rationale, and ADF p-value from Week 1. Explicitly state: "These are candidate pairs selected for engine validation. None passed formal cointegration testing in Week 1. The signal engine is being tested on spreads with varying degrees of mean-reversion evidence — the primary pair showed the strongest (though non-significant) evidence, the secondary pair weaker evidence."

---

## Step 1: Load and Align [30 min]

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_pair(data_dir, ticker_a, ticker_b):
    frames = {}
    for ticker in [ticker_a, ticker_b]:
        files = sorted(Path(data_dir).glob(f"{ticker}_*.csv"))
        assert len(files) > 0, f"No files for {ticker}"
        dfs = []
        for f in files:
            chunk = pd.read_csv(f, usecols=['close', 'window_start'])
            chunk['timestamp'] = pd.to_datetime(chunk['window_start'], unit='ns')
            chunk = chunk.set_index('timestamp')[['close']]
            dfs.append(chunk)
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        frames[ticker] = combined['close'].rename(f'close_{ticker.lower()}')
    result = pd.concat(frames.values(), axis=1, join='inner').dropna()
    result = result.astype('float64')
    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()
    return result

data = {}
for label, (a, b) in pairs.items():
    df = load_pair(data_dir, a, b)
    data[label] = {
        'df_all': df,
        'formation': df[:'2022-06-30'],
        'trading': df['2022-07-01':],
    }
    print(f"{label} ({a}/{b}): {len(df):,} total, "
          f"{len(data[label]['formation']):,} formation, "
          f"{len(data[label]['trading']):,} trading")
```

### Alignment & Session Quality Audit

After loading, audit the data to catch microstructure issues before they corrupt spread statistics.

```python
for label, d in data.items():
    df = d['df_all']
    a, b = pairs[label]
    
    # Expected bars: ~390/day × 252 trading days = ~98,280
    n_trading_days = df.index.normalize().nunique()
    expected_bars = n_trading_days * 390
    actual_bars = len(df)
    coverage = actual_bars / expected_bars * 100
    
    # Timestamp gaps: how many minute-gaps exist?
    diffs = df.index.to_series().diff().dropna()
    median_gap = diffs.median()
    gaps_gt_5min = (diffs > pd.Timedelta(minutes=5)).sum()
    
    # Session boundary check: first and last bar per day
    first_bar = df.groupby(df.index.date).apply(lambda x: x.index[0].time())
    last_bar = df.groupby(df.index.date).apply(lambda x: x.index[-1].time())
    
    print(f"\n{label} ({a}/{b}) — Data Quality:")
    print(f"  Trading days: {n_trading_days}")
    print(f"  Expected bars (~390/day): {expected_bars:,}")
    print(f"  Actual bars after alignment: {actual_bars:,} ({coverage:.1f}% coverage)")
    print(f"  Median bar spacing: {median_gap}")
    print(f"  Gaps > 5 min (session breaks): {gaps_gt_5min}")
    print(f"  Typical first bar: {first_bar.mode().iloc[0]}")
    print(f"  Typical last bar: {last_bar.mode().iloc[0]}")

# Optional: restrict to regular trading hours (9:30-16:00 ET = 14:30-21:00 UTC)
# Uncomment if pre/post market bars are present and should be excluded:
# for label, d in data.items():
#     df = d['df_all']
#     mask = (df.index.time >= pd.Timestamp('14:30').time()) & \
#            (df.index.time <= pd.Timestamp('21:00').time())
#     d['df_all'] = df[mask]
#     d['formation'] = d['df_all'][:'2022-06-30']
#     d['trading'] = d['df_all']['2022-07-01':]
```

**Record in document:** Include the data quality table. If coverage is significantly below 90%, or if many gaps > 5 min exist within sessions, note this as a data limitation. If pre/post market bars are present, state whether they were included or excluded and why.

---

## Step 2: Define All Functions [20 min]

### 2a. Spread Construction with OLS Hedge Ratio

```python
def estimate_hedge_ratio(close_a, close_b):
    """
    OLS regression: log(A) = α + β·log(B) + ε
    Estimated on formation period ONLY.
    Returns (alpha, beta).
    """
    y = np.log(close_a).values
    X = np.column_stack([np.ones(len(close_b)), np.log(close_b).values])
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    return params[0], params[1]  # alpha, beta

def compute_spread(close_a, close_b, alpha, beta):
    """
    Spread = log(A) - α - β·log(B)
    Uses formation-period hedge ratio applied to full series.
    """
    return np.log(close_a) - alpha - beta * np.log(close_b)
```

### 2b. Spread Characterization

```python
def compute_half_life(spread):
    """
    OU discretization: Δs(t) = a + λ·s(t-1) + ε
    Half-life = -ln(2) / λ
    Intercept included to avoid bias for non-zero mean spreads.
    """
    lag = spread.shift(1)
    delta = spread.diff()
    df_reg = pd.DataFrame({'delta': delta, 'lag': lag}).dropna()
    X = np.column_stack([np.ones(len(df_reg)), df_reg['lag'].values])
    y = df_reg['delta'].values
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    a, lam = params[0], params[1]
    half_life = -np.log(2) / lam if lam < 0 else np.inf
    return half_life, lam

def hurst_exponent(series, max_lag=100):
    """
    H < 0.5: mean-reverting | H = 0.5: random walk | H > 0.5: trending
    
    NOTE: This is a simple variance-ratio estimator. On noisy minute-bar data
    it can be unstable. Treat as supporting diagnostic, not confirmation.
    """
    vals = series.dropna().values
    lags = range(2, max_lag)
    tau = [np.std(vals[lag:] - vals[:-lag]) for lag in lags]
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
```

### 2c. Z-Score Engine

```python
def compute_zscore(spread, window, min_periods=None, eps=1e-10):
    """Rolling Z-score. Vectorized Pandas — O(N) Cython accumulators."""
    if min_periods is None:
        min_periods = window // 2
    rolling_mean = spread.rolling(window, min_periods=min_periods).mean()
    rolling_std  = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    zscore       = (spread - rolling_mean) / rolling_std.clip(lower=eps)
    return pd.DataFrame({
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'zscore': zscore,
    }, index=spread.index)
```

### 2d. Signal Generation (Stateless Classification)

```python
def generate_signals(zscore, entry_z=2.0, exit_z=0.0):
    """
    Raw zone classification.
    +1 = long zone | -1 = short zone | 0 = exit zone | NaN = dead zone
    """
    conditions = [
        zscore > entry_z,
        zscore < -entry_z,
        (zscore >= -exit_z) & (zscore <= exit_z),
    ]
    return pd.Series(
        np.select(conditions, [-1, 1, 0], default=np.nan),
        index=zscore.index, name='raw_signal'
    )
```

### 2e. Stateful Position Logic

```python
from numba import njit

@njit(cache=True)
def _position_state_machine(zscores, entry_z, exit_z):
    """
    Path-dependent position tracking.
    Ensures: one entry per excursion, exit before re-entry.
    
    States: 0=flat, 1=long spread, -1=short spread
    
    Returns: position array {-1, 0, +1}
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    state = 0
    
    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            positions[i] = state
            continue
        
        if state == 0:
            if z < -entry_z:
                state = 1
            elif z > entry_z:
                state = -1
        elif state == 1:
            if z >= -exit_z:
                state = 0
        elif state == -1:
            if z <= exit_z:
                state = 0
        
        positions[i] = state
    return positions

def generate_positions(zscore, entry_z=2.0, exit_z=0.0):
    """Pandas wrapper for state machine."""
    pos = _position_state_machine(
        zscore.values.astype(np.float64), entry_z, exit_z
    )
    return pd.Series(pos, index=zscore.index, dtype='int8', name='position')

def count_trades(positions):
    """Count actual round-trip entries, not bars in position."""
    entries = (positions != 0) & (positions.shift(1).fillna(0) == 0)
    return entries.sum()
```

### 2f. Pipeline

```python
def run_pipeline(close_a, close_b, alpha, beta, window,
                 entry_z=2.0, exit_z=0.0):
    """Full pipeline: prices → spread → Z-score → signals → positions."""
    df = pd.DataFrame({'close_a': close_a, 'close_b': close_b})
    df['spread'] = compute_spread(df['close_a'], df['close_b'], alpha, beta)
    df = df.join(compute_zscore(df['spread'], window=window))
    df['raw_signal'] = generate_signals(df['zscore'], entry_z=entry_z, exit_z=exit_z)
    df['position'] = generate_positions(df['zscore'], entry_z=entry_z, exit_z=exit_z)
    return df
```

### Execution Convention

The Z-score at bar[t] is computed from `close[t]` and rolling statistics through bar[t]. The `position[t]` column represents the desired position based on information available at bar[t].

**For any future PnL computation (Week 3), execution must be lagged by one bar:** `position[t]` determines the trade executed at bar[t+1]'s open. This is implemented as `df['position'].shift(1)` when computing returns. The Week 2 deliverable outputs `position[t]` as-is — the shift is Week 3's responsibility, but this convention must be stated explicitly in the Signal Logic Document to avoid subtle lookahead bias.

---

## Step 3: Spread Characterization — Formation Period [30 min]

```python
char_results = {}

for label, d in data.items():
    fa, fb = d['formation'].iloc[:, 0], d['formation'].iloc[:, 1]
    
    # Estimate hedge ratio on formation
    alpha, beta = estimate_hedge_ratio(fa, fb)
    spread_f = compute_spread(fa, fb, alpha, beta)
    
    # Half-life and Hurst
    hl, lam = compute_half_life(spread_f)
    H = hurst_exponent(spread_f)
    
    # Stability: 3-month vs 6-month
    fa_3m, fb_3m = fa['2022-04-01':], fb['2022-04-01':]
    alpha_3m, beta_3m = estimate_hedge_ratio(fa_3m, fb_3m)
    spread_3m = compute_spread(fa_3m, fb_3m, alpha_3m, beta_3m)
    hl_3m, _ = compute_half_life(spread_3m)
    H_3m = hurst_exponent(spread_3m)
    
    char_results[label] = {
        'alpha': alpha, 'beta': beta,
        'half_life_6m': hl, 'half_life_3m': hl_3m,
        'hurst_6m': H, 'hurst_3m': H_3m,
    }
    
    # Convert half-life to window with guardrails
    MIN_WINDOW, MAX_WINDOW, DEFAULT_WINDOW = 10, 240, 60
    if np.isnan(hl) or np.isinf(hl) or hl <= 0:
        w = DEFAULT_WINDOW
        print(f"  ⚠ Half-life invalid ({hl}) — using default window={DEFAULT_WINDOW}")
    else:
        w = int(min(max(round(hl), MIN_WINDOW), MAX_WINDOW))
    cr['window'] = w
    
    a, b = pairs[label]
    print(f"\n{label} ({a}/{b}):")
    print(f"  Hedge ratio: α={alpha:.4f}, β={beta:.4f}")
    print(f"  6-month: HL={hl:.1f} bars, H={H:.4f}")
    print(f"  3-month: HL={hl_3m:.1f} bars, H={H_3m:.4f}")
    print(f"  Window: {w} bars (clamped to [{MIN_WINDOW}, {MAX_WINDOW}])")

# Primary pair window for use in later steps
WINDOW = char_results['primary']['window']
print(f"\nPrimary pair window: {WINDOW} bars")
```

---

## Step 4: Quantile-Based Entry Thresholds — Per Pair [10 min]

Each pair gets its own adaptive threshold derived from its own formation-period Z-score distribution.

The threshold must be computed on `abs(z)` to produce a symmetric two-sided band. Using a one-sided `z.quantile(0.95)` would be inconsistent because the Z-score distribution may be skewed — the 95th percentile of raw Z is not the same as the 5th percentile negated.

```python
for label, d in data.items():
    cr = char_results[label]
    hl = cr['half_life_6m']
    w = cr['window']  # from Step 3 guardrails
    
    spread_f = compute_spread(
        d['formation'].iloc[:, 0], d['formation'].iloc[:, 1],
        cr['alpha'], cr['beta']
    )
    z_f = compute_zscore(spread_f, window=w)['zscore'].dropna()
    
    # Symmetric threshold from absolute Z distribution
    abs_q95 = z_f.abs().quantile(0.95)
    cr['adaptive_entry'] = round(abs_q95, 2)
    cr['formation_coverage_2σ'] = round((z_f.abs() <= 2.0).mean() * 100, 1)
    
    # Also report asymmetry for documentation
    cr['upper_95'] = round(z_f.quantile(0.975), 2)
    cr['lower_05'] = round(z_f.quantile(0.025), 2)
    
    a, b = pairs[label]
    print(f"{label} ({a}/{b}): |Z| 95th pct = {abs_q95:.2f}, "
          f"upper 97.5th = {cr['upper_95']:.2f}, lower 2.5th = {cr['lower_05']:.2f}, "
          f"±2.0 covers {cr['formation_coverage_2σ']}%")
```

---

## Step 5: Run Pipeline — All Pairs [20 min]

```python
pipeline_results = {}

for label, d in data.items():
    cr = char_results[label]
    w = cr['window']  # guardrailed window from Step 3
    
    df_fixed = run_pipeline(
        d['trading'].iloc[:, 0], d['trading'].iloc[:, 1],
        cr['alpha'], cr['beta'], window=w, entry_z=2.0
    )
    df_adaptive = run_pipeline(
        d['trading'].iloc[:, 0], d['trading'].iloc[:, 1],
        cr['alpha'], cr['beta'], window=w, entry_z=cr['adaptive_entry']
    )
    
    pipeline_results[label] = {
        'df_fixed': df_fixed,
        'df_adaptive': df_adaptive,
        'window_used': w,
    }
    
    a, b = pairs[label]
    n_trades_fixed = count_trades(df_fixed['position'])
    n_trades_adapt = count_trades(df_adaptive['position'])
    print(f"{label} ({a}/{b}): β={cr['beta']:.3f}, window={w}, "
          f"trades: fixed±2.0={n_trades_fixed}, "
          f"adaptive±{cr['adaptive_entry']}={n_trades_adapt}")
```

---

## Step 6: Distribution Diagnostics — Primary Pair [35 min]

### 6a. Tail Behavior (emphasized)

```python
z_trading = pipeline_results['primary']['df_fixed']['zscore'].dropna()

print(f"Skewness: {z_trading.skew():.3f}")
print(f"Excess kurtosis: {z_trading.kurtosis():.3f}")
```

### 6b. Empirical Coverage Table (the centerpiece)

```python
from scipy.stats import norm

print(f"{'Threshold':>10} | {'Normal':>8} | {'Empirical':>10} | {'Gap':>6}")
print("-" * 45)
for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
    theory = (1 - 2 * norm.sf(t)) * 100
    actual = (z_trading.abs() <= t).mean() * 100
    print(f"    ±{t}σ   |  {theory:.1f}%  |    {actual:.1f}%   | {theory-actual:+.1f}%")
```

### 6c. QQ-Plot

```python
from scipy.stats import probplot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
probplot(z_trading.values, dist="norm", plot=ax)
ax.set_title("QQ-Plot: Trading Period Z-Score vs Normal")
plt.tight_layout()
plt.savefig('qq_plot.png', dpi=150)
```

### 6d. Normality Tests (reported, not emphasized)

```python
from scipy.stats import shapiro, jarque_bera

_, p_shapiro = shapiro(z_trading.sample(5000, random_state=42))
_, p_jb = jarque_bera(z_trading.values)
# Report in document as supporting evidence, not as the main finding.
# On 50K+ minute bars, rejection is near-guaranteed regardless of actual distribution.
```

### 6e. Quantile Cross-Validation

Check whether the formation-derived threshold generalizes to the trading period.

```python
cr_primary = char_results['primary']

# Formation threshold (already computed in Step 4)
formation_adaptive = cr_primary['adaptive_entry']

# Trading period: same metric
abs_q95_trading = z_trading.abs().quantile(0.95)

print(f"Formation |Z| 95th: {formation_adaptive}")
print(f"Trading |Z| 95th:   {abs_q95_trading:.2f}")
print(f"Difference: {abs(formation_adaptive - abs_q95_trading):.2f}")

if abs(formation_adaptive - abs_q95_trading) > 0.5:
    print("NOTE: threshold shifted meaningfully between periods — regime change likely")
```

### 6f. LTCM Connection (~150 words in document)

Connect Mandelbrot's fat-tail critique from the LTCM guide to your empirical kurtosis and coverage table. Focus on the coverage gap — what ±2σ actually captures vs what theory predicts — not on hypothesis test p-values.

---

## Step 7: Cross-Pair Comparison [15 min]

```python
comparison = []
for label, d in data.items():
    df = pipeline_results[label]['df_fixed']
    z = df['zscore'].dropna()
    cr = char_results[label]
    
    comparison.append({
        'pair': f"{pairs[label][0]}/{pairs[label][1]}",
        'beta': f"{cr['beta']:.3f}",
        'half_life': f"{cr['half_life_6m']:.0f}",
        'hurst': f"{cr['hurst_6m']:.3f}",
        'trades': count_trades(df['position']),
        'kurtosis': f"{z.kurtosis():.1f}",
        'empirical_2σ': f"{(z.abs() <= 2.0).mean()*100:.1f}%",
    })

print(pd.DataFrame(comparison).to_string(index=False))
```

**What to report:** Present the table and let the data speak. If pairs with lower Hurst and shorter half-life happen to show fewer trade entries, note this as an observed pattern consistent with mean-reversion theory — but do not pre-commit to this as a guaranteed outcome. The cross-pair comparison is descriptive, not causal. Trade quality claims require evaluating outcomes (Week 3).

---

## Step 8: Threshold Sensitivity — Primary Pair [20 min]

Now using trade count (from state machine), not bar count.

```python
cr = char_results['primary']
adaptive = cr['adaptive_entry']
thresholds = [1.5, 2.0, adaptive, 2.5, 3.0]
rows = []
for t in thresholds:
    df_t = run_pipeline(
        data['primary']['trading'].iloc[:, 0],
        data['primary']['trading'].iloc[:, 1],
        cr['alpha'], cr['beta'], window=WINDOW, entry_z=t
    )
    n_trades = count_trades(df_t['position'])
    days = df_t.index.normalize().nunique()
    rows.append({
        'threshold': f'±{t:.2f}',
        'trades': n_trades,
        'trades_per_day': round(n_trades / days, 2),
    })

print(pd.DataFrame(rows).to_string(index=False))
```

---

## Step 9: Rolling Hurst — Regime Monitor [15 min]

```python
def rolling_hurst(spread, rolling_window=5000, step=1000, max_lag=100):
    results = []
    vals = spread.dropna()
    for start in range(0, len(vals) - rolling_window, step):
        end = start + rolling_window
        H = hurst_exponent(vals.iloc[start:end], max_lag=max_lag)
        results.append({'timestamp': vals.index[end-1], 'hurst': H})
    return pd.DataFrame(results).set_index('timestamp')

cr = char_results['primary']
spread_trading = compute_spread(
    data['primary']['trading'].iloc[:, 0],
    data['primary']['trading'].iloc[:, 1],
    cr['alpha'], cr['beta']
)
hurst_rolling = rolling_hurst(spread_trading)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hurst_rolling['hurst'])
ax.axhline(0.5, color='red', ls='--', label='H=0.5')
ax.set_title('Rolling Hurst — Primary Pair Trading Period')
ax.set_ylabel('H')
ax.legend()
plt.tight_layout()
plt.savefig('rolling_hurst.png', dpi=150)

pct_mr = (hurst_rolling['hurst'] < 0.5).mean() * 100
print(f"{pct_mr:.1f}% of windows show H < 0.5 (consistent with mean reversion)")
```

Report which periods show H > 0.5. This is a supporting diagnostic — if H stays above 0.5 for extended stretches, the Z-score model's mean-reversion assumption is weaker during those periods. Do not over-interpret short fluctuations around 0.5.

---

## Step 10: Visual Signal Validation [20 min]

```python
df_primary = pipeline_results['primary']['df_fixed']
week = df_primary['2022-09-12':'2022-09-16']
a, b = pairs['primary']

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(week['close_a'], label=a)
axes[0].plot(week['close_b'], label=b)
axes[0].set_title(f'{a} vs {b}')
axes[0].legend()

upper = week['rolling_mean'] + 2.0 * week['rolling_std']
lower = week['rolling_mean'] - 2.0 * week['rolling_std']
axes[1].plot(week['spread'], alpha=0.7, label='Spread')
axes[1].plot(week['rolling_mean'], color='black', label='Mean')
axes[1].fill_between(week.index, lower, upper, alpha=0.15, color='gray')
axes[1].set_title(f'Spread ±2σ (β={char_results["primary"]["beta"]:.3f}, window={WINDOW})')
axes[1].legend()

# Plot POSITIONS (stateful), not raw signals
in_long = week[week['position'] == 1]
in_short = week[week['position'] == -1]
axes[2].plot(week['zscore'], alpha=0.7)
axes[2].axhline(2.0, color='r', ls='--')
axes[2].axhline(-2.0, color='g', ls='--')
axes[2].axhline(0, color='black', alpha=0.3)
# Mark entry points only (position changes from 0)
entries_long = week[(week['position'] == 1) & (week['position'].shift(1) == 0)]
entries_short = week[(week['position'] == -1) & (week['position'].shift(1) == 0)]
exits = week[(week['position'] == 0) & (week['position'].shift(1) != 0)]
axes[2].scatter(entries_long.index, entries_long['zscore'],
                marker='^', c='green', s=40, zorder=5, label='Entry Long')
axes[2].scatter(entries_short.index, entries_short['zscore'],
                marker='v', c='red', s=40, zorder=5, label='Entry Short')
axes[2].scatter(exits.index, exits['zscore'],
                marker='x', c='black', s=30, zorder=5, label='Exit')
axes[2].set_title('Z-Score with Trade Entries/Exits')
axes[2].legend()

plt.tight_layout()
plt.savefig('signal_validation.png', dpi=150)
```

---

## Step 11: Assemble Signal Logic Document [45 min]

```
Signal Logic Document — The Rules of Engagement

1. PAIR SELECTION
   - Candidate pairs for engine validation (NOT claimed as trading pairs)
   - Primary + secondary from Week 1 near-misses (economic logic)
   - Optional ETF robustness check
   - Week 1 context: zero cointegrated pairs after BH-FDR;
     these are candidate pairs with varying mean-reversion evidence

2. DATA QUALITY
   - Alignment audit: expected vs actual bars, coverage %
   - Session boundaries: first/last bar times, gap analysis
   - Any session filtering applied (regular hours only, or full day)

3. SPREAD CONSTRUCTION
   - Hedge ratio β estimated by OLS on formation period
   - Spread = log(A) - α - β·log(B)
   - Per-pair β values reported

4. SPREAD CHARACTERIZATION (formation: Jan–Jun 2022)
   - Half-life → Z-score window (clamped to [10, 240])
   - Hurst exponent → supporting evidence for/against mean reversion
   - 3-month vs 6-month stability

5. Z-SCORE ENGINE
   - Rolling mean/std, ddof=1, ε=1e-10 guard
   - Window derived from half-life with floor/ceiling guardrails
   - Vectorized Pandas

6. POSITION LOGIC & EXECUTION CONVENTION
   - Stateful entry/exit: one trade per excursion
   - Trade count (not bar count) used in all analysis
   - Execution convention: position[t] from bar[t]'s close,
     executed at bar[t+1]'s open. Shift applied in Week 3.

7. DISTRIBUTION ANALYSIS (primary pair)
   - Empirical coverage table (centerpiece)
   - QQ-plot, kurtosis, skewness
   - Normality tests (supporting — rejection near-guaranteed on 50K+ bars)
   - Quantile cross-validation: formation |Z| 95th vs trading |Z| 95th
   - LTCM connection

8. CROSS-PAIR COMPARISON
   - Table: β, half-life, Hurst, trade count, kurtosis, coverage
   - Descriptive observations (no causal claims about signal quality)

9. ENTRY/EXIT RULES & JUSTIFICATION
   - Fixed: ±2.0σ
   - Adaptive: pair-specific, abs(Z) 95th percentile from formation
   - Exit: Z crosses 0.0
   - Threshold sensitivity (trade count)

10. REGIME MONITORING
    - Rolling Hurst (supporting diagnostic, noisy on minute data)

11. SIGNAL VALIDATION
    - 3-panel chart with entry/exit markers

12. PARAMETERS
    | Parameter        | Value           | Source                          |
    |------------------|-----------------|---------------------------------|
    | Hedge ratio β    | {from OLS}      | Formation-period OLS            |
    | Window           | {half-life}     | AR(1), clamped to [10, 240]     |
    | Entry (fixed)    | ±2.0σ           | Standard + empirical comparison |
    | Entry (adaptive) | pair-specific   | abs(Z) 95th pct from formation  |
    | Exit             | 0.0             | Mean reversion target           |
    | Epsilon          | 1e-10           | Division-by-zero guard          |
    | ddof             | 1               | Sample std convention           |

13. KNOWN LIMITATIONS & WEEK 3 HANDOFF
    - No transaction costs → Week 3
    - No stop-loss → Week 3
    - No overnight gap handling → Week 3
    - Hedge ratio static (formation-only) → rolling/Kalman
    - Pairs not formally cointegrated → near-miss, documented
```

---

## Implementation Sequence

| # | Task | Time |
|---|------|------|
| 0 | Pair selection | 15 min |
| 1 | Load + align + split all pairs | 30 min |
| 2 | Define all functions (OLS, spread, Z-score, state machine) | 20 min |
| 3 | Spread characterization (all pairs) | 30 min |
| 4 | Quantile thresholds (primary) | 10 min |
| 5 | Run pipeline all pairs | 20 min |
| 6 | Distribution diagnostics (primary) | 35 min |
| 7 | Cross-pair comparison table | 15 min |
| 8 | Threshold sensitivity (primary, trade count) | 20 min |
| 9 | Rolling Hurst (primary) | 15 min |
| 10 | Visualization (primary, entry/exit markers) | 20 min |
| 11 | Write document | 45 min |
| **Total** | | **~5.5 hours** |

---

## Sanity Checks

**Data quality:**
- [ ] Alignment audit printed: expected bars, actual bars, coverage %
- [ ] Session boundaries checked: typical first/last bar times make sense
- [ ] Decision on session filtering stated (regular hours only, or full day with rationale)

**Pair selection:**
- [ ] Pairs explicitly framed as "candidate pairs for engine validation"
- [ ] Economic rationale stated for each pair
- [ ] ETF pair (if used) labeled as robustness check, not primary

**Formation period (Jan–Jun):**
- [ ] Hedge ratio β estimated on formation only
- [ ] Half-life converted to window with guardrails: clamped to [10, 240], NaN/Inf handled
- [ ] Hurst reported as supporting diagnostic, not confirmation
- [ ] 3m/6m stability reported
- [ ] Adaptive threshold computed from abs(Z).quantile(0.95) — symmetric, pair-specific

**Trading period (Jul–Dec):**
- [ ] Pipeline uses per-pair window from `cr['window']`, not recalculated
- [ ] Pipeline uses per-pair adaptive threshold from `cr['adaptive_entry']`
- [ ] Formation |Z| 95th vs trading |Z| 95th reported (cross-validation)
- [ ] Threshold sensitivity uses trade count from state machine, not bar count
- [ ] `count_trades()` gives plausible numbers
- [ ] First window//2 Z-scores are NaN
- [ ] Position values ∈ {-1, 0, +1}

**Diagnostics:**
- [ ] Empirical coverage at ±2σ reported and compared to 95.4% (report honestly — may or may not be lower)
- [ ] Cross-pair table: observations described as patterns, not causal claims
- [ ] Rolling Hurst chart with caveats about estimator noise on minute data
- [ ] Normality p-values as supporting evidence, not main finding

**Document:**
- [ ] Execution convention stated: position[t] executes at bar[t+1]
- [ ] Every parameter traces to a computation
- [ ] 3-panel chart shows entry/exit transitions, not every bar in zone
- [ ] Data quality section included

---

## Amendment — Dynamic Hedge Ratio via Kalman Filter

> **Status:** Approved. Added after Chunks 1–5 were complete.  
> **Reason:** Static OLS beta frozen at formation_end is overly conservative — it is not a lookahead guard, it is deliberate ignorance of causal information available in the trading period. A Kalman filter corrects this by updating β(t) at every bar using only data `1..t`.

### What changes

| Item | Change |
|------|--------|
| `src/signals/kalman.py` | **New.** `kalman_hedge_ratio(close_a, close_b, delta)` — returns `(alpha_series, beta_series, diagnostics)` |
| `src/signals/spread.py` | **Minor.** `compute_spread()` type hint updated to `alpha: float \| pd.Series`, `.values` strip added before numpy arithmetic |
| `src/pipeline/run_week2.py` | **Minor.** `run_pipeline()` signature updated; `main()` gets a Kalman branch controlled by config flag |
| `configs/params_example.yaml` | **Two keys added** under `engine`: `use_kalman: false`, `kalman_delta: 1.0e-5` |
| `tests/test_kalman.py` | **New.** 6 unit tests: output shapes, constant-beta convergence, beta drift tracking, no-lookahead, no NaN, positive beta |
| `validate_kalman.py` | **New.** Runs Kalman on CMS/DUK full year; prints static vs Kalman comparison table; saves beta-drift plot and Kalman signal CSV |
| `src/visuals/plots.py` | **Addition.** `plot_kalman_beta()` — β(t) over full year with formation/trading boundary line |

### State-space model

```
State:       θ_t = [α_t, β_t]
Transition:  θ_t = θ_{t-1} + w_t,   w_t ~ N(0, Q)        ← random walk in beta
Observation: log(A_t) = [1, log(B_t)] · θ_t + v_t,  v_t ~ N(0, R)

Q = (delta / (1 - delta)) × I₂          default delta = 1e-5
R = var(OLS residuals on first 500 bars)  estimated once from data, never from trading period
θ₀ = [0.0, 1.0]   (neutral prior: zero intercept, unit beta)
P₀ = Q / delta    (uninformative)
```

### Key constraint: no-lookahead preserved

The filter runs on the **full year** (Jan–Dec) from bar 1 to bar n. At every bar t, β(t) uses only data `1..t`. The formation period (Jan–Jun) warms up the filter; by July the filter has a well-informed prior. The trading-period beta is **sliced** from the full-year output — never re-initialized from trading data. This is strictly causal.

### Deliverable from this amendment

`validate_kalman.py` prints a side-by-side comparison table:

| Metric | Static OLS | Kalman (δ=1e-5) |
|--------|-----------|-----------------|
| Formation H | | |
| Trading H (% windows < 0.5) | 100% | |
| Half-life (bars) | 679.7 | |
| Z-score std | 1.3928 | |
| Coverage gap ±2σ | +9.40% | |
| Trades (fixed 2.0) | 265 | |
| Regime shift | No | |

If Kalman produces a shorter half-life, tighter std, and smaller coverage gap — that is the quantified case for upgrading the default. Results go into the Signal Logic Document.

### Why 2.0 is still the right entry threshold (updated reasoning)

Under static OLS, the Z-score std = 1.3928 because the 240-bar window is only 35% of the 680-bar half-life. The true 2σ event in that distribution sits at z ≈ 2.79. The fixed 2.0 threshold is therefore firing at roughly the 1.44σ level of the underlying distribution — more aggressive than intended.

Under Kalman, the spread is better centered at each bar (beta tracks the true relationship more closely), so the spread variance should be lower, the std closer to 1.0, and the 2.0 threshold closer to a true 2σ event.

**The argument for 2.0 over 1.5:**
- At 1.5 (static OLS), you are entering at roughly 1.08σ of the true distribution — pure noise
- At 2.0 (static OLS), you are entering at roughly 1.44σ — marginal
- At 2.0 (Kalman), the std should be closer to 1.0, making 2.0 a genuine 2σ trigger
- At 1.5 (any model), transaction costs eat the edge before mean-reversion materialises
- The sensitivity table confirms: 382 trades at 1.5 vs 265 at 2.0 — 44% more trades with no evidence of better signal quality

### Known limitation retained from original plan

Static OLS path remains intact. `use_kalman: false` in config reproduces all prior results unchanged. The Kalman path is additive, not a replacement.
