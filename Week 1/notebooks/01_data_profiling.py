# %% [markdown]
# # Notebook 01 — Data Profiling & Preparation (FULL RUN)
#
# **Purpose:** Load, clean, and prepare 1-minute OHLC data for later cointegration
# testing in Notebook 02.
#
# **Scope:** Full run — discover all tickers, screen by quality, use ALL that pass.
#
# **Methodology source of truth:**
# - `.agents/workflows/cointegration_methodology_spec.md`
# - `.agents/workflows/implementation_checklist.md`
#
# **Key decisions (approved, not revisited here):**
# - Use `close` price
# - Use log prices
# - Filter to 9:35–15:55 ET (exclude auction periods)
# - Resample to 5-minute bars
# - Outlier threshold: |z| > 10σ on minute returns
# - Universe: tickers present in all 12 months → quality screen → all that pass (no cap)

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
import sys, io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore', category=FutureWarning)

# -- Paths --
PROJECT_ROOT = Path(r"d:\Quant Finance\Quant Program\Week 1")
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

MONTH_FOLDERS = sorted([f"{i:02d}" for i in range(1, 13)])

# -- Approved methodology parameters --
SESSION_START = pd.Timestamp('09:35:00').time()
SESSION_END = pd.Timestamp('15:55:00').time()
RESAMPLE_FREQ = '5min'
OUTLIER_ZSCORE_THRESHOLD = 10
OUTLIER_PCT_REMOVAL_THRESHOLD = 1.0
TOTAL_OUTLIER_BUDGET_PCT = 0.5

# -- Screening thresholds --
SCREEN_MIN_MEDIAN_PRICE = 5.0
SCREEN_MIN_AVG_DAILY_DOLV = 1_000_000
SCREEN_MIN_COMPLETENESS = 90.0
SCREEN_MAX_ZERO_RETURN_PCT = 50.0
UNIVERSE_CAP = None  # No cap — use all tickers that pass screening

print("FULL RUN -- Notebook 01: Data Profiling & Preparation")
print("=" * 60)
print(f"Session window: {SESSION_START} - {SESSION_END} ET")
print(f"Resample: {RESAMPLE_FREQ}")
print(f"Universe cap: None (all tickers that pass screening)")

# %% [markdown]
# ## 2. Universe Discovery
#
# Find all unique tickers across 12 months.
# Primary rule: only keep tickers present in ALL 12 months.

# %%
def discover_tickers(data_dir: Path, month_folders: list[str]) -> dict:
    """Discover all tickers per month from file names.

    Returns dict: {month_str: set of ticker names}
    """
    month_tickers = {}
    for month in month_folders:
        month_dir = data_dir / month
        if not month_dir.exists():
            print(f"  WARNING: {month_dir} does not exist")
            month_tickers[month] = set()
            continue
        files = list(month_dir.glob("*.csv"))
        tickers = set()
        for f in files:
            # Filename pattern: TICKER_YYYY-MM-DD.csv
            name = f.stem  # e.g. "AAPL_2022-01-03"
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                tickers.add(parts[0])
        month_tickers[month] = tickers
    return month_tickers


print("Discovering tickers across 12 months...")
month_tickers = discover_tickers(DATA_DIR, MONTH_FOLDERS)

# All unique tickers across any month
all_tickers = set()
for s in month_tickers.values():
    all_tickers |= s

# Tickers present in ALL 12 months
full_year_tickers = set.intersection(*month_tickers.values()) if month_tickers else set()
excluded_by_12mo = all_tickers - full_year_tickers

print(f"\n-- Universe Completeness Report (Table 1 Part A) --")
print(f"Total unique tickers across all months: {len(all_tickers)}")
print(f"Tickers present in all 12 months:       {len(full_year_tickers)}")
print(f"Tickers excluded by 12-month rule:      {len(excluded_by_12mo)}")
if len(excluded_by_12mo) <= 50:
    print(f"Excluded tickers: {sorted(excluded_by_12mo)}")
else:
    print(f"Excluded tickers (first 50): {sorted(excluded_by_12mo)[:50]}")

# Use full-year tickers as the candidate universe
candidate_tickers = sorted(full_year_tickers)
print(f"\nCandidate universe: {len(candidate_tickers)} tickers entering screening")

# %% [markdown]
# ## 3. Data Loading
#
# Load minute-bar CSVs for ALL candidate tickers across all 12 months.
# Process one ticker at a time to manage memory.

# %%
def load_ticker_data(ticker: str, data_dir: Path, month_folders: list[str]) -> pd.DataFrame:
    """Load all daily CSV files for a single ticker across all months.

    Returns a DataFrame with columns [close, volume] indexed by ET datetime.
    Returns empty DataFrame if no files found.
    """
    all_frames = []

    for month in month_folders:
        month_dir = data_dir / month
        pattern = str(month_dir / f"{ticker}_*.csv")
        files = glob.glob(pattern)
        for fpath in files:
            try:
                df = pd.read_csv(fpath, usecols=['close', 'volume', 'window_start'])
                all_frames.append(df)
            except Exception:
                pass  # Silently skip bad files; screening will catch data gaps

    if not all_frames:
        return pd.DataFrame(columns=['close', 'volume'])

    combined = pd.concat(all_frames, ignore_index=True)

    # Convert nanosecond UTC timestamps to ET DatetimeIndex
    combined['datetime_et'] = (
        pd.to_datetime(combined['window_start'], unit='ns', utc=True)
        .dt.tz_convert('US/Eastern')
    )
    combined = combined.set_index('datetime_et').sort_index()

    # Drop duplicate timestamps (keep last per checklist)
    n_dupes = combined.index.duplicated(keep='last').sum()
    if n_dupes > 0:
        combined = combined[~combined.index.duplicated(keep='last')]

    combined = combined[['close', 'volume']]
    return combined


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to approved trading session 9:35-15:55 ET."""
    times = df.index.time
    mask = (times >= SESSION_START) & (times <= SESSION_END)
    return df.loc[mask]


# -- Load and filter all candidate tickers --
print(f"Loading {len(candidate_tickers)} tickers (this may take a few minutes)...")
filtered_data = {}
load_errors = []

for i, ticker in enumerate(candidate_tickers):
    if (i + 1) % 50 == 0 or (i + 1) == len(candidate_tickers):
        print(f"  Progress: {i+1}/{len(candidate_tickers)}")
    try:
        raw_df = load_ticker_data(ticker, DATA_DIR, MONTH_FOLDERS)
        if len(raw_df) == 0:
            load_errors.append((ticker, "no data"))
            continue
        filt_df = filter_session(raw_df)
        if len(filt_df) == 0:
            load_errors.append((ticker, "empty after session filter"))
            continue
        filtered_data[ticker] = filt_df
    except Exception as e:
        load_errors.append((ticker, str(e)))

print(f"\nLoaded successfully: {len(filtered_data)} tickers")
if load_errors:
    print(f"Load errors: {len(load_errors)}")
    for t, reason in load_errors[:10]:
        print(f"  {t}: {reason}")

# %% [markdown]
# ## 4. Per-Ticker Profiling and Universe Screening
#
# Compute quality metrics for each ticker, then apply hard screening filters.
# If N > 50 after screening, cap at top 50 by avg daily dollar volume.

# %%
def compute_ticker_profile(ticker: str, df: pd.DataFrame) -> dict:
    """Compute data quality profile for a single ticker."""
    if len(df) == 0:
        return {'ticker': ticker, 'n_filtered_minutes': 0, 'status': 'EMPTY'}

    trading_dates = df.index.normalize().unique()
    n_trading_days = len(trading_dates)
    first_date = trading_dates.min().date()
    last_date = trading_dates.max().date()

    # Completeness: expected 381 minutes per full trading day
    expected_minutes = n_trading_days * 381
    actual_minutes = len(df)
    completeness_pct = (actual_minutes / expected_minutes * 100) if expected_minutes > 0 else 0

    median_close = df['close'].median()
    min_close = df['close'].min()
    max_close = df['close'].max()

    # Dollar volume
    df_copy = df.copy()
    df_copy['dollar_volume'] = df_copy['volume'] * df_copy['close']
    daily_dollar_vol = df_copy.groupby(df_copy.index.date)['dollar_volume'].sum()
    avg_daily_dollar_volume = daily_dollar_vol.mean()

    # Zero-return fraction
    returns = df['close'].pct_change()
    day_starts = df.index.to_series().diff() > pd.Timedelta(minutes=5)
    returns[day_starts] = np.nan
    valid_returns = returns.dropna()
    zero_return_pct = ((valid_returns == 0).sum() / len(valid_returns) * 100) if len(valid_returns) > 0 else 0

    return {
        'ticker': ticker,
        'n_filtered_minutes': actual_minutes,
        'n_trading_days': n_trading_days,
        'first_date': str(first_date),
        'last_date': str(last_date),
        'completeness_pct': round(completeness_pct, 1),
        'median_close': round(median_close, 2),
        'min_close': round(min_close, 2),
        'max_close': round(max_close, 2),
        'avg_daily_dollar_volume': round(avg_daily_dollar_volume, 0),
        'zero_return_pct': round(zero_return_pct, 1),
    }


# -- Profile all loaded tickers --
print(f"Profiling {len(filtered_data)} tickers...")
profiles = []
for i, ticker in enumerate(sorted(filtered_data.keys())):
    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{len(filtered_data)}")
    profile = compute_ticker_profile(ticker, filtered_data[ticker])
    profiles.append(profile)

profile_df = pd.DataFrame(profiles)
print(f"Profiling complete: {len(profile_df)} tickers")

# %% [markdown]
# ### 4a. Apply screening filters

# %%
screening_results = []
for _, row in profile_df.iterrows():
    reason = None
    if row.get('n_filtered_minutes', 0) == 0:
        reason = "no data"
    elif row['median_close'] < SCREEN_MIN_MEDIAN_PRICE:
        reason = f"median_close={row['median_close']} < {SCREEN_MIN_MEDIAN_PRICE}"
    elif row['avg_daily_dollar_volume'] < SCREEN_MIN_AVG_DAILY_DOLV:
        reason = f"avg_daily_dolv={row['avg_daily_dollar_volume']:,.0f} < {SCREEN_MIN_AVG_DAILY_DOLV:,.0f}"
    elif row['completeness_pct'] < SCREEN_MIN_COMPLETENESS:
        reason = f"completeness={row['completeness_pct']}% < {SCREEN_MIN_COMPLETENESS}%"
    elif row['zero_return_pct'] >= SCREEN_MAX_ZERO_RETURN_PCT:
        reason = f"zero_return={row['zero_return_pct']}% >= {SCREEN_MAX_ZERO_RETURN_PCT}%"

    screening_results.append({
        'ticker': row['ticker'],
        'passed_screening': reason is None,
        'rejection_reason': reason if reason else '',
        'avg_daily_dollar_volume': row.get('avg_daily_dollar_volume', 0),
    })

screening_df = pd.DataFrame(screening_results)
passed_df = screening_df[screening_df['passed_screening']].copy()
failed_df = screening_df[~screening_df['passed_screening']].copy()

n_passed = len(passed_df)
n_failed = len(failed_df)

print(f"\nScreening results: {n_passed} passed, {n_failed} failed")
print(f"\nRejection reasons breakdown:")
if n_failed > 0:
    reason_counts = failed_df['rejection_reason'].str.split('=').str[0].value_counts()
    for reason, cnt in reason_counts.items():
        print(f"  {reason}: {cnt}")

# -- No universe cap: use all tickers that passed screening --
surviving_tickers = sorted(passed_df['ticker'].tolist())

print(f"\nFinal screened universe: {len(surviving_tickers)} tickers")
print(f"Tickers: {surviving_tickers}")

# %% [markdown]
# ## 5. Cleaning: Outlier Treatment
#
# **Approved rule:** Flag minute returns with |z| > 10 sigma. Replace the corresponding
# close price with NaN, then forward-fill (max 1 bar).
# If any ticker exceeds 1% outlier rate, remove it entirely.

# %%
outlier_log = []
cleaned_data = {}
total_points = 0
total_outliers = 0

for ticker in surviving_tickers:
    df = filtered_data[ticker][['close', 'volume']].copy()
    n_total = len(df)
    total_points += n_total

    returns = df['close'].pct_change()
    day_starts = df.index.to_series().diff() > pd.Timedelta(minutes=5)
    returns[day_starts] = np.nan

    valid_returns = returns.dropna()
    if len(valid_returns) < 100:
        outlier_log.append({
            'ticker': ticker, 'n_total': n_total,
            'n_outliers_flagged': 0, 'pct_outliers': 0.0,
            'action': 'kept (too few returns)',
        })
        cleaned_data[ticker] = df
        continue

    ret_mean = valid_returns.mean()
    ret_std = valid_returns.std()

    if ret_std == 0:
        outlier_log.append({
            'ticker': ticker, 'n_total': n_total,
            'n_outliers_flagged': 0, 'pct_outliers': 0.0,
            'action': 'kept (zero std)',
        })
        cleaned_data[ticker] = df
        continue

    z_scores = (returns - ret_mean) / ret_std
    outlier_mask = z_scores.abs() > OUTLIER_ZSCORE_THRESHOLD
    n_outliers = outlier_mask.sum()
    pct_outliers = n_outliers / n_total * 100
    total_outliers += n_outliers

    if pct_outliers > OUTLIER_PCT_REMOVAL_THRESHOLD:
        outlier_log.append({
            'ticker': ticker, 'n_total': n_total,
            'n_outliers_flagged': int(n_outliers), 'pct_outliers': round(pct_outliers, 3),
            'action': f'REMOVED (>{OUTLIER_PCT_REMOVAL_THRESHOLD}% outliers)',
        })
        continue

    if n_outliers > 0:
        df.loc[outlier_mask, 'close'] = np.nan
        df['close'] = df['close'].ffill(limit=1)

    outlier_log.append({
        'ticker': ticker, 'n_total': n_total,
        'n_outliers_flagged': int(n_outliers), 'pct_outliers': round(pct_outliers, 4),
        'action': f'kept ({n_outliers} patched)' if n_outliers > 0 else 'kept (clean)',
    })
    cleaned_data[ticker] = df

outlier_df = pd.DataFrame(outlier_log)
print("Outlier Treatment Summary:")
print(f"  Tickers processed: {len(outlier_log)}")
print(f"  Tickers kept: {len(cleaned_data)}")
removed_by_outliers = set(surviving_tickers) - set(cleaned_data.keys())
if removed_by_outliers:
    print(f"  Tickers REMOVED by outlier rule: {sorted(removed_by_outliers)}")

total_outlier_pct = (total_outliers / total_points * 100) if total_points > 0 else 0
print(f"  Total outliers: {total_outliers:,} / {total_points:,} ({total_outlier_pct:.4f}%)")
if total_outlier_pct > TOTAL_OUTLIER_BUDGET_PCT:
    print(f"  WARNING: exceeds {TOTAL_OUTLIER_BUDGET_PCT}% budget")
else:
    print(f"  Within budget (<{TOTAL_OUTLIER_BUDGET_PCT}%)")

# Show top 10 most-modified tickers
top_modified = outlier_df[outlier_df['n_outliers_flagged'] > 0].nlargest(10, 'n_outliers_flagged')
if len(top_modified) > 0:
    print("\nTop 10 most-modified tickers:")
    print(top_modified[['ticker', 'n_outliers_flagged', 'pct_outliers', 'action']].to_string(index=False))

surviving_tickers = sorted(cleaned_data.keys())
print(f"\nFinal universe after outlier treatment: {len(surviving_tickers)} tickers")

# Free memory: drop filtered_data for tickers not in cleaned_data
filtered_data = {k: v for k, v in filtered_data.items() if k in cleaned_data}

# %% [markdown]
# ## 6. Log Transform and Resampling

# %%
# -- Assert positive prices --
for ticker in surviving_tickers:
    close_series = cleaned_data[ticker]['close'].dropna()
    assert (close_series > 0).all(), f"{ticker} has non-positive prices"
print("Price positivity check PASSED for all tickers.")

# -- Log transform and resample --
resampled = {}
resample_report = []

for ticker in surviving_tickers:
    df = cleaned_data[ticker].copy()
    n_before = len(df)

    df['log_close'] = np.log(df['close'])
    log_5min = df['log_close'].resample(RESAMPLE_FREQ).last()
    log_5min = log_5min.dropna()

    n_after = len(log_5min)
    resampled[ticker] = log_5min

    resample_report.append({
        'ticker': ticker,
        'n_1min_bars': n_before,
        'n_5min_bars': n_after,
        'ratio': round(n_before / n_after, 1) if n_after > 0 else 0,
    })

resample_report_df = pd.DataFrame(resample_report)
print(f"Resample complete: {len(resampled)} tickers")
print(f"  Avg ratio: {resample_report_df['ratio'].mean():.1f} (expected ~5.0)")
print(f"  Avg 5-min bars per ticker: {resample_report_df['n_5min_bars'].mean():,.0f}")

# Free memory
del cleaned_data

# %% [markdown]
# ### 6a. Align tickers on a common timestamp grid

# %%
log_price_panel = pd.DataFrame(resampled)

n_rows_before = len(log_price_panel)
nan_before = log_price_panel.isna().sum()
n_tickers_with_nan = (nan_before > 0).sum()
print(f"Before alignment: {n_rows_before} rows, {n_tickers_with_nan} tickers have NaN")

# Inner join: keep only rows where ALL tickers have data
log_price_panel = log_price_panel.dropna()
n_rows_after = len(log_price_panel)
n_dropped = n_rows_before - n_rows_after
pct_loss = n_dropped / n_rows_before * 100 if n_rows_before > 0 else 0

print(f"After alignment:  {n_rows_after} rows ({n_dropped} dropped, {pct_loss:.1f}% loss)")
print(f"Final panel shape: {log_price_panel.shape}")
print(f"Date range: {log_price_panel.index[0]} to {log_price_panel.index[-1]}")

# Free memory
del resampled

# %% [markdown]
# ## 7. Output Artifacts

# %%
# -- Save main data artifact --
output_path = INTERMEDIATE_DIR / "log_prices_5min.parquet"
log_price_panel.to_parquet(output_path, engine='pyarrow')
print(f"Saved: {output_path}")
print(f"  Shape: {log_price_panel.shape}")
print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

# -- Build Data Audit Summary (Table 1 Part B) --
audit_records = []
for ticker in surviving_tickers:
    prof_row = profile_df[profile_df['ticker'] == ticker]
    screen_row = screening_df[screening_df['ticker'] == ticker]
    out_row = outlier_df[outlier_df['ticker'] == ticker]
    resamp_row = resample_report_df[resample_report_df['ticker'] == ticker]

    audit_records.append({
        'ticker': ticker,
        'n_raw_minutes': int(prof_row.iloc[0]['n_filtered_minutes']) if len(prof_row) > 0 else 0,
        'n_filtered_minutes': int(prof_row.iloc[0]['n_filtered_minutes']) if len(prof_row) > 0 else 0,
        'n_5min_bars': int(resamp_row.iloc[0]['n_5min_bars']) if len(resamp_row) > 0 else 0,
        'median_close': float(prof_row.iloc[0]['median_close']) if len(prof_row) > 0 else np.nan,
        'avg_daily_dollar_volume': float(prof_row.iloc[0]['avg_daily_dollar_volume']) if len(prof_row) > 0 else np.nan,
        'completeness_pct': float(prof_row.iloc[0]['completeness_pct']) if len(prof_row) > 0 else np.nan,
        'zero_return_pct': float(prof_row.iloc[0]['zero_return_pct']) if len(prof_row) > 0 else np.nan,
        'n_outliers_flagged': int(out_row.iloc[0]['n_outliers_flagged']) if len(out_row) > 0 else 0,
        'pct_outliers': float(out_row.iloc[0]['pct_outliers']) if len(out_row) > 0 else 0.0,
        'passed_screening': True,
        'rejection_reason': '',
    })

audit_summary = pd.DataFrame(audit_records)
audit_path = INTERMEDIATE_DIR / "universe_metadata.parquet"
audit_summary.to_parquet(audit_path, engine='pyarrow')
print(f"Saved: {audit_path}")

# Save universe completeness report
completeness_report = pd.DataFrame([{
    'total_unique_tickers_all_months': len(all_tickers),
    'tickers_in_all_12_months': len(full_year_tickers),
    'tickers_excluded_by_12mo_rule': len(excluded_by_12mo),
    'tickers_after_quality_screen': n_passed,
    'tickers_after_universe_cap': len(surviving_tickers),
    'tickers_after_outlier_treatment': len(surviving_tickers),
    'final_panel_tickers': log_price_panel.shape[1],
    'final_panel_rows': log_price_panel.shape[0],
}])
completeness_path = INTERMEDIATE_DIR / "universe_completeness.parquet"
completeness_report.to_parquet(completeness_path, engine='pyarrow')
print(f"Saved: {completeness_path}")

# %% [markdown]
# ## 8. Validation

# %%
print("-- Validation --")

# Reload test
reloaded = pd.read_parquet(output_path, engine='pyarrow')
assert reloaded.shape == log_price_panel.shape, \
    f"Shape mismatch: {reloaded.shape} vs {log_price_panel.shape}"
print(f"1. Parquet reload: shape {reloaded.shape} OK")

# Timezone
if reloaded.index.tz is None:
    print("   NOTE: Parquet stripped timezone. Notebook 02 must re-localize.")
else:
    print(f"   Index tz: {reloaded.index.tz}")

# Duplicates
n_dupes = reloaded.index.duplicated().sum()
assert n_dupes == 0, f"Found {n_dupes} duplicate timestamps"
print(f"2. Duplicate timestamps: {n_dupes} OK")

# NaN
total_nan = reloaded.isna().sum().sum()
assert total_nan == 0, f"Found {total_nan} NaN values"
print(f"3. NaN: {total_nan} OK")

# Dtypes
assert all(reloaded.dtypes == np.float64), "Not all columns float64"
print(f"4. All {len(reloaded.columns)} columns are float64 OK")

# Monotonic
assert reloaded.index.is_monotonic_increasing, "Index not sorted"
print(f"5. Index is monotonic increasing OK")

# Log price sanity
for col in reloaded.columns:
    prices = np.exp(reloaded[col])
    assert prices.min() > 1, f"{col} min price suspiciously low"
    assert prices.max() < 10000, f"{col} max price suspiciously high"
print(f"6. Log price sanity (exp -> reasonable stock prices) OK")

# %% [markdown]
# ## Full Run Status Summary

# %%
print("=" * 60)
print("NOTEBOOK 01 FULL RUN -- STATUS SUMMARY")
print("=" * 60)
print()
print(f"Universe discovery:")
print(f"  Total unique tickers:     {len(all_tickers)}")
print(f"  In all 12 months:         {len(full_year_tickers)}")
print(f"  After quality screening:  {n_passed}")
print(f"  After screening (no cap):  {len(surviving_tickers)}")
print()
print(f"Data pipeline:")
print(f"  Session filter:           9:35-15:55 ET")
print(f"  Resample:                 5-minute bars")
print(f"  Price transform:          log(close)")
print(f"  Outlier treatment:        |z| > 10 sigma, ffill(1)")
print()
print(f"Final panel:")
print(f"  Shape:    {log_price_panel.shape[0]} rows x {log_price_panel.shape[1]} tickers")
print(f"  Range:    {log_price_panel.index[0].date()} to {log_price_panel.index[-1].date()}")
print(f"  NaN:      0")
print(f"  Dupes:    0")
print()
print(f"Output files:")
print(f"  {output_path}")
print(f"  {audit_path}")
print(f"  {completeness_path}")
print()
print(f"Tickers in panel: {sorted(log_price_panel.columns.tolist())}")
print()
n_pairs = log_price_panel.shape[1] * (log_price_panel.shape[1] - 1) // 2
print(f"Expected pairs for Notebook 02: C({log_price_panel.shape[1]},2) = {n_pairs}")
print()
print("FULL RUN COMPLETE. Ready for Notebook 02.")
