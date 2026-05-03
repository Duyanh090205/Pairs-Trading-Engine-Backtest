import pandas as pd
import numpy as np

def compute_rolling_instability(df: pd.DataFrame, seasonality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling spread instability (390-bar rolling standard deviation)
    on the seasonality-adjusted spread.
    
    Args:
        df: DataFrame with 'ticker', 'timestamp_et', 'full_spread_l1_bps', 'is_valid'
            This includes the trading window data.
        seasonality_df: Output from compute_intraday_seasonality (computed ONLY on formation window)
        
    Returns:
        DataFrame with rolling stats added.
    """
    df = df.copy()
    
    # 1. Assign buckets to the data (same logic as in seasonality.py)
    bins = [pd.to_datetime(f"09:30:00").time()]
    for h in range(10, 16):
        bins.append(pd.to_datetime(f"{h}:00:00").time())
        bins.append(pd.to_datetime(f"{h}:30:00").time())
    bins.append(pd.to_datetime(f"16:00:00").time())
    
    labels = []
    for i in range(len(bins)-1):
        t1_str = bins[i].strftime("%H:%M")
        t2_str = bins[i+1].strftime("%H:%M")
        if t2_str == "16:00":
            t2_str = "15:59"
        labels.append(f"{t1_str}-{t2_str}")
        
    common_date = pd.Timestamp('2000-01-01')
    seconds = (
        df['timestamp_et'].dt.hour * 3600
        + df['timestamp_et'].dt.minute * 60
        + df['timestamp_et'].dt.second
    )
    temp_dt = common_date + pd.to_timedelta(seconds, unit='s')
    dt_bins = [
        common_date + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        for t in bins
    ]

    df['bucket'] = pd.cut(temp_dt, bins=dt_bins, labels=labels, right=False, include_lowest=True)
    
    # 2. Merge seasonality medians back to the main dataframe
    # We only need the median for de-meaning
    medians = seasonality_df[['ticker', 'bucket', 'median']].copy()
    df = df.merge(medians, on=['ticker', 'bucket'], how='left')
    
    # For valid rows, calculate adjusted spread
    df['adj_spread_bps'] = np.nan
    mask = df['is_valid'] & df['median'].notna()
    df.loc[mask, 'adj_spread_bps'] = df.loc[mask, 'full_spread_l1_bps'] - df.loc[mask, 'median']
    
    # 3. Calculate rolling stats per ticker
    # Ensure data is sorted by timestamp within each ticker
    df = df.sort_values(['ticker', 'timestamp_et'])
    
    # 390 bars = 1 trading day
    rolling_window = 390
    
    # 390-bar burn-in: min_periods=390 ensures NaN until a full trading day of data exists
    df['spread_std_1d'] = df.groupby('ticker')['adj_spread_bps'].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=390).std()
    )

    # Pre-mask invalid bars before computing rolling mean (avoids fragile closure over df)
    df['_spread_valid'] = df['full_spread_l1_bps'].where(df['is_valid'])
    df['raw_spread_mean_1d'] = df.groupby('ticker')['_spread_valid'].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=390).mean()
    )
    
    # Clean up temporary columns
    df = df.drop(columns=['bucket', 'median', 'adj_spread_bps', '_spread_valid'])
    
    return df
