import pandas as pd

def compute_intraday_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per ticker, per 30-min bucket spread aggregates (mean, median, p95).
    
    No-Lookahead Rule: Seasonality medians must be estimated ONLY on the 
    formation window of each fold. This function assumes the input `df` 
    is ALREADY filtered to only contain formation window data.
    
    Args:
        df: DataFrame containing 'ticker', 'timestamp_et', 'full_spread_l1_bps', 'is_valid'
            Filtered to ONLY the formation window.
            
    Returns:
        DataFrame with multi-index (ticker, bucket) and stats (mean, median, p95)
    """
    # Only use valid rows for calculating seasonality
    valid_df = df[df['is_valid']].copy()
    
    # Assign each row to a 30-min bucket (09:30-10:00, 10:00-10:30, etc.)
    # We can use pd.cut or custom rounding
    valid_df['time'] = valid_df['timestamp_et'].dt.time
    
    # Create 13 buckets from 09:30 to 16:00
    # pd.date_range can create the bins
    bins = [pd.to_datetime(f"09:30:00").time()]
    for h in range(10, 16):
        bins.append(pd.to_datetime(f"{h}:00:00").time())
        bins.append(pd.to_datetime(f"{h}:30:00").time())
    bins.append(pd.to_datetime(f"16:00:00").time()) # slightly past 15:59:59 to include it
    
    # Create labels for the 13 buckets
    labels = []
    for i in range(len(bins)-1):
        t1_str = bins[i].strftime("%H:%M")
        t2_str = bins[i+1].strftime("%H:%M")
        # 16:00 goes back to 15:59 for naming
        if t2_str == "16:00":
            t2_str = "15:59"
        labels.append(f"{t1_str}-{t2_str}")
        
    # Vectorized: map each timestamp to seconds-since-midnight on a common date for pd.cut
    common_date = pd.Timestamp('2000-01-01')
    seconds = (
        valid_df['timestamp_et'].dt.hour * 3600
        + valid_df['timestamp_et'].dt.minute * 60
        + valid_df['timestamp_et'].dt.second
    )
    temp_dt = common_date + pd.to_timedelta(seconds, unit='s')

    dt_bins = [
        common_date + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        for t in bins
    ]

    valid_df['bucket'] = pd.cut(temp_dt, bins=dt_bins, labels=labels, right=False, include_lowest=True)
    
    def p95(x):
        return x.quantile(0.95)
        
    # Group by ticker and bucket
    seasonality = valid_df.groupby(['ticker', 'bucket'], observed=False)['full_spread_l1_bps'].agg(
        ['mean', 'median', p95]
    ).reset_index()
    
    return seasonality
