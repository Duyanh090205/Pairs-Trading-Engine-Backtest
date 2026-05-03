import pandas as pd

def process_orderbook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingests raw orderbook data, localizes timestamps to US/Eastern,
    applies session filtering (09:30-15:59 ET), and flags invalid quotes.
    
    Args:
        df: Raw orderbook DataFrame. Must contain 'timestamp', 'l1_bid_px', 'l1_ask_px'.
            
    Returns:
        DataFrame with ET timestamps, session filtered, and invalid quotes flagged.
    """
    df = df.copy()

    # 1. Localize to US/Eastern
    if df['timestamp'].dt.tz is None:
        df['timestamp_et'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        df['timestamp_et'] = df['timestamp'].dt.tz_convert('US/Eastern')
        
    # 2. Session filter: 09:30-15:59 ET
    # Create time column for easier filtering
    times = df['timestamp_et'].dt.time
    # 09:30:00 to 15:59:59
    session_mask = (times >= pd.to_datetime('09:30:00').time()) & (times <= pd.to_datetime('15:59:59').time())
    
    df_filtered = df[session_mask].copy()
    
    # 3. Flag (but DO NOT DROP) invalid quotes
    # Reason: Dropping breaks 1-min timeline alignment with Week 4 trades
    # Valid if: ask > bid AND bid > 0
    df_filtered['is_valid'] = (df_filtered['l1_ask_px'] > df_filtered['l1_bid_px']) & (df_filtered['l1_bid_px'] > 0)
    
    return df_filtered
