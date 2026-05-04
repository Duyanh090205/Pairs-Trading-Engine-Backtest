import pandas as pd

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per ticker, per bar microstructure features:
    mid_px, full_spread_l1_bps, half_spread_l1_bps, 
    full_spread_l2_bps, full_spread_l3_bps, liquidity_l1
    
    Args:
        df: DataFrame processed by ingest.process_orderbook
        
    Returns:
        DataFrame with new microstructure feature columns added.
    """
    df = df.copy()

    # Compute per ticker, per bar
    df['mid_px'] = (df['l1_bid_px'] + df['l1_ask_px']) / 2

    # L1 Spread — clip at 0: crossed-quote bars (bid >= ask, flagged by ingest but not
    # dropped) would produce negative bps and therefore negative cost (revenue), which
    # violates the monotonicity contract.  Clamp rather than drop so the row count and
    # index alignment stay intact.
    df['full_spread_l1_bps'] = (
        ((df['l1_ask_px'] - df['l1_bid_px']) / df['mid_px']) * 10000
    ).clip(lower=0.0)
    df['half_spread_l1_bps'] = df['full_spread_l1_bps'] / 2

    # L2 Spread
    df['full_spread_l2_bps'] = (
        ((df['l2_ask_px'] - df['l2_bid_px']) / df['mid_px']) * 10000
    ).clip(lower=0.0)

    # L3 Spread
    df['full_spread_l3_bps'] = (
        ((df['l3_ask_px'] - df['l3_bid_px']) / df['mid_px']) * 10000
    ).clip(lower=0.0)
    
    # Liquidity: bid side depth in dollars (one side, per spec)
    df['liquidity_l1'] = df['l1_bid_sz'] * df['mid_px']
    
    return df
