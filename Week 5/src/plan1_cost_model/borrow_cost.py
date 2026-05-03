import pandas as pd

def calculate_borrow_cost(short_notional: float, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, borrow_rate_bps_annual: float = 50.0) -> float:
    """
    Calculates the total borrow cost for a short position held overnight.
    borrow_cost_daily = (borrow_rate_bps_annual / 10,000) / 252 * short_notional_$
    
    Applies only to overnight short exposure. Intraday (same-day) shorts have zero borrow cost.
    
    Args:
        short_notional: The dollar value of the short position.
        entry_ts: The entry timestamp.
        exit_ts: The exit timestamp.
        borrow_rate_bps_annual: The annual borrow rate in bps (default 50).
        
    Returns:
        The total borrow cost in dollars.
    """
    # If entry and exit are on the same calendar day, borrow cost is 0
    if entry_ts.date() == exit_ts.date():
        return 0.0
        
    # Calculate number of holding days (number of overnight periods)
    # E.g., entered Mon, exited Tue -> 1 overnight period
    holding_days = (exit_ts.date() - entry_ts.date()).days
    
    borrow_cost_daily = (borrow_rate_bps_annual / 10000.0) / 252.0 * short_notional
    
    return borrow_cost_daily * holding_days
