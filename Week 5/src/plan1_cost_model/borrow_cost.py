import pandas as pd

def calculate_borrow_cost(short_notional: float, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, borrow_rate_bps_annual: float = 50.0) -> float:
    """
    Calculates the total borrow cost for a short position held overnight.
    borrow_cost_daily = (borrow_rate_bps_annual / 10,000) / 365 * short_notional_$

    Day-count convention: Actual/365. Borrow accrues on every calendar day
    (including weekends — the short cannot be returned on Saturday/Sunday), so the
    annual rate is divided by 365, not 252.  holding_days is calendar days.

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
        
    # holding_days = calendar days the position is held overnight (includes weekends).
    # Borrow accrues every calendar day, so the daily fraction is annual_rate / 365,
    # not annual_rate / 252 (which would over-charge for weekend spans).
    holding_days = (exit_ts.date() - entry_ts.date()).days

    borrow_cost_daily = (borrow_rate_bps_annual / 10000.0) / 365.0 * short_notional

    return borrow_cost_daily * holding_days
