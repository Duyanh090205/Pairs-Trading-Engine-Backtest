def calculate_round_trip_cost(
    entry_cost_A: float, 
    entry_cost_B: float, 
    exit_cost_A: float, 
    exit_cost_B: float, 
    borrow_cost: float
) -> float:
    """
    Calculates the total round-trip execution cost for a pair trade in dollars.
    
    total_trade_cost_$ = cost_entry_A_$ + cost_entry_B_$ + cost_exit_A_$ + cost_exit_B_$ + total_borrow_$
    
    Args:
        entry_cost_A: Dollar cost of entering leg A.
        entry_cost_B: Dollar cost of entering leg B.
        exit_cost_A: Dollar cost of exiting leg A.
        exit_cost_B: Dollar cost of exiting leg B.
        borrow_cost: Total dollar borrow cost for the trade.
        
    Returns:
        The total round-trip cost in dollars.
    """
    return entry_cost_A + entry_cost_B + exit_cost_A + exit_cost_B + borrow_cost


def calculate_static_round_trip_cost(
    notional_A_entry: float,
    notional_B_entry: float,
    notional_A_exit: float,
    notional_B_exit: float,
    tc_bps_per_leg: float = 30.0,
) -> float:
    """
    Week 4 static cost baseline: fixed bps per leg regardless of market conditions.
    Default 30 bps/leg = 60 bps per side = 120 bps total round-trip notional-weighted.

    Matches Week 4 pnl.py: tc_rate * (notional_A + notional_B) applied at entry and exit.
    """
    entry_cost = (notional_A_entry + notional_B_entry) * (tc_bps_per_leg / 10000.0)
    exit_cost = (notional_A_exit + notional_B_exit) * (tc_bps_per_leg / 10000.0)
    return entry_cost + exit_cost
