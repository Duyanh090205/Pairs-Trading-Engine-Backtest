import pandas as pd
import numpy as np

def calculate_spread_cost(half_spread_l1_bps: float, notional_dollars: float) -> float:
    """
    Calculates the spread cost component (in dollars) for a given trade leg.
    C_spread(t) = half_spread_l1_bps(t) * notional_dollars
    
    Args:
        half_spread_l1_bps: The half spread in basis points at time t.
        notional_dollars: The dollar value of the trade leg.
        
    Returns:
        The spread cost in dollars.
    """
    return notional_dollars * (half_spread_l1_bps / 10000.0)
