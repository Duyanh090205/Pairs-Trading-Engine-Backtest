import pandas as pd
import numpy as np

def assign_kappa_tier(median_full_spread_bps: float) -> float:
    """
    Assigns the impact coefficient (κ) based on the formation-window median spread.
    
    Tier 1 (Tight, <8 bps median):   κ = 0.3
    Tier 2 (Medium, 8-20 bps):       κ = 0.5
    Tier 3 (Wide, >20 bps):          κ = 0.8
    
    Args:
        median_full_spread_bps: Median full L1 spread in bps over the formation window.
        
    Returns:
        The assigned kappa (κ) value.
    """
    if median_full_spread_bps < 8.0:
        return 0.3
    elif median_full_spread_bps <= 20.0:
        return 0.5
    else:
        return 0.8

def calculate_impact_cost(kappa: float, spread_std_1d: float, notional_dollars: float, multiplier: float = 1.0) -> float:
    """
    Calculates the market impact cost component (in dollars) for a given trade leg.
    C_impact(t) = κ * spread_std_1d(t) * notional_dollars
    
    Args:
        kappa: The impact coefficient assigned to the ticker.
        spread_std_1d: The rolling 1-day standard deviation of the seasonality-adjusted spread.
        notional_dollars: The dollar value of the trade leg.
        multiplier: Sensitivity multiplier for OAT sweeps (default 1.0).
        
    Returns:
        The market impact cost in dollars.
    """
    if pd.isna(spread_std_1d):
        # Fallback: Conservative 15 bps impact cost during burn-in or dropped data periods
        impact_bps = 15.0 * multiplier
    else:
        impact_bps = kappa * spread_std_1d * multiplier
        
    return notional_dollars * (impact_bps / 10000.0)
