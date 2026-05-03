def check_dynamic_cost_blowup(mean_rt_cost_bps: float) -> bool:
    """
    Trigger: Mean RT cost > 150 bps any fold.
    Action: Audit spread data alignment, check for stale quotes.
    Returns True if red flag is triggered.
    """
    return mean_rt_cost_bps > 150.0

def check_cost_exceeds_alpha(net_sharpe: float, gross_sharpe: float) -> bool:
    """
    Trigger: Net Sharpe < 0 AND Gross Sharpe > 1.0.
    Action: Flag: alpha real but untradeable at this frequency.
    Returns True if red flag is triggered.
    """
    return net_sharpe < 0.0 and gross_sharpe > 1.0

def check_spread_data_gap(missing_pct: float) -> bool:
    """
    Trigger: > 5% missing spread observations in trading window.
    Action: Fall back to static 60 bps for affected bars.
    Returns True if red flag is triggered.
    """
    return missing_pct > 0.05

def check_kappa_instability(tier_changes_count: int) -> bool:
    """
    Trigger: κ tier changes > 2× across consecutive folds for same ticker.
    Action: Flag: liquidity regime shift.
    Returns True if red flag is triggered.
    """
    return tier_changes_count > 2

def check_nc_leak(nc_dynamic_sharpe: float) -> bool:
    """
    Trigger: NC Sharpe under dynamic model > 0.5.
    Action: Audit cost model for sign errors or look-ahead.
    Returns True if red flag is triggered.
    """
    return nc_dynamic_sharpe > 0.5

def check_dsr_degradation(failure_prob_net: float, failure_prob_gross: float) -> bool:
    """
    Trigger: net DSR < 0.90 (failure_prob > 0.10) while gross DSR > 0.95 (failure_prob < 0.05).
    Inputs are 1 - DSR (probability that true Sharpe <= 0), not DSR itself.
    Action: Flag: gross alpha is strongly significant but costs erode statistical confidence.
    Returns True if red flag is triggered.
    """
    return failure_prob_net > 0.10 and failure_prob_gross < 0.05

def check_math_violation(total_cost_dollars: float, net_pnl_dollars: float, gross_pnl_dollars: float) -> bool:
    """
    Trigger: total_cost_dollars < 0 OR net_pnl_dollars > gross_pnl_dollars.
    Action: Throw Error: Cost model must strictly subtract PnL.
    Returns True if red flag is triggered.
    """
    return total_cost_dollars < 0.0 or net_pnl_dollars > gross_pnl_dollars
