"""
Per-trade cost computation hooks.
Calls Plan 1 primitives; produces dollar costs for each trade event.
"""
import pandas as pd
import numpy as np
from src.plan1_cost_model import (
    calculate_spread_cost,
    calculate_impact_cost,
    calculate_borrow_cost,
    calculate_static_round_trip_cost,
)


# Default fallback used when spread lookup fails completely (no bar at-or-before ts).
# Conservative — higher than any realistic L1 half-spread under stress.
_FALLBACK_HALF_SPREAD_BPS = 30.0
_FALLBACK_SIGMA_BPS = float("nan")  # triggers Plan 1's 15-bps impact fallback


def _lookup_at(spreads_idx: pd.DataFrame, ticker: str, ts: pd.Timestamp) -> tuple[float, float]:
    """
    Returns (half_spread_l1_bps, spread_std_1d) for the bar at-or-before `ts`.

    Parameters
    ----------
    spreads_idx : DataFrame
        MultiIndex [ticker, timestamp_et], sorted. Must contain columns
        'half_spread_l1_bps' and 'spread_std_1d'.

    Returns conservative fallback if no bar exists at-or-before ts for that ticker.
    """
    try:
        ticker_slice = spreads_idx.loc[ticker]
    except KeyError:
        return _FALLBACK_HALF_SPREAD_BPS, _FALLBACK_SIGMA_BPS

    # Find bar at-or-before ts
    sub = ticker_slice.loc[:ts]
    if sub.empty:
        return _FALLBACK_HALF_SPREAD_BPS, _FALLBACK_SIGMA_BPS

    row = sub.iloc[-1]
    hs = float(row["half_spread_l1_bps"]) if not pd.isna(row["half_spread_l1_bps"]) else _FALLBACK_HALF_SPREAD_BPS
    sigma = float(row["spread_std_1d"]) if not pd.isna(row["spread_std_1d"]) else _FALLBACK_SIGMA_BPS
    return hs, sigma


def trade_dynamic_cost(
    trade_row,
    spreads_idx: pd.DataFrame,
    kappa_map: dict[str, float],
    borrow_rate_bps_annual: float = 50.0,
) -> dict[str, float]:
    """
    Dynamic-cost dollars for one round-trip pair trade.

    Returns {spread_$, impact_$, borrow_$, total_$}.

    `trade_row` exposes attribute access: ticker_A, ticker_B, side_A, side_B,
    entry_ts, exit_ts, notional_A_entry, notional_B_entry,
    notional_A_exit, notional_B_exit.
    """
    hsa_in, sa_in = _lookup_at(spreads_idx, trade_row.ticker_A, trade_row.entry_ts)
    hsb_in, sb_in = _lookup_at(spreads_idx, trade_row.ticker_B, trade_row.entry_ts)
    hsa_out, sa_out = _lookup_at(spreads_idx, trade_row.ticker_A, trade_row.exit_ts)
    hsb_out, sb_out = _lookup_at(spreads_idx, trade_row.ticker_B, trade_row.exit_ts)

    kA = kappa_map.get(trade_row.ticker_A, 0.5)
    kB = kappa_map.get(trade_row.ticker_B, 0.5)

    e_sp_A = calculate_spread_cost(hsa_in, trade_row.notional_A_entry)
    e_sp_B = calculate_spread_cost(hsb_in, trade_row.notional_B_entry)
    e_im_A = calculate_impact_cost(kA, sa_in, trade_row.notional_A_entry)
    e_im_B = calculate_impact_cost(kB, sb_in, trade_row.notional_B_entry)

    x_sp_A = calculate_spread_cost(hsa_out, trade_row.notional_A_exit)
    x_sp_B = calculate_spread_cost(hsb_out, trade_row.notional_B_exit)
    x_im_A = calculate_impact_cost(kA, sa_out, trade_row.notional_A_exit)
    x_im_B = calculate_impact_cost(kB, sb_out, trade_row.notional_B_exit)

    spread_total = e_sp_A + e_sp_B + x_sp_A + x_sp_B
    impact_total = e_im_A + e_im_B + x_im_A + x_im_B

    # Identify short leg by side; borrow on short notional only
    if trade_row.side_A == -1 and trade_row.side_B == 1:
        short_notional = trade_row.notional_A_entry
    elif trade_row.side_B == -1 and trade_row.side_A == 1:
        short_notional = trade_row.notional_B_entry
    else:
        short_notional = 0.0  # degenerate: both legs same side; no borrow

    borrow = calculate_borrow_cost(
        short_notional, trade_row.entry_ts, trade_row.exit_ts, borrow_rate_bps_annual
    )

    total = spread_total + impact_total + borrow
    return {
        "spread_$": spread_total,
        "impact_$": impact_total,
        "borrow_$": borrow,
        "total_$": total,
    }


def trade_static_cost(trade_row, tc_bps_per_leg: float = 30.0) -> float:
    """Week 4 static baseline: Plan 1's calculate_static_round_trip_cost()."""
    return calculate_static_round_trip_cost(
        trade_row.notional_A_entry,
        trade_row.notional_B_entry,
        trade_row.notional_A_exit,
        trade_row.notional_B_exit,
        tc_bps_per_leg=tc_bps_per_leg,
    )


def rebalance_dynamic_cost(
    reb_row,
    spreads_idx: pd.DataFrame,
    kappa_map: dict[str, float],
) -> float:
    """
    One-sided spread+impact at rebalance_ts on notional_rebalanced.
    Used for each row in rebalance_log.
    """
    hs, sigma = _lookup_at(spreads_idx, reb_row.ticker, reb_row.rebalance_ts)
    k = kappa_map.get(reb_row.ticker, 0.5)
    sp = calculate_spread_cost(hs, abs(reb_row.notional_rebalanced))
    im = calculate_impact_cost(k, sigma, abs(reb_row.notional_rebalanced))
    return sp + im
