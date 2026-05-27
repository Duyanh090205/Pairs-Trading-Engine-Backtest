"""Per-trade sanity checks on fills (slippage, latency, etc.)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyThresholds:
    max_slippage_bps: float = 50.0
    max_l1_depth_share: float = 0.80
    max_order_latency_s: float = 2.0
    max_borrow_bps_yr: float = 100.0


def check_fill(decision_price: float, fill_price: float, side: str,
               thresholds: AnomalyThresholds | None = None) -> list[str]:
    """Return list of anomaly flags for a fill. Empty = clean.

    Slippage convention: BUY filled ABOVE decision = adverse (positive bps).
                          SELL filled BELOW decision = adverse (positive bps).
    """
    th = thresholds or AnomalyThresholds()
    flags: list[str] = []
    if decision_price <= 0 or fill_price <= 0:
        flags.append("invalid_prices")
        return flags
    if side == "buy":
        slip_bps = (fill_price - decision_price) / decision_price * 10_000
    elif side == "sell":
        slip_bps = (decision_price - fill_price) / decision_price * 10_000
    else:
        flags.append(f"unknown_side:{side}")
        return flags
    if slip_bps > th.max_slippage_bps:
        flags.append(f"slippage_bps={slip_bps:.1f}>{th.max_slippage_bps}")
    return flags


def check_latency_s(submitted_ts_epoch: float, filled_ts_epoch: float,
                    thresholds: AnomalyThresholds | None = None) -> list[str]:
    th = thresholds or AnomalyThresholds()
    flags: list[str] = []
    latency = filled_ts_epoch - submitted_ts_epoch
    if latency > th.max_order_latency_s:
        flags.append(f"latency_s={latency:.2f}>{th.max_order_latency_s}")
    return flags
