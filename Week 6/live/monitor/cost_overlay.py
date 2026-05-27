"""Dual cost track: broker raw (slippage) + backtest dynamic overlay.

Why this exists:
  Alpaca paper account has $0 commission, $0 borrow, near-zero spread cost
  (fills at NBBO mid). So paper P&L is artificially BETTER than the live
  thing would be. To honestly report what the strategy would have done in
  reality, we overlay the backtest's dynamic cost model on top of broker
  fills.

Two cost numbers per closed pair trade:
  realized_cost_bps  = ((fill_price - decision_price)*signed_qty per leg)
                       / notional, summed → BROKER's actual slippage
  predicted_cost_bps = backtest compute_pair_trade_cost / notional
                       → what the V4 cost model would have charged

  drift_bps = realized_cost_bps - predicted_cost_bps
              (positive: real cost worse than model; negative: better)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine_daily.cost_engine import (
    _COMMISSION_BPS_PER_SIDE_PER_LEG,
    CostData,
    compute_pair_trade_cost,
    load_cost_data,
)


@dataclass
class CostBreakdown:
    realized_cost_bps: float        # broker slippage normalized to notional
    predicted_cost_bps: float        # V4 dynamic cost model
    drift_bps: float                 # realized - predicted
    realized_cost_usd: float
    predicted_cost_usd: float


def compute_realized_cost_usd(conn: sqlite3.Connection, pair_id: str,
                              bar_ts_entry: str, bar_ts_exit: str) -> tuple[float, float]:
    """Return (slippage_usd, total_traded_notional_usd) for both entry + exit legs.

    Slippage convention: BUY filled HIGHER than decision = positive cost.
                          SELL filled LOWER than decision = positive cost.
    """
    rows = conn.execute(
        "SELECT side, qty, fill_qty, fill_price, decision_price "
        "FROM orders WHERE pair_id = ? AND bar_ts IN (?, ?) AND status = 'filled'",
        (pair_id, bar_ts_entry, bar_ts_exit),
    ).fetchall()
    slip_usd = 0.0
    notional_usd = 0.0
    for r in rows:
        if r["fill_qty"] is None or r["fill_price"] is None or r["decision_price"] is None:
            continue
        qty = float(r["fill_qty"])
        fp = float(r["fill_price"])
        dp = float(r["decision_price"])
        if r["side"] == "buy":
            slip_usd += qty * (fp - dp)
        else:  # sell
            slip_usd += qty * (dp - fp)
        notional_usd += qty * dp
    return slip_usd, notional_usd


def compute_predicted_cost_usd(cost_data: CostData | None,
                               ticker_a: str, ticker_b: str,
                               entry_ts: str, exit_ts: str,
                               notional_per_leg: float,
                               side_a: int) -> float:
    """Apply backtest's dynamic cost model. Returns total $ cost (entry + exit + borrow)."""
    if cost_data is None:
        # Fallback (when Week 5 cost cache unavailable):
        #   Per fill: 30 bps half-spread + 2 bps commission = 32 bps
        #   Total fills per pair trade: 4 (2 legs * (entry + exit))
        #   Each fill applies to `notional_per_leg`.
        #   Total bps cost in $ = notional_per_leg * (4 * 32) / 10_000
        #                       = notional_per_leg * 2 * 64 / 10_000
        bps_per_fill = 30 + 2          # half-spread + commission
        flat_total_bps = 2 * bps_per_fill * 2   # 2 legs * (entry + exit)
        flat_cost = notional_per_leg * flat_total_bps / 10_000
        # Borrow (calendar-day accrual on the short leg)
        d_entry = pd.Timestamp(entry_ts).normalize()
        d_exit = pd.Timestamp(exit_ts).normalize()
        days = max(0, (d_exit.date() - d_entry.date()).days)
        borrow = (50.0 / 10_000.0) / 365.0 * notional_per_leg * days
        return flat_cost + borrow
    cost = compute_pair_trade_cost(
        cost_data, ticker_a, ticker_b,
        pd.Timestamp(entry_ts), pd.Timestamp(exit_ts),
        notional_per_leg=notional_per_leg, side_a=side_a,
    )
    return cost["total_$"]


def compute_dual_cost(conn: sqlite3.Connection, cost_data: CostData | None,
                      pair_id: str, ticker_a: str, ticker_b: str,
                      entry_ts: str, exit_ts: str,
                      notional_per_leg: float, side_a: int,
                      bar_ts_entry: str, bar_ts_exit: str) -> CostBreakdown:
    realized_usd, traded_usd = compute_realized_cost_usd(
        conn, pair_id, bar_ts_entry, bar_ts_exit,
    )
    predicted_usd = compute_predicted_cost_usd(
        cost_data, ticker_a, ticker_b, entry_ts, exit_ts,
        notional_per_leg=notional_per_leg, side_a=side_a,
    )
    denom = traded_usd if traded_usd > 0 else (notional_per_leg * 2)
    realized_bps = realized_usd / denom * 10_000 if denom > 0 else 0.0
    predicted_bps = predicted_usd / denom * 10_000 if denom > 0 else 0.0
    return CostBreakdown(
        realized_cost_bps=realized_bps,
        predicted_cost_bps=predicted_bps,
        drift_bps=realized_bps - predicted_bps,
        realized_cost_usd=realized_usd,
        predicted_cost_usd=predicted_usd,
    )


def try_load_cost_data() -> CostData | None:
    """Try to load Week 5 cost cache; return None if missing (fallback to flat)."""
    try:
        return load_cost_data()
    except FileNotFoundError:
        return None
