"""
Cost engine wrapper for V4 daily — wires Week 5 dynamic cost model into V4.

Three primitive cost functions (from Week 5 cost/plan1_cost_model):
    spread_cost  = half_spread_bps × notional
    impact_cost  = kappa × spread_std_1d × notional
    borrow_cost  = (annual_rate_bps / 10000) / 365 × short_notional × holding_days

Kappa tiers (from formation-window median spread):
    Tier 1 (Tight,  < 8 bps median):  kappa = 0.3
    Tier 2 (Medium, 8-20 bps):        kappa = 0.5
    Tier 3 (Wide,  > 20 bps):         kappa = 0.8

Daily strategy adaptation: per-ticker per-day MEDIAN spread / spread_std are
used (built by build_daily_spread_cache.py). Look up at entry day + exit day
for each round-trip leg. If a ticker has no spread data for a given date,
fall back to formation median (from spread_summary).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Conservative fallback when daily cache has no entry for a ticker/date
_FALLBACK_HALF_SPREAD_BPS = 30.0
_FALLBACK_SPREAD_STD = 15.0   # triggers Plan 1's 15-bps impact fallback equivalent

# Explicit broker commission (per side, per leg) — Week 5 cost engine models
# only spread + impact + borrow. Institutional commission for liquid US equity
# is ~1-2 bps. We use 2 bps as a conservative default so the full-cost number
# is audit-defensible against real execution.
_COMMISSION_BPS_PER_SIDE_PER_LEG = 2.0

_WEEK5_SUMMARY = Path(r"d:\Quant Finance\Quant Program\Week 5\data\microstructure\spread_summary.parquet")
_DEFAULT_CACHE = Path(r"d:\Quant Finance\Quant Program\Week 6\cost\daily_spread_cache.parquet")


def assign_kappa_tier(median_full_spread_bps: float) -> float:
    """Per AL2010 spec: tier-based impact coefficient from formation median spread."""
    if median_full_spread_bps < 8.0:
        return 0.3
    if median_full_spread_bps <= 20.0:
        return 0.5
    return 0.8


@dataclass
class CostData:
    """Container for everything the dynamic cost calculation needs."""
    # MultiIndex (ticker, date) -> [half_spread_bps, spread_std_1d]
    daily_spread: pd.DataFrame
    # ticker -> kappa
    kappa_map: dict[str, float]
    # Fallback for tickers with no daily-cache entry
    fallback_kappa: float = 0.5


def load_cost_data(
    daily_cache_path: Path = _DEFAULT_CACHE,
    summary_path: Path = _WEEK5_SUMMARY,
) -> CostData:
    """Load daily spread cache + spread summary; build kappa map."""
    if not daily_cache_path.exists():
        raise FileNotFoundError(
            f"Daily spread cache not found at {daily_cache_path}\n"
            "Run `python build_daily_spread_cache.py` first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Spread summary not found: {summary_path}")

    cache = pd.read_parquet(daily_cache_path)
    # Ensure tz-naive midnight Timestamps for date column
    cache["date"] = pd.to_datetime(cache["date"]).dt.tz_localize(None).dt.normalize()
    cache = cache.set_index(["ticker", "date"]).sort_index()

    summary = pd.read_parquet(summary_path)
    kappa_map = {
        row["ticker"]: assign_kappa_tier(float(row["median_bps"]))
        for _, row in summary.iterrows()
    }
    log.info("cost_engine: loaded %d daily rows, %d tickers in kappa_map",
             len(cache), len(kappa_map))
    return CostData(daily_spread=cache, kappa_map=kappa_map)


def _lookup_daily_spread(
    cd: CostData, ticker: str, date: pd.Timestamp,
) -> tuple[float, float]:
    """Returns (half_spread_bps, spread_std_1d) for the ticker on this date.
    Falls back to ticker-level summary if the exact date is missing; falls back
    to (_FALLBACK_HALF_SPREAD_BPS, _FALLBACK_SPREAD_STD) if ticker not in cache.
    """
    # normalize date to tz-naive midnight to match cache index
    date = pd.Timestamp(date).tz_localize(None).normalize()
    try:
        row = cd.daily_spread.loc[(ticker, date)]
        hs = float(row["half_spread_bps"])
        sd = float(row["spread_std_1d"])
        if not np.isfinite(hs):
            hs = _FALLBACK_HALF_SPREAD_BPS
        if not np.isfinite(sd):
            sd = _FALLBACK_SPREAD_STD
        return hs, sd
    except KeyError:
        # Try at-or-before lookup within this ticker
        try:
            ticker_slice = cd.daily_spread.loc[ticker]
            sub = ticker_slice.loc[:date]
            if len(sub) > 0:
                row = sub.iloc[-1]
                hs = float(row["half_spread_bps"])
                sd = float(row["spread_std_1d"])
                if not np.isfinite(hs):
                    hs = _FALLBACK_HALF_SPREAD_BPS
                if not np.isfinite(sd):
                    sd = _FALLBACK_SPREAD_STD
                return hs, sd
        except KeyError:
            pass
        return _FALLBACK_HALF_SPREAD_BPS, _FALLBACK_SPREAD_STD


def _impact_bps(kappa: float, sigma: float) -> float:
    """Impact in bps for one leg-side. NaN sigma → 15 bps fallback."""
    if not np.isfinite(sigma) or sigma <= 0:
        return 15.0
    return kappa * sigma


def compute_pair_trade_cost(
    cd: CostData,
    ticker_a: str, ticker_b: str,
    entry_date: pd.Timestamp, exit_date: pd.Timestamp,
    notional_per_leg: float,          # half of per-pair notional, applied to each leg
    side_a: int,                       # +1 if long A, -1 if short A
    borrow_rate_bps_annual: float = 50.0,
    commission_bps_per_side: float = _COMMISSION_BPS_PER_SIDE_PER_LEG,
) -> dict[str, float]:
    """Per-side decomposed cost in dollars for one pair trade.

    Returns a dict with entry-side and exit-side components separated, so
    a caller (e.g. the daily engine) can book costs at the right bars and
    keep per-component decomposition for audit.

    Keys:
        spread_entry_$, spread_exit_$         (bid-ask spread, both legs)
        impact_entry_$, impact_exit_$         (kappa × spread_std × notional)
        commission_entry_$, commission_exit_$ (broker commission)
        borrow_$                              (calendar-day accrual, short leg)
        total_$                               (sum)

    Each leg has `notional_per_leg` at entry and exit. The short leg accrues
    borrow on calendar days from entry to exit. Commission is added explicitly
    because the Week 5 engine models only spread + impact + borrow.
    """
    zero_return = {
        "spread_entry_$": 0.0, "spread_exit_$": 0.0,
        "impact_entry_$": 0.0, "impact_exit_$": 0.0,
        "commission_entry_$": 0.0, "commission_exit_$": 0.0,
        "borrow_$": 0.0, "total_$": 0.0,
    }
    if notional_per_leg <= 0:
        return zero_return

    hsa_in, sa_in = _lookup_daily_spread(cd, ticker_a, entry_date)
    hsb_in, sb_in = _lookup_daily_spread(cd, ticker_b, entry_date)
    hsa_out, sa_out = _lookup_daily_spread(cd, ticker_a, exit_date)
    hsb_out, sb_out = _lookup_daily_spread(cd, ticker_b, exit_date)

    kA = cd.kappa_map.get(ticker_a, cd.fallback_kappa)
    kB = cd.kappa_map.get(ticker_b, cd.fallback_kappa)

    # Spread cost — per leg per side
    spread_entry = notional_per_leg * (hsa_in + hsb_in) / 10000.0
    spread_exit  = notional_per_leg * (hsa_out + hsb_out) / 10000.0

    # Impact cost — per leg per side; NaN-σ → 15 bps fallback inside _impact_bps
    impact_entry = notional_per_leg * (_impact_bps(kA, sa_in) + _impact_bps(kB, sb_in)) / 10000.0
    impact_exit  = notional_per_leg * (_impact_bps(kA, sa_out) + _impact_bps(kB, sb_out)) / 10000.0

    # Commission — per leg per side; 2 legs × commission_bps_per_side per fill
    commission_entry = notional_per_leg * 2.0 * commission_bps_per_side / 10000.0
    commission_exit  = notional_per_leg * 2.0 * commission_bps_per_side / 10000.0

    # Borrow — calendar-day accrual on the short leg; same-day → 0
    short_notional = notional_per_leg if side_a != 0 else 0.0
    entry_ts = pd.Timestamp(entry_date).tz_localize(None).normalize()
    exit_ts  = pd.Timestamp(exit_date).tz_localize(None).normalize()
    if exit_ts.date() == entry_ts.date():
        borrow = 0.0
    else:
        holding_cal_days = (exit_ts.date() - entry_ts.date()).days
        borrow = (borrow_rate_bps_annual / 10000.0) / 365.0 * short_notional * holding_cal_days

    total = (spread_entry + spread_exit + impact_entry + impact_exit
             + commission_entry + commission_exit + borrow)
    return {
        "spread_entry_$": float(spread_entry),
        "spread_exit_$":  float(spread_exit),
        "impact_entry_$": float(impact_entry),
        "impact_exit_$":  float(impact_exit),
        "commission_entry_$": float(commission_entry),
        "commission_exit_$":  float(commission_exit),
        "borrow_$": float(borrow),
        "total_$":  float(total),
    }
