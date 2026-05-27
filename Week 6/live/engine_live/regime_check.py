"""Composite z-score regime filter — DELEGATES to engine_daily.regime_detector.

This is the live wrapper that the deep-audit-bug flagged as a HIGH-risk twin
function pattern. The fix per [[week6-live-invariants]]: import and call the
backtest functions directly, do NOT reimplement.

Cache source: live/state/daily_cache (Alpaca split-adjusted), NOT the legacy
Polygon Week 4 cache. See week6_live_invariants.md item 12 — Polygon cache
has raw unadjusted prices that would poison the trailing 252d stress features.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# Delegate target — the SAME functions the V4 backtest used to ship Sharpe +1.28.
from engine_daily.regime_detector import (
    build_features,
    compute_stress_zscore,
    halt_for_fold,
)


@dataclass
class RegimeDecision:
    month: str           # YYYY-MM
    stress_z: float | None
    q67_thresh: float | None
    halted: bool
    diag: dict


# Live cache (Alpaca split-adjusted). Override via env var DAILY_CACHE_DIR for tests.
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "live" / "state" / "daily_cache"
DAILY_CACHE_DIR = Path(os.environ.get("DAILY_CACHE_DIR", str(_DEFAULT_CACHE)))


def _load_daily_cache(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load only the tickers we trade. Saves memory vs loading all 528."""
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        fp = DAILY_CACHE_DIR / f"{tk}.parquet"
        if not fp.exists():
            continue
        out[tk] = pd.read_parquet(fp)
    return out


def decide_month(conn: sqlite3.Connection, decision_ts: datetime,
                 universe_tickers: list[str]) -> RegimeDecision:
    """Decide halt for the calendar month containing `decision_ts`.

    Uses the SAME functions as the V4 backtest:
      build_features → compute_stress_zscore → halt_for_fold

    Persists the decision into `regime_decisions` table for audit.
    """
    cache = _load_daily_cache(universe_tickers)
    if not cache:
        raise RuntimeError(
            f"No daily cache found for {len(universe_tickers)} tickers at {DAILY_CACHE_DIR}"
        )
    feats = build_features(cache)
    feats = compute_stress_zscore(feats)

    # First day of the month containing decision_ts
    trade_start = pd.Timestamp(decision_ts.replace(day=1, hour=0, minute=0, second=0,
                                                    microsecond=0, tzinfo=None))
    halt, diag = halt_for_fold(feats, trade_start)

    month_str = trade_start.strftime("%Y-%m")
    stress_z = diag.get("stress_z")
    q67 = diag.get("q67_trailing")

    # Only persist a regime_decisions row when stress_z is computable. If burnin
    # is insufficient (diag returns stress_z=None / q67 missing), skip the
    # INSERT — the schema requires NOT NULL on these columns, and persisting
    # NaN would either be rejected (SQLite NaN→NULL conversion on some platforms)
    # or hide the fact that we couldn't decide.
    if stress_z is not None and q67 is not None:
        conn.execute(
            "INSERT OR REPLACE INTO regime_decisions "
            "(month, stress_z, q67_thresh, halted, decided_ts) VALUES (?, ?, ?, ?, ?)",
            (month_str, float(stress_z), float(q67),
             1 if halt else 0, decision_ts.isoformat()),
        )
    return RegimeDecision(
        month=month_str, stress_z=stress_z, q67_thresh=q67, halted=halt, diag=diag,
    )
