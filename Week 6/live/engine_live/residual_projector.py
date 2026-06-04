"""Project new daily bars through formation factor loadings.

Wraps engine.phase1_cointegration.factor_residual.project_residual (same
function backtest uses in run_v4_pipeline.py) + Path A re-anchor.

Path A re-anchor: shift each ticker's trading residual so its FIRST bar
continues from the LAST formation residual value. This avoids a spurious
log-price discontinuity at the formation/trading boundary.

Per [[week6-live-invariants]] item 5: re-anchor is required for live to
match backtest spread calculation.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from engine.phase1_cointegration.factor_residual import project_residual


class ResidualProjector:
    """Apply factor projection + re-anchor to incoming daily bars.

    Lifecycle:
      1. Construct with path to saved factor_state.pkl (from discovery).
      2. Call compute_residuals(trading_data) per day (or per batch of bars).
         - trading_data: dict[ticker] -> DataFrame with 'log_close' column
      3. Returns dict[ticker] -> residual log-price Series for the trading window.
      4. Re-anchor shift is computed ONCE per ticker on its first trading bar,
         then cached for subsequent calls.
    """

    def __init__(self, factor_state_path: Path | str) -> None:
        path = Path(factor_state_path)
        with path.open("rb") as f:
            fs = pickle.load(f)
        self.loadings_W: np.ndarray = fs["loadings_W"]
        self.tickers: list[str] = list(fs["tickers"])
        self.formation_residuals: pd.DataFrame = fs["residual_log_prices"]
        # Per-ticker last formation residual value (used for re-anchor)
        last_row = self.formation_residuals.dropna(how="all").iloc[-1]
        self.formation_last: dict[str, float] = {
            tk: float(v) for tk, v in last_row.items() if pd.notna(v)
        }
        # Cached re-anchor shift per ticker (computed once on first trading bar)
        self._anchor_shifts: dict[str, float] = {}

    def compute_residuals(self, trading_data: dict[str, pd.DataFrame],
                          min_obs: int = 2) -> dict[str, pd.Series]:
        """Project + re-anchor. Returns dict[ticker] -> residual Series.

        min_obs=2: the LIVE engine runs day-by-day and only consumes the LATEST
        residual value each day (main._run_daily_decisions pushes one point into
        the formation-seeded ZTracker). The Z is computed from the 60-obs
        formation seed + that one new point, so it does NOT need a full window of
        trading obs. The backtest's run_v4_pipeline.run_fold_v4 uses min_obs=10
        because it scores a whole month-fold at once (always >=10 obs); applying
        that gate to the day-by-day live engine wrongly blocks the first ~9
        trading days of every month (the 'too few aligned trading obs' crash).
        2 is the floor: >=2 aligned obs are needed to compute one return. Day 1
        of a month (1 obs, no return) is correctly skipped — its residual would
        be pinned to the stale formation-end anchor with no new information.
        Residual/Z MATH is unchanged; only the gate is relaxed (live-only).
        """
        raw = project_residual(
            trading_data, self.loadings_W, self.tickers, min_obs=min_obs,
        )
        out: dict[str, pd.Series] = {}
        for tk, series in raw.items():
            if series is None or series.empty:
                out[tk] = series
                continue
            anchor = self._get_or_compute_anchor(tk, series)
            if anchor is None:
                out[tk] = series
            else:
                out[tk] = series + anchor
        return out

    def _get_or_compute_anchor(self, ticker: str, series: pd.Series) -> float | None:
        """Path A: shift = formation_last - trading_first.

        Caches per ticker — once computed, reused. If formation_last unknown
        (ticker not in formation), no anchor applied.
        """
        if ticker in self._anchor_shifts:
            return self._anchor_shifts[ticker]
        if ticker not in self.formation_last:
            return None
        first_clean = series.dropna()
        if first_clean.empty:
            return None
        shift = self.formation_last[ticker] - float(first_clean.iloc[0])
        self._anchor_shifts[ticker] = shift
        return shift

    def reset_anchors(self) -> None:
        """Clear cached re-anchor shifts (e.g. at start of a new fold / month)."""
        self._anchor_shifts.clear()

    def n_formation_tickers(self) -> int:
        return len(self.tickers)
