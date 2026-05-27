"""Rolling 60-day Z-score with formation warmup (mirrors backtest).

CRITICAL: must match engine_daily.engine_daily.rolling_z_with_warmup exactly.
Backtest uses sample std (ddof=1) via pandas `.std(ddof=1)` and clips to 1e-10.
Backtest's pandas .rolling().std() silently skips NaN — we mirror that.
"""
from __future__ import annotations

import math
import statistics
from collections import deque

_STD_FLOOR = 1e-10  # mirrors backtest .clip(lower=1e-10)


class ZTracker:
    """Rolling Z-score on residual spread.

    Seed with last `window` formation-window spread values so first live bar
    already has a full window. Std is **sample** (ddof=1) to match backtest.
    NaN handling: matches pandas .rolling().std() — NaN values are skipped
    in the stats; if fewer than `window` non-NaN values present, returns None.
    """

    def __init__(self, window: int = 60, seed: list[float] | None = None) -> None:
        self.window = window
        self.buf: deque[float] = deque(maxlen=window)
        if seed:
            self.buf.extend(seed[-window:])

    def push(self, spread: float) -> float | None:
        """Append new spread; return current Z (or None if insufficient data)."""
        self.buf.append(spread)
        if len(self.buf) < self.window:
            return None
        # Mirror backtest NaN-skipping: filter NaN before computing stats.
        clean = [v for v in self.buf if not math.isnan(v)]
        if len(clean) < self.window:
            return None
        # If the *current* bar is NaN, backtest would emit NaN Z — we emit None.
        if math.isnan(spread):
            return None
        mean = statistics.fmean(clean)
        std = max(statistics.stdev(clean), _STD_FLOOR)  # ddof=1; floor matches backtest
        return (spread - mean) / std
