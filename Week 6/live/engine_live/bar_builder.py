"""Aggregate trade ticks into 1-minute bars (OHLCV). In-memory.

Conventions:
  - Bar timestamp = minute START in UTC (e.g. tick at 14:30:42.5 → bar 14:30:00).
  - A bar closes when the FIRST tick of the next minute arrives.
  - VWAP (`vwap_num` / `vwap_den`) computed for diagnostic; not used by engine.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Bar:
    ticker: str
    minute_start_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap_num: float = 0.0
    vwap_den: float = 0.0
    n_ticks: int = 0

    def vwap(self) -> float | None:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else None


def _minute_floor(ts: datetime) -> datetime:
    """Floor a UTC timestamp to its minute start."""
    return ts.replace(second=0, microsecond=0, tzinfo=timezone.utc) \
        if ts.tzinfo is None else \
        ts.astimezone(timezone.utc).replace(second=0, microsecond=0)


class BarBuilder:
    """Maintains an in-progress bar per ticker. Emits bar via callback when minute rolls."""

    def __init__(self, on_bar_close: Callable[[Bar], None] | None = None) -> None:
        self._open: dict[str, Bar] = {}
        self.on_bar_close = on_bar_close

    def on_tick(self, ticker: str, ts: datetime, price: float, size: float) -> Bar | None:
        """Process one trade tick. Returns the CLOSED bar (if this tick crossed a minute), else None."""
        if size <= 0 or price <= 0:
            return None
        bar_ts = _minute_floor(ts)
        cur = self._open.get(ticker)
        closed: Bar | None = None
        if cur is not None and cur.minute_start_utc != bar_ts:
            closed = cur
            if self.on_bar_close:
                self.on_bar_close(closed)
            cur = None
        if cur is None:
            cur = Bar(
                ticker=ticker, minute_start_utc=bar_ts,
                open=price, high=price, low=price, close=price,
                volume=size, vwap_num=price * size, vwap_den=size, n_ticks=1,
            )
            self._open[ticker] = cur
            return closed
        cur.high = max(cur.high, price)
        cur.low = min(cur.low, price)
        cur.close = price
        cur.volume += size
        cur.vwap_num += price * size
        cur.vwap_den += size
        cur.n_ticks += 1
        return closed

    def flush(self, ticker: str | None = None) -> list[Bar]:
        """Force-close any in-progress bars (e.g. on stream end or session end)."""
        keys = [ticker] if ticker else list(self._open.keys())
        out: list[Bar] = []
        for k in keys:
            b = self._open.pop(k, None)
            if b is not None:
                if self.on_bar_close:
                    self.on_bar_close(b)
                out.append(b)
        return out

    def in_progress(self, ticker: str) -> Bar | None:
        return self._open.get(ticker)
