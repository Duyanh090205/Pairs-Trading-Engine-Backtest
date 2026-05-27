"""Alpaca WebSocket: subscribe to bars + trades, reconnect with exp backoff,
expose connection state + per-ticker freshness so the engine can act on stale data.

Design:
  - Subscribe to BARS (Alpaca server-side aggregation) → fed to engine
  - Subscribe to TRADES → fed to BarBuilder cross-check (data-quality safety layer)
  - On bar minute-close, compare built-from-trades vs server-aggregated; emit
    DataQualityEvent if divergence > 5 bps for 3 consecutive bars.
  - is_stale_for(ticker) is per-ticker. is_stream_dead() = ALL tickers stale.
    Engine queries both; partial-stale doesn't kill the engine, only affects the
    tickers that are quiet.
  - connection_state property survives reconnect loop — engine polls it.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from loguru import logger


class ConnState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


@dataclass
class StreamConfig:
    backoff_initial_s: float = 1.0
    backoff_cap_s: float = 60.0
    stale_threshold_s: float = 90.0
    feed: str = "iex"
    enable_trades_crosscheck: bool = True
    crosscheck_bps_threshold: float = 5.0
    crosscheck_consecutive_fail_max: int = 3


@dataclass
class DataQualityEvent:
    ticker: str
    bar_ts: datetime
    server_close: float
    built_close: float
    bps_diff: float
    consecutive_fails: int


class AlpacaStream:
    """Streams bars+trades with reconnect, per-ticker stale tracking, and
    optional BarBuilder cross-check."""

    def __init__(self, cfg, tickers: list[str], stream_cfg: StreamConfig | None = None,
                 on_data_quality: Callable[[DataQualityEvent], None] | None = None) -> None:
        self.cfg = cfg
        self.tickers = tickers
        self.stream_cfg = stream_cfg or StreamConfig()
        self._last_bar_ts: dict[str, float] = {}      # ticker → wall-clock epoch of last bar
        self._stop = False
        self._state = ConnState.DISCONNECTED
        self._last_state_change = time.time()
        self.on_data_quality = on_data_quality
        self._crosscheck_fails: dict[str, int] = defaultdict(int)
        # In-memory partial bars (built from trades), keyed by (ticker, minute_start)
        self._builder_partial: dict[tuple[str, datetime], dict] = {}

    @property
    def state(self) -> ConnState:
        return self._state

    def _set_state(self, s: ConnState) -> None:
        if s != self._state:
            self._state = s
            self._last_state_change = time.time()
            logger.info(f"alpaca stream state -> {s.value}")

    def state_age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self._last_state_change

    async def run(self, on_bar: Callable[[dict], Awaitable[None]]) -> None:
        backoff = self.stream_cfg.backoff_initial_s
        attempt = 0
        while not self._stop:
            attempt += 1
            self._set_state(ConnState.CONNECTING if attempt == 1 else ConnState.RECONNECTING)
            try:
                logger.info(f"alpaca stream connecting (attempt {attempt}, feed={self.stream_cfg.feed})")
                self._set_state(ConnState.CONNECTED)
                await self._consume_one_session(on_bar)
                if self._stop:
                    break
                logger.warning("stream ended without exception; reconnecting")
            except asyncio.CancelledError:
                self._set_state(ConnState.STOPPED)
                raise
            except Exception as e:
                logger.error(f"stream error: {type(e).__name__}: {e}")
            self._set_state(ConnState.DISCONNECTED)
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.stream_cfg.backoff_cap_s)
        self._set_state(ConnState.STOPPED)

    async def _consume_one_session(self, on_bar: Callable[[dict], Awaitable[None]]) -> None:
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream
        # alpaca-py expects DataFeed enum, not str. Map string config -> enum.
        feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
        feed = feed_map.get(self.stream_cfg.feed.lower(), DataFeed.IEX)
        stream = StockDataStream(self.cfg.api_key, self.cfg.secret_key, feed=feed)

        async def _bar_handler(bar) -> None:
            self._last_bar_ts[getattr(bar, "symbol", None) or bar["symbol"]] = time.time()
            payload = self._bar_to_dict(bar)
            self._evaluate_crosscheck(payload)
            await on_bar(payload)

        async def _trade_handler(tr) -> None:
            if not self.stream_cfg.enable_trades_crosscheck:
                return
            self._update_partial_from_trade(tr)

        stream.subscribe_bars(_bar_handler, *self.tickers)
        if self.stream_cfg.enable_trades_crosscheck:
            stream.subscribe_trades(_trade_handler, *self.tickers)
        try:
            await stream._run_forever()
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    def _update_partial_from_trade(self, tr) -> None:
        sym = getattr(tr, "symbol", None)
        price = getattr(tr, "price", None)
        size = getattr(tr, "size", None)
        ts = getattr(tr, "timestamp", None)
        if not (sym and price and size and ts):
            return
        minute = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        key = (sym, minute)
        p = self._builder_partial.get(key)
        if p is None:
            self._builder_partial[key] = {
                "open": float(price), "high": float(price), "low": float(price),
                "close": float(price), "volume": float(size), "n": 1,
            }
        else:
            p["high"] = max(p["high"], float(price))
            p["low"] = min(p["low"], float(price))
            p["close"] = float(price)
            p["volume"] += float(size)
            p["n"] += 1

    def _evaluate_crosscheck(self, server_bar: dict) -> None:
        """When an official bar arrives, compare it to our trades-built bar for that minute."""
        if not self.stream_cfg.enable_trades_crosscheck:
            return
        ticker = server_bar["ticker"]
        try:
            bar_ts = datetime.fromisoformat(server_bar["ts_utc"].replace("Z", "+00:00"))
        except Exception:
            return
        minute = bar_ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        built = self._builder_partial.pop((ticker, minute), None)
        if built is None:
            return
        srv_close = server_bar["close"]
        bps = abs(built["close"] - srv_close) / srv_close * 10_000 if srv_close > 0 else 0.0
        if bps > self.stream_cfg.crosscheck_bps_threshold:
            self._crosscheck_fails[ticker] += 1
        else:
            self._crosscheck_fails[ticker] = 0
        if (self._crosscheck_fails[ticker] >= self.stream_cfg.crosscheck_consecutive_fail_max
                and self.on_data_quality is not None):
            self.on_data_quality(DataQualityEvent(
                ticker=ticker, bar_ts=minute,
                server_close=srv_close, built_close=built["close"],
                bps_diff=bps, consecutive_fails=self._crosscheck_fails[ticker],
            ))

    @staticmethod
    def _bar_to_dict(bar) -> dict:
        def _g(name):
            return getattr(bar, name, None) if not isinstance(bar, dict) else bar.get(name)
        ts = _g("timestamp")
        if isinstance(ts, datetime):
            ts_iso = ts.astimezone(timezone.utc).isoformat()
        else:
            ts_iso = str(ts)
        return {
            "ticker": _g("symbol"),
            "ts_utc": ts_iso,
            "open": float(_g("open")),
            "high": float(_g("high")),
            "low": float(_g("low")),
            "close": float(_g("close")),
            "volume": float(_g("volume") or 0.0),
            "vwap": float(_g("vwap") or 0.0),
            "n_trades": int(_g("trade_count") or 0),
        }

    # ---------- Freshness API ----------

    def is_stale_for(self, ticker: str, now_epoch_s: float | None = None) -> bool:
        """Per-ticker freshness check. Used by engine to skip individual ticker decisions."""
        ts = self._last_bar_ts.get(ticker)
        if ts is None:
            return False  # never received a bar — not 'stale', just unseen
        now = now_epoch_s if now_epoch_s is not None else time.time()
        return (now - ts) > self.stream_cfg.stale_threshold_s

    def stale_tickers(self, now_epoch_s: float | None = None) -> list[str]:
        now = now_epoch_s if now_epoch_s is not None else time.time()
        return [t for t in self._last_bar_ts
                if (now - self._last_bar_ts[t]) > self.stream_cfg.stale_threshold_s]

    def is_stream_dead(self, now_epoch_s: float | None = None) -> bool:
        """True iff ALL subscribed tickers are stale. Engine should halt entries."""
        if not self._last_bar_ts:
            return False
        now = now_epoch_s if now_epoch_s is not None else time.time()
        return all((now - ts) > self.stream_cfg.stale_threshold_s
                   for ts in self._last_bar_ts.values())

    def last_bar_age_s(self, ticker: str, now_epoch_s: float | None = None) -> float | None:
        if ticker not in self._last_bar_ts:
            return None
        now = now_epoch_s if now_epoch_s is not None else time.time()
        return now - self._last_bar_ts[ticker]

    def heartbeat(self, now_epoch_s: float | None = None) -> dict:
        """Engine polls this every few seconds — single source of truth for stream health."""
        now = now_epoch_s if now_epoch_s is not None else time.time()
        return {
            "state": self._state.value,
            "state_age_s": now - self._last_state_change,
            "tickers_seen": len(self._last_bar_ts),
            "stale_count": len(self.stale_tickers(now)),
            "stream_dead": self.is_stream_dead(now),
        }

    def stop(self) -> None:
        self._stop = True


def compute_backoff_schedule(initial: float = 1.0, cap: float = 60.0,
                              n: int = 10) -> list[float]:
    out = []
    b = initial
    for _ in range(n):
        out.append(b)
        b = min(b * 2, cap)
    return out
