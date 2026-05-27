"""Phase 1 Day 2 smoketest — bar builder + WebSocket scaffolding. No network needed.

Checks:
  1. BarBuilder produces expected OHLCV from a synthetic tick sequence
  2. BarBuilder closes bar exactly at minute boundary (first tick of next minute)
  3. BarBuilder rejects bad ticks (size<=0, price<=0)
  4. BarBuilder.flush emits the in-progress bar (e.g. session end)
  5. compute_backoff_schedule produces 1,2,4,8,16,32,60,60,... (capped)
  6. AlpacaStream.is_stale returns False before any bar, True past threshold
  7. _bar_to_dict normalizes both attr-style and dict-style payloads
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
failures: list[str] = []


def check(name, fn):
    try:
        fn()
        print(f"{GREEN}PASS{RESET}  {name}")
    except Exception as e:
        print(f"{RED}FAIL{RESET}  {name}: {type(e).__name__}: {e}")
        failures.append(name)


def t_bar_builder_ohlcv():
    from live.engine_live.bar_builder import BarBuilder
    bb = BarBuilder()
    base = datetime(2026, 5, 25, 13, 30, 0, tzinfo=timezone.utc)
    # 5 ticks in minute 13:30
    ticks = [
        (base + timedelta(seconds=1), 100.0, 10),
        (base + timedelta(seconds=15), 101.5, 5),
        (base + timedelta(seconds=30), 99.0, 8),
        (base + timedelta(seconds=45), 100.5, 4),
        (base + timedelta(seconds=59), 100.25, 2),
    ]
    for ts, p, s in ticks:
        bb.on_tick("AAPL", ts, p, s)
    bar = bb.in_progress("AAPL")
    assert bar.open == 100.0, f"open {bar.open}"
    assert bar.high == 101.5, f"high {bar.high}"
    assert bar.low == 99.0, f"low {bar.low}"
    assert bar.close == 100.25, f"close {bar.close}"
    assert bar.volume == 29, f"vol {bar.volume}"
    assert bar.n_ticks == 5


def t_bar_builder_minute_close():
    from live.engine_live.bar_builder import BarBuilder
    closed_bars = []
    bb = BarBuilder(on_bar_close=closed_bars.append)
    base = datetime(2026, 5, 25, 13, 30, 0, tzinfo=timezone.utc)
    bb.on_tick("AAPL", base + timedelta(seconds=5), 100.0, 10)
    bb.on_tick("AAPL", base + timedelta(seconds=30), 101.0, 5)
    # Cross minute boundary
    closed = bb.on_tick("AAPL", base + timedelta(seconds=61), 102.0, 7)
    assert closed is not None and closed.close == 101.0, "minute close should emit prev bar"
    assert closed.minute_start_utc == base, "closed bar should be the 13:30 minute"
    assert len(closed_bars) == 1
    # New bar started at 13:31
    new_bar = bb.in_progress("AAPL")
    assert new_bar.minute_start_utc == base + timedelta(minutes=1)
    assert new_bar.open == 102.0


def t_bar_builder_bad_ticks():
    from live.engine_live.bar_builder import BarBuilder
    bb = BarBuilder()
    ts = datetime(2026, 5, 25, 13, 30, 0, tzinfo=timezone.utc)
    assert bb.on_tick("AAPL", ts, 100.0, 0) is None
    assert bb.on_tick("AAPL", ts, -1.0, 5) is None
    assert bb.in_progress("AAPL") is None  # nothing was recorded


def t_bar_builder_flush():
    from live.engine_live.bar_builder import BarBuilder
    bb = BarBuilder()
    ts = datetime(2026, 5, 25, 13, 30, 0, tzinfo=timezone.utc)
    bb.on_tick("AAPL", ts, 100.0, 10)
    bb.on_tick("MSFT", ts, 400.0, 3)
    out = bb.flush()
    assert len(out) == 2
    assert bb.in_progress("AAPL") is None
    assert bb.in_progress("MSFT") is None


def t_backoff_schedule():
    from live.broker.websocket_handler import compute_backoff_schedule
    seq = compute_backoff_schedule(initial=1.0, cap=60.0, n=10)
    assert seq[0] == 1.0
    assert seq[1] == 2.0
    assert seq[2] == 4.0
    assert seq[3] == 8.0
    assert seq[4] == 16.0
    assert seq[5] == 32.0
    assert seq[6] == 60.0   # 64 would exceed cap
    assert seq[7] == 60.0   # capped
    assert all(x <= 60.0 for x in seq)


def _make_stream(tickers, stale_threshold_s=90.0):
    from live.broker.websocket_handler import AlpacaStream, ConnState, StreamConfig
    s = AlpacaStream.__new__(AlpacaStream)
    s.cfg = None
    s.tickers = tickers
    s.stream_cfg = StreamConfig(stale_threshold_s=stale_threshold_s)
    s._last_bar_ts = {}
    s._stop = False
    s._state = ConnState.DISCONNECTED
    s._last_state_change = 0.0
    s.on_data_quality = None
    from collections import defaultdict
    s._crosscheck_fails = defaultdict(int)
    s._builder_partial = {}
    return s


def t_is_stale_per_ticker():
    """Fix Bug 2: is_stale_for is per-ticker; stream_dead requires ALL stale."""
    s = _make_stream(["AAPL", "MSFT", "QUIET"])
    # No bars yet → not stale (just unseen)
    assert s.is_stale_for("AAPL", now_epoch_s=1000.0) is False
    assert s.is_stream_dead(now_epoch_s=1000.0) is False
    # AAPL active, MSFT active, QUIET silent for 5 min
    s._last_bar_ts["AAPL"] = 1000.0
    s._last_bar_ts["MSFT"] = 1000.0
    s._last_bar_ts["QUIET"] = 700.0   # silent 300s ago
    # Globally NOT dead (AAPL+MSFT fresh), but QUIET is per-ticker stale
    assert s.is_stream_dead(now_epoch_s=1010.0) is False, "two active tickers → not dead"
    assert s.is_stale_for("AAPL", now_epoch_s=1010.0) is False
    assert s.is_stale_for("QUIET", now_epoch_s=1010.0) is True
    assert s.stale_tickers(now_epoch_s=1010.0) == ["QUIET"]
    # All silent ≥ 91s → stream_dead
    assert s.is_stream_dead(now_epoch_s=1095.0) is True


def t_heartbeat_exposes_state():
    """Fix Bug 3: engine can poll heartbeat to detect reconnect/stale conditions."""
    from live.broker.websocket_handler import ConnState
    s = _make_stream(["AAPL"])
    s._state = ConnState.RECONNECTING
    s._last_state_change = 1000.0
    s._last_bar_ts["AAPL"] = 1000.0
    hb = s.heartbeat(now_epoch_s=1010.0)
    assert hb["state"] == "reconnecting"
    assert hb["state_age_s"] == 10.0
    assert hb["tickers_seen"] == 1
    assert hb["stale_count"] == 0
    # Push time past stale threshold
    hb2 = s.heartbeat(now_epoch_s=1100.0)
    assert hb2["stale_count"] == 1
    assert hb2["stream_dead"] is True


def t_crosscheck_emits_event():
    """Fix Bug 1: BarBuilder cross-check via trades emits DataQualityEvent on divergence."""
    from datetime import datetime, timezone
    events: list = []
    s = _make_stream(["AAPL"])
    s.on_data_quality = events.append
    s.stream_cfg.crosscheck_consecutive_fail_max = 2
    s.stream_cfg.crosscheck_bps_threshold = 5.0

    minute = datetime(2026, 5, 25, 14, 30, 0, tzinfo=timezone.utc)

    # Simulate 3 trade-built bars at $100 close
    for i in range(3):
        m = minute.replace(minute=30 + i)
        s._builder_partial[("AAPL", m)] = {
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 1000.0, "n": 10,
        }
    # Server-side bars at $100.20 close (~20 bps off → trigger fail)
    for i in range(3):
        m = minute.replace(minute=30 + i)
        s._evaluate_crosscheck({
            "ticker": "AAPL", "ts_utc": m.isoformat(),
            "open": 100.20, "high": 100.20, "low": 100.20, "close": 100.20,
            "volume": 1000.0, "vwap": 100.20, "n_trades": 10,
        })
    # First fail bumps to 1, second to 2 → triggers event at consecutive_max=2
    assert len(events) >= 1, f"expected DataQualityEvent emitted, got {len(events)}"
    ev = events[0]
    assert ev.ticker == "AAPL"
    assert ev.bps_diff > 5.0
    assert ev.consecutive_fails >= 2


def t_bar_to_dict_normalization():
    from live.broker.websocket_handler import AlpacaStream
    from types import SimpleNamespace
    ts = datetime(2026, 5, 25, 13, 30, 0, tzinfo=timezone.utc)
    attr_bar = SimpleNamespace(
        symbol="AAPL", timestamp=ts,
        open=100.0, high=101.0, low=99.0, close=100.5,
        volume=12345, vwap=100.3, trade_count=42,
    )
    d = AlpacaStream._bar_to_dict(attr_bar)
    assert d["ticker"] == "AAPL"
    assert d["open"] == 100.0
    assert d["volume"] == 12345.0
    assert d["n_trades"] == 42
    # Dict-style payload (some alpaca-py paths pass dicts)
    dict_bar = {
        "symbol": "MSFT", "timestamp": ts.isoformat(),
        "open": 400.0, "high": 401.0, "low": 399.0, "close": 400.5,
        "volume": 1000, "vwap": 400.2, "trade_count": 10,
    }
    d2 = AlpacaStream._bar_to_dict(dict_bar)
    assert d2["ticker"] == "MSFT"
    assert d2["close"] == 400.5


def main() -> int:
    print(f"{YELLOW}== Phase 1 Day 2 Smoketest =={RESET}")
    check("bar_builder_ohlcv", t_bar_builder_ohlcv)
    check("bar_builder_minute_close", t_bar_builder_minute_close)
    check("bar_builder_bad_ticks", t_bar_builder_bad_ticks)
    check("bar_builder_flush", t_bar_builder_flush)
    check("backoff_schedule", t_backoff_schedule)
    check("is_stale_per_ticker (fix Bug 2)", t_is_stale_per_ticker)
    check("heartbeat_exposes_state (fix Bug 3)", t_heartbeat_exposes_state)
    check("crosscheck_emits_event (fix Bug 1)", t_crosscheck_emits_event)
    check("bar_to_dict_normalization", t_bar_to_dict_normalization)
    if failures:
        print(f"{RED}{len(failures)} FAILED:{RESET} {failures}")
        return 1
    print(f"{GREEN}All Day 2 smoketests passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
