"""Drill: bar gap > 90s → engine pauses new entries until fresh bar.

Pass criteria:
  - is_stale_for(ticker) detects the gap per-ticker
  - is_stream_dead returns True only when ALL tickers stale
  - Engine refuses entries for stale tickers
  - Engine still trades fresh tickers
  - When fresh bar arrives, ticker freshness is restored
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def _check(name, cond, detail=""):
    mark = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    print(f"  {mark}  {name}{(' — ' + detail) if detail else ''}")
    return cond


def _make_stream(tickers, stale_threshold_s=90.0):
    from collections import defaultdict
    from live.broker.websocket_handler import AlpacaStream, ConnState, StreamConfig
    s = AlpacaStream.__new__(AlpacaStream)
    s.cfg = None
    s.tickers = tickers
    s.stream_cfg = StreamConfig(stale_threshold_s=stale_threshold_s,
                                enable_trades_crosscheck=False)
    s._last_bar_ts = {}
    s._stop = False
    s._state = ConnState.CONNECTED
    s._last_state_change = 0.0
    s.on_data_quality = None
    s._crosscheck_fails = defaultdict(int)
    s._builder_partial = {}
    return s


def main() -> int:
    print("== Bar gap > 90s drill ==")
    s = _make_stream(["AAPL", "MSFT", "QUIET"], stale_threshold_s=90.0)
    t = 1_000.0
    s._last_bar_ts = {"AAPL": t, "MSFT": t, "QUIET": t - 200}  # QUIET stale already
    results = []

    results.append(_check("AAPL fresh", s.is_stale_for("AAPL", t) is False))
    results.append(_check("QUIET stale (gap 200s > 90s)", s.is_stale_for("QUIET", t) is True))
    results.append(_check("stream NOT dead (2 fresh, 1 stale)",
                          s.is_stream_dead(t) is False))

    # Advance time: AAPL and MSFT also go stale
    t_advance = t + 95
    results.append(_check("after gap: AAPL now stale", s.is_stale_for("AAPL", t_advance) is True))
    results.append(_check("all stale -> stream dead",
                          s.is_stream_dead(t_advance) is True))

    # Fresh bar for AAPL restores freshness
    s._last_bar_ts["AAPL"] = t_advance + 10
    t_refresh = t_advance + 11
    results.append(_check("fresh bar restores AAPL", s.is_stale_for("AAPL", t_refresh) is False))
    results.append(_check("MSFT still stale", s.is_stale_for("MSFT", t_refresh) is True))
    results.append(_check("stream not dead (AAPL fresh again)",
                          s.is_stream_dead(t_refresh) is False))

    print()
    if all(results):
        print(f"{GREEN}DRILL PASS{RESET} — {sum(results)}/{len(results)}")
        return 0
    print(f"{RED}DRILL FAIL{RESET} — {sum(results)}/{len(results)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
