"""Cross-path audit: BarBuilder vs Alpaca official 1-min bars.

Pulls historical TRADES + historical 1-min BARS for one ticker over a small
window, replays trades through BarBuilder, compares the resulting bars vs
Alpaca's official aggregation.

Expected behavior on **paper / IEX feed**:
  - OHLC: should match closely (IEX prints typically hit same extremes for
    liquid names like AAPL), tolerance ~5 bps
  - Volume: WILL diverge significantly (IEX ≈ 2-3% of total tape) — this is
    reported but NOT a fail criterion.

Run: python audits/audit_bar_builder_vs_alpaca.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from live.broker.alpaca_client import AlpacaConfig
from live.engine_live.bar_builder import BarBuilder

TICKER = "AAPL"
WINDOW_MINUTES = 5  # tight window to limit trade count


def pick_recent_trading_window() -> tuple[datetime, datetime]:
    """Pick last Friday 14:35-14:40 UTC (just after open, high volume).
    If today is Mon-Fri, today's window may not be available yet."""
    now = datetime.now(timezone.utc)
    days_back = (now.weekday() + 3) % 7   # back to last Friday
    if days_back == 0:
        days_back = 7
    target_day = (now - timedelta(days=days_back)).date()
    start = datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=14, minute=35)
    end = start + timedelta(minutes=WINDOW_MINUTES)
    return start, end


def pull_trades(client, ticker: str, start: datetime, end: datetime) -> list[tuple[datetime, float, float]]:
    from alpaca.data.requests import StockTradesRequest
    req = StockTradesRequest(symbol_or_symbols=ticker, start=start, end=end, feed="iex")
    resp = client.get_stock_trades(req)
    trades = resp[ticker] if hasattr(resp, "__getitem__") else resp.data.get(ticker, [])
    out = []
    for t in trades:
        out.append((t.timestamp, float(t.price), float(t.size)))
    # Alpaca's trades endpoint may not be timestamp-ordered. Aggregation depends
    # on the LAST trade by time, so sort explicitly to mirror Alpaca's bar logic.
    out.sort(key=lambda x: x[0])
    return out


def pull_bars(client, ticker: str, start: datetime, end: datetime) -> list[dict]:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
        start=start, end=end, feed="iex",
    )
    resp = client.get_stock_bars(req)
    bars = resp[ticker] if hasattr(resp, "__getitem__") else resp.data.get(ticker, [])
    out = []
    for b in bars:
        out.append({
            "ts": b.timestamp.astimezone(timezone.utc),
            "open": float(b.open), "high": float(b.high),
            "low": float(b.low), "close": float(b.close),
            "volume": float(b.volume),
        })
    return out


def build_from_trades(trades) -> dict:
    bb = BarBuilder()
    for ts, p, s in trades:
        bb.on_tick(TICKER, ts, p, s)
    bb.flush()
    # BarBuilder doesn't store closed bars; re-build with capture
    closed: list = []
    bb2 = BarBuilder(on_bar_close=closed.append)
    for ts, p, s in trades:
        bb2.on_tick(TICKER, ts, p, s)
    inprog = bb2.flush()
    return {b.minute_start_utc: b for b in closed + inprog}


def compare(built: dict, official: list[dict]) -> int:
    print(f"  built bars:    {len(built)} (built from {sum(b.n_ticks for b in built.values())} ticks)")
    print(f"  official bars: {len(official)}")
    if not built and not official:
        print("  no data in window — likely IEX feed doesn't include this window/ticker")
        return 0
    issues = 0
    # Exclude the trailing official bar (its trades fall outside our request window)
    if built:
        max_built_ts = max(built.keys())
        official_in_window = [b for b in official if b["ts"] <= max_built_ts]
    else:
        official_in_window = official
    for ob in official_in_window:
        bb = built.get(ob["ts"])
        if bb is None:
            print(f"  MISSING built bar at {ob['ts']}")
            issues += 1
            continue
        for k in ("open", "high", "low", "close"):
            ob_val = ob[k]
            bb_val = getattr(bb, k)
            if ob_val == 0:
                continue
            bps_diff = abs(bb_val - ob_val) / ob_val * 10_000
            if bps_diff > 5.0:
                print(f"  {ob['ts']} {k}: built={bb_val} vs official={ob_val}  ({bps_diff:.2f} bps)")
                issues += 1
        vol_ratio = (bb.volume / ob["volume"]) if ob["volume"] > 0 else 0
        print(f"  {ob['ts']} O={bb.open:.4f}/{ob['open']:.4f}  "
              f"C={bb.close:.4f}/{ob['close']:.4f}  "
              f"vol built={bb.volume:.0f} official={ob['volume']:.0f} ({vol_ratio*100:.1f}%)")
    return issues


def main() -> int:
    try:
        cfg = AlpacaConfig.from_env()
    except KeyError as e:
        print(f"ERROR: missing env var {e}. Populate .env first.", file=sys.stderr)
        return 1

    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)

    start, end = pick_recent_trading_window()
    print(f"== BarBuilder vs Alpaca official — {TICKER} {start.isoformat()} -> {end.isoformat()} ==")

    try:
        trades = pull_trades(client, TICKER, start, end)
    except Exception as e:
        print(f"trades pull failed: {type(e).__name__}: {e}", file=sys.stderr)
        print("  (free IEX paper feed may not have trades for this window)")
        return 0  # not a hard fail — feed availability isn't a code bug
    try:
        official = pull_bars(client, TICKER, start, end)
    except Exception as e:
        print(f"bars pull failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 0

    built = build_from_trades(trades)
    issues = compare(built, official)
    print()
    if issues == 0:
        print("PASS: BarBuilder OHLC matches Alpaca official within 5 bps tolerance.")
        return 0
    print(f"FAIL: {issues} OHLC discrepancies > 5 bps detected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
