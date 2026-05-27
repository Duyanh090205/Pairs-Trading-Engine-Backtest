"""Build split-adjusted daily cache from Alpaca for the live formation window.

Why this exists (per TODO 2 audit finding):
  Polygon cache in Week 4/data/validated/daily_phase3 contains RAW prices —
  major splits show 10-95% spurious jumps. Live engine using Polygon for
  discovery would inherit this bug. Solution: rebuild a fresh cache from
  Alpaca historical with explicit adjustment="split".

Output:
  live/state/daily_cache/<TICKER>.parquet — same schema as Polygon cache
  (index = tz-naive Timestamp at midnight; columns = log_close, volume).

Universe:
  All tickers in live/universe_top50.json (the live trading universe).
  Plus optional --extended flag to include all 528 backtest tickers for
  discovery purposes (so PCA factor model matches backtest scale).

Window: ~280 trading days (252 formation + 28 buffer for re-anchor + new bars).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from live.broker.alpaca_client import AlpacaConfig, build_data_client

OUT_DIR = ROOT / "live" / "state" / "daily_cache"
UNIVERSE_FILE = ROOT / "live" / "universe_top50.json"
POLYGON_REFERENCE = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\daily_phase3")

WINDOW_DAYS = 380   # ~252 trading days + buffer for weekends/holidays


def pull_daily_adjusted(client, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=TimeFrame.Day,
        start=start, end=end, feed="iex",
        adjustment="split",   # split-adjusted (NOT raw)
    )
    try:
        resp = client.get_stock_bars(req)
    except Exception as e:
        return pd.DataFrame()
    bars = resp[ticker] if hasattr(resp, "__getitem__") else resp.data.get(ticker, [])
    rows = []
    for b in bars:
        rows.append({
            "date": b.timestamp.astimezone(timezone.utc).date(),
            "close": float(b.close),
            "volume": float(b.volume),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])   # tz-naive Timestamp
    df = df.set_index("date").sort_index()
    # Convert to log_close (match Polygon cache schema)
    df["log_close"] = np.log(df["close"])
    return df[["log_close", "volume"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="top50",
                        help="'top50' (live universe) or 'extended' (all backtest tickers) "
                             "or path to JSON with 'tickers' list (vd. live/universe_top300.json)")
    parser.add_argument("--end-date", default=None,
                        help="YYYY-MM-DD inclusive; default = today UTC")
    parser.add_argument("--days", type=int, default=WINDOW_DAYS,
                        help=f"Window in calendar days (default {WINDOW_DAYS})")
    args = parser.parse_args()

    try:
        cfg = AlpacaConfig.from_env()
    except KeyError as e:
        print(f"ERROR: missing env var {e}", file=sys.stderr)
        return 1

    # Resolve universe → list of tickers.
    if args.universe == "top50":
        path = UNIVERSE_FILE
    elif args.universe == "extended":
        # Local: glob Polygon dir. Render: that path won't exist, so fall back
        # to committed universe_extended_528.json if present.
        if POLYGON_REFERENCE.exists():
            tickers = sorted(p.stem for p in POLYGON_REFERENCE.iterdir()
                              if p.suffix == ".parquet")
            path = None
        else:
            path = ROOT / "live" / "universe_extended_528.json"
            if not path.exists():
                print(f"ERROR: Polygon ref missing AND fallback {path} not present.",
                      file=sys.stderr)
                return 1
    else:
        # Treat --universe as a path to JSON {tickers: [...]}
        path = Path(args.universe)
        if not path.is_absolute():
            path = ROOT / args.universe
        if not path.exists():
            print(f"ERROR: universe file not found: {path}", file=sys.stderr)
            return 1
    if path is not None:
        tickers = json.loads(path.read_text())["tickers"]

    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.end_date:
        end = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)

    print(f"== Building Alpaca-adjusted daily cache ==")
    print(f"  Universe:  {args.universe} ({len(tickers)} tickers)")
    print(f"  Window:    {start.date()} -> {end.date()} ({args.days} days)")
    print(f"  Output:    {OUT_DIR}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = build_data_client(cfg)

    t0 = time.time()
    ok = 0
    empty = 0
    failed = 0
    for i, tk in enumerate(tickers, 1):
        try:
            df = pull_daily_adjusted(client, tk, start, end)
        except Exception as e:
            print(f"  [{i:3d}/{len(tickers)}] {tk}: FAIL {type(e).__name__}: {e}")
            failed += 1
            continue
        if df.empty:
            print(f"  [{i:3d}/{len(tickers)}] {tk}: empty (delisted or not in Alpaca IEX)")
            empty += 1
            continue
        df.to_parquet(OUT_DIR / f"{tk}.parquet")
        ok += 1
        if i % 10 == 0 or i == len(tickers):
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(tickers)}] ... ok={ok} empty={empty} failed={failed} "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {ok} tickers ({OUT_DIR})")
    print(f"  Empty: {empty}")
    print(f"  Failed: {failed}")

    # Sanity check on one ticker
    sample_tk = "AAPL" if (OUT_DIR / "AAPL.parquet").exists() else (tickers[0] if tickers else None)
    if sample_tk and (OUT_DIR / f"{sample_tk}.parquet").exists():
        df_sample = pd.read_parquet(OUT_DIR / f"{sample_tk}.parquet")
        print(f"\nSanity ({sample_tk}):")
        print(f"  date range: {df_sample.index.min().date()} -> {df_sample.index.max().date()}")
        print(f"  bars: {len(df_sample)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
