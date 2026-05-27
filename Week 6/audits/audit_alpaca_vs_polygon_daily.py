"""Audit: Alpaca daily bars vs Polygon cache (split/dividend adjustment check).

Backtest cache is from Polygon (5-min source, resampled). It is split-adjusted
because Polygon's adjusted endpoint is the standard. Live will pull bars from
Alpaca historical. If Alpaca's bars are NOT split-adjusted (or use a different
adjustment policy), close prices will JUMP on ex-split dates, causing:
  - Spurious Z-score moves (large fake mean-reversion signals)
  - Wrong notional sizing (price-based)
  - Discovery may reject the pair (spurious vol)

This audit pulls Alpaca's historical daily bars for the OVERLAPPING window
(2022-01 to 2026-03-19) for a handful of tickers and compares vs Polygon
cache. If they differ > 1% on any common date, flag.

Tickers chosen to cover known split events:
  - NVDA (10-for-1 split June 2024)
  - AAPL (4-for-1 split Aug 2020 — outside window but still good baseline)
  - TSLA (3-for-1 split Aug 2022)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
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

POLYGON_CACHE = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\daily_phase3")
TICKERS = ["NVDA", "TSLA", "AAPL", "GOOGL", "AMZN"]
# Cover known recent splits:
#   NVDA  10-for-1 2024-06-10
#   GOOGL 20-for-1 2022-07-18
#   AMZN  20-for-1 2022-06-06
#   TSLA  3-for-1  2022-08-25
COMPARE_WINDOW_START = "2022-06-01"
COMPARE_WINDOW_END = "2024-07-15"


def pull_alpaca_daily(client, ticker: str, start: str, end: str) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        feed="iex", adjustment="split",   # explicitly request split-adjusted
    )
    resp = client.get_stock_bars(req)
    bars = resp[ticker] if hasattr(resp, "__getitem__") else resp.data.get(ticker, [])
    rows = []
    for b in bars:
        rows.append({
            "date": b.timestamp.astimezone(timezone.utc).date(),
            "alpaca_close": float(b.close),
            "alpaca_volume": float(b.volume),
        })
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def load_polygon_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    fp = POLYGON_CACHE / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df = df[(df.index >= start) & (df.index <= end)]
    df["polygon_close"] = np.exp(df["log_close"])
    df["polygon_volume"] = df["volume"]
    df.index = df.index.date
    return df[["polygon_close", "polygon_volume"]]


def main() -> int:
    try:
        cfg = AlpacaConfig.from_env()
    except KeyError as e:
        print(f"ERROR: missing env var {e}", file=sys.stderr)
        return 1

    client = build_data_client(cfg)

    print(f"== Alpaca vs Polygon daily bars audit ==")
    print(f"  window: {COMPARE_WINDOW_START} -> {COMPARE_WINDOW_END}")
    print(f"  Alpaca adjustment policy: SPLIT-adjusted (explicit param)")
    print()

    issues = 0
    for tk in TICKERS:
        a = pull_alpaca_daily(client, tk, COMPARE_WINDOW_START, COMPARE_WINDOW_END)
        p = load_polygon_daily(tk, COMPARE_WINDOW_START, COMPARE_WINDOW_END)
        if a.empty:
            print(f"  {tk}: SKIP — Alpaca returned no data")
            continue
        if p.empty:
            print(f"  {tk}: SKIP — Polygon cache missing")
            continue
        joined = a.join(p, how="inner")
        if joined.empty:
            print(f"  {tk}: SKIP — no overlapping dates")
            continue
        joined["pct_diff"] = (joined["alpaca_close"] / joined["polygon_close"] - 1.0) * 100
        max_diff_pct = joined["pct_diff"].abs().max()
        median_diff_pct = joined["pct_diff"].abs().median()
        n_big = (joined["pct_diff"].abs() > 1.0).sum()
        print(f"  {tk}: n_overlap={len(joined)} max_diff={max_diff_pct:.3f}% "
              f"median={median_diff_pct:.4f}% n_diff>1%={n_big}")
        if max_diff_pct > 1.0:
            issues += 1
            joined["abs_diff_pct"] = joined["pct_diff"].abs()
            print(f"    >>> WORST DATES (by absolute %):")
            print(joined.nlargest(5, "abs_diff_pct")[
                ["alpaca_close", "polygon_close", "pct_diff"]
            ].to_string())

    print()
    if issues == 0:
        print("PASS: Alpaca bars match Polygon within 1% (split-adjusted equivalent).")
        return 0
    print(f"FAIL: {issues} tickers diverged >1% — Alpaca adjustment policy differs from Polygon.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
