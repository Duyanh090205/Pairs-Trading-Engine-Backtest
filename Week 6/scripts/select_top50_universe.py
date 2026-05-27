"""Select top-50 most-liquid tickers from V4 backtest universe for live paper trading.

Liquidity score = mean(price * volume) over last `LIQ_WINDOW_DAYS`, where
price = exp(log_close). This is a Dollar-Volume proxy. We require each ticker
to have at least `MIN_BARS_RECENT` bars in the last 30 days (filters out
delisted / halted names).

Output:
  live/universe_top50.json — list of tickers + per-ticker liquidity stats
  live/universe_top50.csv  — same data, spreadsheet-friendly

Usage:
  python scripts/select_top50_universe.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

DAILY_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\daily_phase3")
WEEK6 = Path(__file__).resolve().parents[1]
LIQ_WINDOW_DAYS = 252        # one trading year for stability
MIN_BARS_RECENT = 15         # require ≥15 bars in last 30 calendar days (drop delisted)
RECENT_DAYS = 30
TOP_N = int(os.environ.get("TOP_N", "50"))

OUT_JSON = WEEK6 / "live" / f"universe_top{TOP_N}.json"
OUT_CSV = WEEK6 / "live" / f"universe_top{TOP_N}.csv"


def main() -> int:
    if not DAILY_DIR.exists():
        print(f"ERROR: daily cache not found at {DAILY_DIR}", file=sys.stderr)
        return 1

    files = sorted(p for p in DAILY_DIR.iterdir() if p.suffix == ".parquet")
    print(f"Scanning {len(files)} ticker parquets in {DAILY_DIR.name}/")

    rows: list[dict] = []
    dropped_delisted = 0
    dropped_short = 0

    for fp in files:
        ticker = fp.stem
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"  skip {ticker}: read error {e}")
            continue

        if "log_close" not in df.columns or "volume" not in df.columns:
            continue
        if len(df) < LIQ_WINDOW_DAYS:
            dropped_short += 1
            continue

        cutoff_recent = df.index.max() - pd.Timedelta(days=RECENT_DAYS)
        bars_recent = (df.index >= cutoff_recent).sum()
        if bars_recent < MIN_BARS_RECENT:
            dropped_delisted += 1
            continue

        tail = df.tail(LIQ_WINDOW_DAYS)
        price = np.exp(tail["log_close"].to_numpy())
        vol = tail["volume"].to_numpy()
        dollar_vol = price * vol
        dv_mask = np.isfinite(dollar_vol) & (dollar_vol > 0)
        if dv_mask.sum() < 60:
            continue

        rows.append({
            "ticker": ticker,
            "avg_dollar_volume_usd": float(np.mean(dollar_vol[dv_mask])),
            "median_dollar_volume_usd": float(np.median(dollar_vol[dv_mask])),
            "last_price_usd": float(price[-1]),
            "last_date": str(df.index.max().date()),
            "bars_total": int(len(df)),
        })

    if not rows:
        print("ERROR: no eligible tickers", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows).sort_values(
        "avg_dollar_volume_usd", ascending=False
    ).reset_index(drop=True)

    top = out.head(TOP_N).copy()
    top["rank"] = top.index + 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe_size": len(top),
        "selection_date": pd.Timestamp.utcnow().isoformat(),
        "liquidity_window_days": LIQ_WINDOW_DAYS,
        "tickers": top["ticker"].tolist(),
        "stats": top.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    top.to_csv(OUT_CSV, index=False)

    print()
    print(f"Eligible: {len(out)} / {len(files)} tickers")
    print(f"  dropped (history <{LIQ_WINDOW_DAYS} bars): {dropped_short}")
    print(f"  dropped (likely delisted, no recent bars):  {dropped_delisted}")
    print()
    print(f"Top-{TOP_N} written to:")
    print(f"  {OUT_JSON.relative_to(WEEK6)}")
    print(f"  {OUT_CSV.relative_to(WEEK6)}")
    print()
    print(f"Top 10 by avg $ volume (last {LIQ_WINDOW_DAYS} bars):")
    for _, r in top.head(10).iterrows():
        adv_b = r["avg_dollar_volume_usd"] / 1e9
        print(f"  {int(r['rank']):3d}. {r['ticker']:6s}  ${adv_b:6.2f}B/day  "
              f"px=${r['last_price_usd']:.2f}  last={r['last_date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
