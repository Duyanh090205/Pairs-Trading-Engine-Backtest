"""
Build daily spread cache from Week 5 1-min microstructure data.

Source: Week 5/data/microstructure/spreads_1min.parquet
  - 7.7 GB, 195M rows, 526 row groups (one per ticker likely)
  - columns: timestamp_et, ticker, is_valid, half_spread_l1_bps, spread_std_1d, ...

Output: Week 6/cost/daily_spread_cache.parquet
  - One row per (ticker, date)
  - columns: ticker, date, half_spread_bps, spread_std_1d
  - ~30 MB

Aggregation: for each (ticker, date) take the MEDIAN of intraday 1-min bars,
restricted to is_valid=True. Median is robust to spikes/wide quotes from
the 09:30 and 15:59 ET edges.

Runs in ~15-30 min (streaming row group by row group; ~14 MB per group).
"""

from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ[_v] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WEEK6 = Path(__file__).resolve().parents[1]
SRC = Path(r"d:\Quant Finance\Quant Program\Week 5\data\microstructure\spreads_1min.parquet")
DST = WEEK6 / "cost" / "daily_spread_cache.parquet"

READ_COLS = ["timestamp_et", "ticker", "is_valid", "half_spread_l1_bps", "spread_std_1d"]


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1
    DST.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source: {SRC}  ({SRC.stat().st_size / 1024**3:.2f} GB)")
    print(f"Dest:   {DST}\n")

    pf = pq.ParquetFile(SRC)
    n_groups = pf.metadata.num_row_groups
    n_rows_total = pf.metadata.num_rows
    print(f"  parquet: {n_groups} row groups, {n_rows_total:,} rows total\n")

    t0 = time.time()
    daily_rows: list[pd.DataFrame] = []
    n_rows_seen = 0
    n_rows_kept = 0

    for gi in range(n_groups):
        rg = pf.read_row_group(gi, columns=READ_COLS).to_pandas()
        n_rows_seen += len(rg)
        # Keep only valid quotes
        rg = rg[rg["is_valid"] == True]
        if len(rg) == 0:
            continue
        n_rows_kept += len(rg)

        # Build date column (tz-naive midnight)
        rg["date"] = pd.to_datetime(rg["timestamp_et"]).dt.tz_convert("US/Eastern").dt.normalize().dt.tz_localize(None)

        # Aggregate per (ticker, date): median of intraday 1-min spreads + std
        agg = rg.groupby(["ticker", "date"], sort=False).agg(
            half_spread_bps=("half_spread_l1_bps", "median"),
            spread_std_1d=("spread_std_1d", "median"),
            n_bars=("half_spread_l1_bps", "size"),
        ).reset_index()
        daily_rows.append(agg)

        if (gi + 1) % 50 == 0 or gi == n_groups - 1:
            elapsed = time.time() - t0
            rate = n_rows_seen / max(elapsed, 1e-6)
            print(f"  row group {gi+1:>3d}/{n_groups} | "
                  f"seen={n_rows_seen:>11,} valid={n_rows_kept:>11,} | "
                  f"{elapsed:.0f}s ({rate:,.0f} rows/s)")

    print(f"\nAggregating across row groups ({len(daily_rows)} chunks)...")
    daily = pd.concat(daily_rows, ignore_index=True)
    # If a ticker appears across multiple row groups for the same date, merge
    daily = (
        daily.groupby(["ticker", "date"], as_index=False)
        .agg(
            half_spread_bps=("half_spread_bps", "median"),
            spread_std_1d=("spread_std_1d", "median"),
            n_bars=("n_bars", "sum"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    print(f"  daily rows: {len(daily):,}")
    print(f"  unique tickers: {daily['ticker'].nunique()}")
    print(f"  date range: {daily['date'].min().date()} .. {daily['date'].max().date()}")
    print(f"  half_spread_bps  percentiles 25/50/75: {daily['half_spread_bps'].quantile([.25,.5,.75]).values}")
    print(f"  spread_std_1d    percentiles 25/50/75: {daily['spread_std_1d'].quantile([.25,.5,.75]).values}")
    print(f"  median bars/day: {daily['n_bars'].median():.0f}")

    daily.to_parquet(DST, index=False)
    print(f"\nWrote {DST}  ({DST.stat().st_size / 1024**2:.1f} MB) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
