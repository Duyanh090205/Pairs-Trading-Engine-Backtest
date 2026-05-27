"""
Build daily parquet cache from 5-min source — one-shot data prep.

Reads every ticker in `Week 4/data/validated/5min_phase1/`, resamples to daily
close-to-close (via engine_daily.data_daily.resample_5min_to_daily), and writes
the result to `Week 4/data/validated/daily_phase3/<TICKER>.parquet`.

Run once after any 5-min data refresh. The V4 pipeline reads daily_phase3
directly — no on-the-fly resampling.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
from pathlib import Path

import pandas as pd

WEEK6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK6))

from engine_daily.data_daily import resample_5min_to_daily

SRC = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\5min_phase1")
DST = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated\daily_phase3")


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source dir not found: {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    print(f"Source: {SRC}")
    print(f"Destination: {DST}")

    parquet_files = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet"))
    print(f"Found {len(parquet_files)} 5-min parquets\n")

    t0 = time.time()
    n_done = 0
    n_skip = 0
    total_5min_rows = 0
    total_daily_rows = 0
    skipped_reasons: dict[str, int] = {}

    for fname in parquet_files:
        tk = fname[:-len(".parquet")]
        try:
            df_5min = pd.read_parquet(SRC / fname)
            daily = resample_5min_to_daily(df_5min)
            if len(daily) == 0:
                n_skip += 1
                skipped_reasons["empty_after_resample"] = \
                    skipped_reasons.get("empty_after_resample", 0) + 1
                continue
            daily.to_parquet(DST / fname)
            n_done += 1
            total_5min_rows += len(df_5min)
            total_daily_rows += len(daily)
        except Exception as e:
            n_skip += 1
            reason = type(e).__name__
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            print(f"  skip {tk}: {e}")

    elapsed = time.time() - t0
    print(f"\nBuilt {n_done} daily parquets in {elapsed:.1f}s")
    if n_skip:
        print(f"Skipped {n_skip}: {skipped_reasons}")
    if n_done > 0:
        avg_5min = total_5min_rows / n_done
        avg_daily = total_daily_rows / n_done
        print(f"Average per ticker: {avg_5min:.0f} 5-min bars -> {avg_daily:.0f} daily bars "
              f"({100.0 * avg_daily / avg_5min:.2f}% of size)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
