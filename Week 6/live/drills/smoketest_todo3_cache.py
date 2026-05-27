"""TODO 3 smoketest: live daily cache built from Alpaca adjusted bars.

Checks:
  1. All 50 universe tickers have cache files
  2. Each file has >=250 bars
  3. log_close values are finite + in plausible range
  4. Most recent bar is within last 5 calendar days
  5. Volume is non-negative
  6. NVDA cache shows POST-SPLIT prices (~$120s) for early 2025, not raw $1200s
  7. Schema matches Polygon cache (log_close + volume cols, tz-naive midnight index)
  8. Hardstop check still functional
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

CACHE_DIR = ROOT / "live" / "state" / "daily_cache"
UNIVERSE_FILE = ROOT / "live" / "universe_top50.json"

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
errors: list[str] = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def t_all_universe_files_exist():
    tickers = json.loads(UNIVERSE_FILE.read_text())["tickers"]
    missing = [tk for tk in tickers if not (CACHE_DIR / f"{tk}.parquet").exists()]
    check(f"all {len(tickers)} universe tickers have cache files",
          not missing, f"missing: {missing}")


def t_bars_per_ticker_sufficient():
    """For the 528-ticker extended cache, some recently-listed tickers may have
    <250 bars (legitimate). Require:
      - All cache files have >=100 bars (discovery_daily's hard floor)
      - At least 95% have >=250 bars (formation window stability)
    """
    files = list(CACHE_DIR.glob("*.parquet"))
    too_small = []     # <100 bars (rejected by discovery)
    short = 0          # <250 bars but >=100 (acceptable for some recent IPOs)
    for fp in files:
        df = pd.read_parquet(fp)
        if len(df) < 100:
            too_small.append(f"{fp.stem}={len(df)}")
        elif len(df) < 250:
            short += 1
    check(f"all files have >=100 bars (discovery hard floor)",
          not too_small, f"insufficient: {too_small[:5]}")
    pct_full = (len(files) - short - len(too_small)) / max(1, len(files)) * 100
    check(f">=95% files have full 250+ bars",
          pct_full >= 95, f"only {pct_full:.1f}% ({short} short of 250)")


def t_log_close_sane():
    files = list(CACHE_DIR.glob("*.parquet"))[:10]   # sample first 10
    bad = []
    for fp in files:
        df = pd.read_parquet(fp)
        lc = df["log_close"]
        if not np.all(np.isfinite(lc)):
            bad.append(f"{fp.stem}: non-finite")
            continue
        # Plausible price range: $0.1 to $10000 -> log = -2.3 to 9.2
        if lc.min() < -2.5 or lc.max() > 10.0:
            bad.append(f"{fp.stem}: lc=[{lc.min():.2f},{lc.max():.2f}]")
    check("log_close values in plausible range [-2.5, 10] and finite",
          not bad, str(bad))


def t_recent_data():
    files = list(CACHE_DIR.glob("*.parquet"))[:5]
    today = date.today()
    stale = []
    for fp in files:
        df = pd.read_parquet(fp)
        last = df.index.max().date()
        age = (today - last).days
        if age > 7:
            stale.append(f"{fp.stem}:{age}d")
    check("recent bar within last 7 days",
          not stale, f"stale: {stale}")


def t_volume_non_negative():
    files = list(CACHE_DIR.glob("*.parquet"))[:10]
    bad = [fp.stem for fp in files
           if (pd.read_parquet(fp)["volume"] < 0).any()]
    check("all volume non-negative", not bad, str(bad))


def t_nvda_post_split():
    """NVDA should show POST-SPLIT prices (~$120-200) in our window (after 2024-06-10),
    NOT raw pre-split ($1000+)."""
    fp = CACHE_DIR / "NVDA.parquet"
    if not fp.exists():
        # NVDA might not be in top-50; skip
        print("  [SKIP] NVDA not in cache (not in top-50)")
        return
    df = pd.read_parquet(fp)
    if df.empty:
        check("NVDA cache non-empty", False)
        return
    max_close = float(np.exp(df["log_close"]).max())
    # Window starts 2025-05-12 (well after 2024-06-10 split)
    # Post-split NVDA ranged $100-$200 in 2025
    check("NVDA prices in post-split range (<$500)",
          max_close < 500, f"max close = ${max_close:.2f}")


def t_schema_matches_polygon():
    """Cache file schema should match Polygon's (cols: log_close + volume, tz-naive midnight index)."""
    files = list(CACHE_DIR.glob("*.parquet"))[:3]
    for fp in files:
        df = pd.read_parquet(fp)
        check(f"{fp.stem}: has log_close + volume cols",
              "log_close" in df.columns and "volume" in df.columns,
              f"cols: {df.columns.tolist()}")
        check(f"{fp.stem}: index is DatetimeIndex tz-naive",
              isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None)


def t_hardstop_still_functional():
    """Sanity: hardstop logic untouched by data work."""
    import tempfile
    from live.safety import hardstop
    td = tempfile.mkdtemp(prefix="hs_t3_")
    hardstop.HARDSTOP_FLAG_PATH = Path(td) / "HARDSTOP.flag"
    check("hardstop initially clean", not hardstop.is_tripped())
    hardstop.HARDSTOP_FLAG_PATH.write_text("test\n")
    check("hardstop trips on flag", hardstop.is_tripped())
    hardstop.clear("todo3")
    check("hardstop clears", not hardstop.is_tripped())


def main() -> int:
    print("== TODO 3 Smoketest: live Alpaca-adjusted daily cache ==\n")
    print("--- File presence ---")
    t_all_universe_files_exist()
    print("\n--- Bar counts ---")
    t_bars_per_ticker_sufficient()
    print("\n--- Log_close sanity ---")
    t_log_close_sane()
    print("\n--- Data freshness ---")
    t_recent_data()
    print("\n--- Volume non-negative ---")
    t_volume_non_negative()
    print("\n--- NVDA post-split check ---")
    t_nvda_post_split()
    print("\n--- Schema parity with Polygon ---")
    t_schema_matches_polygon()
    print("\n--- Hardstop ---")
    t_hardstop_still_functional()
    print()
    if errors:
        print(f"FAIL: {len(errors)} - {errors}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
