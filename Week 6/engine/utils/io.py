"""
I/O utilities: raw CSV loading from minute_ohlc_flatfiles and validated parquet access.
Adapted from Week 1/notebooks/01_data_profiling.py — flat-file layout, ns UTC timestamps.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

RAW_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\minute_ohlc_flatfiles")
VALIDATED_DIR = Path(r"d:\Quant Finance\Quant Program\Week 4\data\validated")

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_LOAD_COLS = ["open", "high", "low", "close", "volume", "window_start"]


def discover_tickers(raw_dir: Path = RAW_DIR) -> list[str]:
    """Return sorted list of unique tickers found in raw_dir by parsing filenames."""
    tickers: set[str] = set()
    for fpath in Path(raw_dir).glob("*.csv"):
        parts = fpath.stem.rsplit("_", 1)
        if len(parts) == 2:
            tickers.add(parts[0])
    return sorted(tickers)


def load_ticker_raw(
    ticker: str,
    raw_dir: Path = RAW_DIR,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """
    Load all daily CSV files for `ticker` and return a 1-min DataFrame.

    Index  : DatetimeIndex in US/Eastern (tz-aware).
    Columns: open, high, low, close, volume  (float / int).

    Timestamps from window_start (nanosecond UTC int) are converted via:
        pd.to_datetime(ns_int, unit='ns', utc=True).tz_convert('US/Eastern')

    date_start / date_end (YYYY-MM-DD strings) filter which CSV files are loaded;
    exact timestamp filtering happens in the gateway session filter.
    """
    pattern = str(Path(raw_dir) / f"{ticker}_*.csv")
    files = sorted(glob.glob(pattern))

    if date_start:
        files = [f for f in files if Path(f).stem.split("_", 1)[1] >= date_start]
    if date_end:
        files = [f for f in files if Path(f).stem.split("_", 1)[1] <= date_end]

    if not files:
        return pd.DataFrame(columns=_OHLCV_COLS)

    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, usecols=_LOAD_COLS)
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=_OHLCV_COLS)

    raw = pd.concat(frames, ignore_index=True)

    raw["timestamp_et"] = (
        pd.to_datetime(raw["window_start"], unit="ns", utc=True)
        .dt.tz_convert("US/Eastern")
    )
    raw = raw.set_index("timestamp_et").sort_index()
    raw = raw[_OHLCV_COLS]

    # Drop duplicate timestamps (keep last, per Week 1 convention)
    dup_mask = raw.index.duplicated(keep="last")
    if dup_mask.any():
        raw = raw[~dup_mask]

    return raw


# ---------------------------------------------------------------------------
# Validated parquet helpers
# Phase 0 writes per-ticker files into subdirectories:
#   data/validated/5min_phase1/{TICKER}.parquet
#   data/validated/1min_phase2/{TICKER}.parquet
#   data/validated/meta_flags.parquet  (all tickers, single file)
# ---------------------------------------------------------------------------

def read_5min(ticker: str, validated_dir: Path = VALIDATED_DIR) -> pd.DataFrame:
    path = Path(validated_dir) / "5min_phase1" / f"{ticker}.parquet"
    return pd.read_parquet(path)


def read_1min(ticker: str, validated_dir: Path = VALIDATED_DIR) -> pd.DataFrame:
    path = Path(validated_dir) / "1min_phase2" / f"{ticker}.parquet"
    return pd.read_parquet(path)


def list_validated_tickers(window: str = "5min", validated_dir: Path = VALIDATED_DIR) -> list[str]:
    """Return tickers that have a validated parquet file for the given window."""
    subdir = "5min_phase1" if window == "5min" else "1min_phase2"
    folder = Path(validated_dir) / subdir
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.parquet"))


def read_meta_flags(validated_dir: Path = VALIDATED_DIR) -> pd.DataFrame:
    path = Path(validated_dir) / "meta_flags.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "timestamp_et", "flag_type", "flag_value"])
    return pd.read_parquet(path)
