"""
Phase 0 — Data Quality Gateway
================================
Standalone, vectorized. All downstream phases MUST read from validated outputs.

Pipeline spec §0: raw CSVs → 5min_phase1/ + 1min_phase2/ + meta_flags.parquet

Storage deviation from spec (practical): per-ticker parquet files in subdirectories
instead of one massive single file, due to memory constraints (~196M rows for 1-min).
Phase 1 / Phase 2 load individual tickers by path — functionally equivalent.

Output layout:
    data/validated/
    ├── 5min_phase1/{TICKER}.parquet   # 09:35-15:55 ET, log_close + volume
    ├── 1min_phase2/{TICKER}.parquet   # 09:30-15:59 ET, OHLCV
    ├── meta_flags.parquet             # all bad-data flags (all tickers)
    └── gateway_summary.json           # pass/drop counts + binding filters
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import RAW_DIR, VALIDATED_DIR, discover_tickers, load_ticker_raw
from src.utils.stats import rolling_zscore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session boundaries (spec §0.2)
# ---------------------------------------------------------------------------
_SESSION_P1_START = pd.Timestamp("09:35:00").time()
_SESSION_P1_END = pd.Timestamp("15:55:00").time()
_SESSION_P2_START = pd.Timestamp("09:30:00").time()
_SESSION_P2_END = pd.Timestamp("15:59:00").time()

# ---------------------------------------------------------------------------
# Thresholds (spec §0.2 / §0.3)
# ---------------------------------------------------------------------------
_OUTLIER_Z_THRESH = 10.0          # |Z| > 10 → bad print
_OUTLIER_ROLLING_WINDOW = 390     # 1 trading day of 1-min bars
_OUTLIER_DROP_THRESH = 0.01       # >1% outlier fraction → drop ticker
_STALE_BARS = 10                  # ≥10 identical consecutive close → stale
_FREEZE_PCTILE = 0.30             # ≥30% tickers frozen → market-halt day
_VOL_Z_THRESH = 10.0              # |vol Z| > 10 for vol-price coherence check


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _session_mask(index: pd.DatetimeIndex, start, end) -> pd.Series:
    """Boolean mask for bars within session time range (inclusive)."""
    times = index.time
    return pd.Series(
        (times >= start) & (times <= end),
        index=index,
    )


def _outlier_treatment(close: pd.Series) -> tuple[pd.Series, float]:
    """
    Compute log-return Z-scores, flag |Z| > 10 as bad prints.
    Replace flagged close values with NaN, then forward-fill limit=1.

    Returns (cleaned_close, outlier_fraction).
    """
    log_ret = np.log(close / close.shift(1))

    roll_mean = log_ret.rolling(_OUTLIER_ROLLING_WINDOW, min_periods=2).mean()
    roll_std = log_ret.rolling(_OUTLIER_ROLLING_WINDOW, min_periods=2).std(ddof=1)

    z = (log_ret - roll_mean) / roll_std.clip(lower=1e-10)
    bad = z.abs() > _OUTLIER_Z_THRESH

    # bad is a boolean Series (NaN comparisons → False, never NaN); no dropna needed
    outlier_fraction = float(bad.sum() / max(len(bad), 1))

    cleaned = close.copy()
    cleaned[bad] = np.nan
    cleaned = cleaned.ffill(limit=1)

    return cleaned, outlier_fraction


def _check_hard_assertions(df: pd.DataFrame, ticker: str) -> None:
    """
    Fail-fast assertions (spec §0.3). Raises AssertionError on violation.

    1. Monotonic timestamps
    2. No duplicate timestamps
    3. OHLC valid: L ≤ O ≤ H, L ≤ C ≤ H, H ≥ L
    4. Non-negative prices and volume
    """
    # 1. Monotonic
    if not df.index.is_monotonic_increasing:
        raise AssertionError(f"{ticker}: timestamps not monotonically increasing")

    # 2. No duplicates (already deduped in load_ticker_raw, but guard here)
    if df.index.duplicated().any():
        raise AssertionError(f"{ticker}: duplicate timestamps detected")

    # 3. OHLC validity
    hi, lo = df["high"], df["low"]
    bad_hl = (hi < lo).any()
    bad_o = ((df["open"] < lo) | (df["open"] > hi)).any()
    bad_c = ((df["close"] < lo) | (df["close"] > hi)).any()
    if bad_hl:
        raise AssertionError(f"{ticker}: high < low detected")
    if bad_o:
        raise AssertionError(f"{ticker}: open outside [low, high]")
    if bad_c:
        raise AssertionError(f"{ticker}: close outside [low, high]")

    # 4. Non-negative prices and volume
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise AssertionError(f"{ticker}: non-positive prices detected")
    if (df["volume"] < 0).any():
        raise AssertionError(f"{ticker}: negative volume detected")


def _compute_bad_flags(df: pd.DataFrame, ticker: str) -> list[dict]:
    """
    Compute bad-data flags (spec §0.4). Log only — do NOT drop rows.

    Flags:
        stale_price     : ≥10 consecutive identical close values
        intra_session_gap: missing bar within session (expected at 1-min resolution)
        vol_price_incoherence: |vol Z| > 10 AND |return Z| < 1
    """
    flags: list[dict] = []

    # -- Stale price (≥10 consecutive identical close) --
    close = df["close"]
    stale_run = (close == close.shift(1)).astype(int)
    # Use cumsum trick to find run lengths
    group_id = (stale_run == 0).cumsum()
    run_lens = stale_run.groupby(group_id).transform("sum")
    stale_mask = (stale_run == 1) & (run_lens >= _STALE_BARS - 1)
    for ts in df.index[stale_mask]:
        flags.append({
            "ticker": ticker,
            "timestamp_et": ts,
            "flag_type": "stale_price",
            "flag_value": "≥10 consecutive identical close",
        })

    # -- Intra-session gap: missing 1-min bars within session --
    for d, day_df in df.groupby(df.index.normalize()):
        expected = pd.date_range(
            start=d.replace(hour=9, minute=30),
            end=d.replace(hour=15, minute=59),
            freq="1min",
            tz=df.index.tz,
        )
        missing = expected.difference(day_df.index)
        if len(missing) == 0:
            continue
        day_start, day_end = day_df.index[0], day_df.index[-1]
        for ts in missing[(missing >= day_start) & (missing <= day_end)]:
            flags.append({
                "ticker": ticker,
                "timestamp_et": ts,
                "flag_type": "intra_session_gap",
                "flag_value": "missing 1-min bar within session",
            })

    # -- Volume-price coherence: |vol Z| > 10 AND |return Z| < 1 --
    vol_z = rolling_zscore(df["volume"].astype(float), _OUTLIER_ROLLING_WINDOW, min_periods=2)
    log_ret = np.log(close / close.shift(1))
    ret_z = rolling_zscore(log_ret, _OUTLIER_ROLLING_WINDOW, min_periods=2)

    incoherent = (vol_z.abs() > _VOL_Z_THRESH) & (ret_z.abs() < 1.0)
    for ts in df.index[incoherent.fillna(False)]:
        flags.append({
            "ticker": ticker,
            "timestamp_et": ts,
            "flag_type": "vol_price_incoherence",
            "flag_value": f"vol_z={vol_z[ts]:.1f}, ret_z={ret_z[ts]:.2f}",
        })

    return flags


def _resample_5min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min OHLCV to 5-min bars (spec §0.1).
    close → last, volume → sum. Session: 09:35-15:55 ET.
    """
    # Filter to P1 session before resampling
    mask = _session_mask(df_1min.index, _SESSION_P1_START, _SESSION_P1_END)
    df = df_1min.loc[mask]

    if len(df) == 0:
        return pd.DataFrame(columns=["log_close", "volume"])

    df_5 = df.resample("5min", closed="left", label="left").agg(
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    df_5 = df_5.between_time(_SESSION_P1_START, _SESSION_P1_END)
    df_5 = df_5.dropna(subset=["close"])

    df_5["log_close"] = np.log(df_5["close"])
    return df_5[["log_close", "volume"]]


def _check_session_volume(df: pd.DataFrame, ticker: str) -> list[dict]:
    """
    Flag sessions where all volume = 0. Returns flag records (log only).
    Volume-zero exemption for market-halt days is resolved by the caller
    via cross-ticker freeze flags.
    """
    daily_vol = df.groupby(df.index.normalize())["volume"].sum()
    return [
        {
            "ticker": ticker,
            "timestamp_et": d,
            "flag_type": "zero_volume_session",
            "flag_value": f"all-zero volume on {d.date()}",
        }
        for d in daily_vol[daily_vol == 0].index
    ]


# ---------------------------------------------------------------------------
# Cross-ticker freeze detection (spec §0.4, needs all tickers)
# ---------------------------------------------------------------------------

def _detect_cross_ticker_freeze(
    all_close: pd.DataFrame,
) -> list[dict]:
    """
    For each timestamp, check if ≥30% of tickers have frozen close
    (close_t == close_{t-1}). Returns flag records for freeze timestamps.

    all_close: DataFrame with tickers as columns, timestamps as index (1-min).
    """
    flags: list[dict] = []
    if all_close.empty or all_close.shape[1] == 0:
        return flags

    is_frozen = (all_close == all_close.shift(1))
    freeze_pct = is_frozen.mean(axis=1)
    freeze_mask = freeze_pct >= _FREEZE_PCTILE

    for ts in all_close.index[freeze_mask]:
        flags.append({
            "ticker": "__MARKET__",
            "timestamp_et": ts,
            "flag_type": "cross_ticker_freeze",
            "flag_value": f"{freeze_pct[ts]:.1%} of tickers frozen — likely market halt",
        })

    return flags


# ---------------------------------------------------------------------------
# Per-ticker worker (called in parallel)
# ---------------------------------------------------------------------------

def _process_single_ticker(
    ticker: str,
    raw_dir: Path,
    p1_dir: Path,
    p2_dir: Path,
    date_start: str,
    date_end: str,
) -> dict:
    """
    Process one ticker end-to-end: load → assert → clean → flag → write parquets.
    Returns a result dict consumed by run_gateway's aggregation step.
    Designed to be called from a ThreadPoolExecutor worker.
    """
    try:
        df_raw = load_ticker_raw(ticker, raw_dir, date_start, date_end)
    except Exception as exc:
        return {"ticker": ticker, "status": "dropped", "reason": f"load error: {exc}"}

    if len(df_raw) == 0:
        return {"ticker": ticker, "status": "dropped", "reason": "no data after load"}

    # Hard assertions on raw data — before outlier treatment (see gateway.py §0.3 comment)
    try:
        _check_hard_assertions(df_raw, ticker)
    except AssertionError as exc:
        return {"ticker": ticker, "status": "dropped", "reason": str(exc)}

    cleaned_close, outlier_frac = _outlier_treatment(df_raw["close"])
    if outlier_frac > _OUTLIER_DROP_THRESH:
        return {"ticker": ticker, "status": "dropped",
                "reason": f"outlier_fraction={outlier_frac:.3%} > 1%"}

    df_raw["close"] = cleaned_close

    mask_p2 = _session_mask(df_raw.index, _SESSION_P2_START, _SESSION_P2_END)
    df_1min = df_raw.loc[mask_p2].copy()
    if len(df_1min) == 0:
        return {"ticker": ticker, "status": "dropped", "reason": "empty after P2 session filter"}

    flags = _compute_bad_flags(df_1min, ticker) + _check_session_volume(df_1min, ticker)

    df_5min = _resample_5min(df_raw)
    if len(df_5min) == 0:
        return {"ticker": ticker, "status": "dropped", "reason": "empty after 5-min resample"}

    # Parquet writes are safe — each ticker writes to a unique path
    df_1min[["open", "high", "low", "close", "volume"]].to_parquet(p2_dir / f"{ticker}.parquet")
    df_5min.to_parquet(p1_dir / f"{ticker}.parquet")

    return {
        "ticker": ticker,
        "status": "passed",
        "flags": flags,
        "close_series": df_1min["close"],  # for cross-ticker freeze check
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gateway(
    raw_dir: Path = RAW_DIR,
    validated_dir: Path = VALIDATED_DIR,
    date_start: str = "2022-01-03",
    date_end: str = "2026-03-19",
    tickers: list[str] | None = None,
    n_workers: int = 6,
) -> dict:
    """
    Run Phase 0 Data Quality Gateway.

    Processes all tickers in parallel (ThreadPoolExecutor, n_workers=6 default).
    Returns a summary dict with counts, dropped tickers, and flag counts.

    tickers  : if provided, only process this subset (useful for smoke tests).
    n_workers: concurrent threads for ticker processing (I/O-bound → threads effective).
    """
    t0 = time.time()
    raw_dir = Path(raw_dir)
    validated_dir = Path(validated_dir)

    p1_dir = validated_dir / "5min_phase1"
    p2_dir = validated_dir / "1min_phase2"
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = tickers or discover_tickers(raw_dir)
    log.info("Gateway: %d tickers | %d workers | %s to %s",
             len(all_tickers), n_workers, date_start, date_end)

    passed: list[str] = []
    dropped: dict[str, str] = {}
    all_flags: list[dict] = []
    close_frames: dict[str, pd.Series] = {}
    completed = 0
    lock = threading.Lock()

    def _on_done(result: dict) -> None:
        nonlocal completed
        with lock:
            completed += 1
            if completed % 50 == 0:
                log.info("  Progress: %d/%d tickers (%.0fs)", completed, len(all_tickers), time.time() - t0)
            if result["status"] == "dropped":
                dropped[result["ticker"]] = result["reason"]
            else:
                passed.append(result["ticker"])
                all_flags.extend(result.get("flags", []))
                close_frames[result["ticker"]] = result["close_series"]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_single_ticker,
                ticker, raw_dir, p1_dir, p2_dir, date_start, date_end
            ): ticker
            for ticker in all_tickers
        }
        for future in as_completed(futures):
            try:
                _on_done(future.result())
            except Exception as exc:
                ticker = futures[future]
                with lock:
                    dropped[ticker] = f"worker exception: {exc}"

    # -- Cross-ticker freeze check (spec §0.4, needs all tickers) --
    log.info("Running cross-ticker freeze check on %d passed tickers...", len(passed))
    halt_timestamps: set = set()
    if len(close_frames) > 1:
        try:
            close_wide = pd.DataFrame(close_frames)
            freeze_flags = _detect_cross_ticker_freeze(close_wide)
            all_flags.extend(freeze_flags)
            halt_timestamps = {f["timestamp_et"] for f in freeze_flags}
            log.info("  Cross-ticker freeze: %d halt events found", len(freeze_flags))
        except MemoryError:
            log.warning("  Cross-ticker freeze check skipped: MemoryError — "
                        "volume-zero assertion cannot be enforced for this run")
        finally:
            del close_frames

    # -- Enforce §0.3 Hard Assertion 5: volume-zero session drop --
    # A ticker with all-zero volume on a non-halt day violates the hard assertion.
    # We defer this check until after the freeze check so market-halt days
    # (where zero volume is expected) can be correctly exempted.
    zero_vol_flags = [f for f in all_flags if f["flag_type"] == "zero_volume_session"]
    tickers_to_drop: set[str] = set()
    for flag in zero_vol_flags:
        ticker = flag["ticker"]
        # flag["timestamp_et"] is a normalized date (midnight), not a 1-min bar.
        # Check if ANY freeze halt exists on the same calendar date.
        flag_date = pd.Timestamp(flag["timestamp_et"]).date()
        halt_on_same_day = any(
            pd.Timestamp(ts).date() == flag_date for ts in halt_timestamps
        )
        if not halt_on_same_day:
            tickers_to_drop.add(ticker)

    if tickers_to_drop:
        log.warning(
            "§0.3 Hard Assertion: dropping %d tickers with non-halt zero-volume sessions: %s",
            len(tickers_to_drop), sorted(tickers_to_drop),
        )
        p1_dir_path = Path(validated_dir) / "5min_phase1"
        p2_dir_path = Path(validated_dir) / "1min_phase2"
        for ticker in tickers_to_drop:
            for parquet in [p1_dir_path / f"{ticker}.parquet",
                            p2_dir_path / f"{ticker}.parquet"]:
                if parquet.exists():
                    parquet.unlink()
            if ticker in passed:
                passed.remove(ticker)
            dropped[ticker] = "zero_volume_session on non-halt day (§0.3)"
        # Remove their flags from all_flags too (they're dropped, not just flagged)
        all_flags = [f for f in all_flags if f.get("ticker") not in tickers_to_drop]

    # -- Write meta_flags.parquet --
    if all_flags:
        flags_df = pd.DataFrame(all_flags)
        flags_df.to_parquet(validated_dir / "meta_flags.parquet", index=False)
    else:
        pd.DataFrame(
            columns=["ticker", "timestamp_et", "flag_type", "flag_value"]
        ).to_parquet(validated_dir / "meta_flags.parquet", index=False)

    # -- Write gateway summary --
    summary = {
        "n_tickers_attempted": len(all_tickers),
        "n_tickers_passed": len(passed),
        "n_tickers_dropped": len(dropped),
        "dropped_tickers": dropped,
        "n_flags_total": len(all_flags),
        "flag_type_counts": (
            pd.Series([f["flag_type"] for f in all_flags])
            .value_counts()
            .to_dict()
        ) if all_flags else {},
        "elapsed_seconds": round(time.time() - t0, 1),
        "date_range": f"{date_start} to {date_end}",
    }

    with open(validated_dir / "gateway_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    log.info(
        "Gateway complete: %d passed, %d dropped, %d flags | %.0fs",
        len(passed), len(dropped), len(all_flags), time.time() - t0,
    )
    return summary
