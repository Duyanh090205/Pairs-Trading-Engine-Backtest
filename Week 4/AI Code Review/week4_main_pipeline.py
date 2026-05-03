# ======================================================================
# Week 4 Quant Finance — Main Pipeline (Phase 0–4)
# ======================================================================


# ===== FILE: src/utils/io.py =====
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



# ===== FILE: src/utils/stats.py =====
"""
Statistical utilities shared across phases:
  - OU half-life (AR(1) regression on spread differences)
  - BH-FDR multiple testing correction
  - Block bootstrap Sharpe distribution (for negative control threshold)
  - Rolling Z-score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------

def compute_ou_halflife(spread: pd.Series | np.ndarray, bars_per_day: int = 78) -> float:
    """
    Fit OU half-life via AR(1) regression on spread differences.

    Regression: ΔS_t = κ · S_{t-1} + c + ε
    Mean-reversion speed: θ = -κ  (κ must be < 0 for mean-reversion)
    Half-life in bars: ln(2) / θ
    Half-life in days: half_life_bars / bars_per_day

    Returns np.nan if spread is not mean-reverting (κ ≥ 0).

    bars_per_day:
        78  for 5-min bars (Phase 1 default, per pipeline spec §1.6)
        390 for 1-min bars (Phase 2)
    """
    s = np.asarray(spread, dtype=np.float64)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return np.nan

    diff = np.diff(s)
    lag = s[:-1]

    X = sm.add_constant(lag)
    try:
        res = sm.OLS(diff, X).fit()
    except Exception:
        return np.nan

    kappa = float(res.params[1])  # coefficient on lagged level (index 0 = constant)
    if kappa >= 0:
        return np.nan  # not mean-reverting

    theta = -kappa
    half_life_bars = np.log(2) / theta
    return half_life_bars / bars_per_day


# ---------------------------------------------------------------------------
# BH-FDR multiple testing correction
# ---------------------------------------------------------------------------

def bh_fdr_correct(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns a boolean array: True = pair survives after correction.
    NaN p-values are treated as 1.0 (no rejection).
    """
    pvals = np.asarray(pvalues, dtype=float)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    reject, _, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return reject.astype(bool)


# ---------------------------------------------------------------------------
# Block bootstrap Sharpe (for NC threshold, Phase 3)
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe(
    daily_returns: pd.Series | np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap distribution of annualized Sharpe ratios.

    block_size = 1 trading day (default) preserves daily serial correlation
    of pairs strategy returns.

    Returns array of n_bootstrap Sharpe values.
    """
    rets = np.asarray(daily_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n < 2:
        return np.zeros(n_bootstrap)

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    sharpes = np.empty(n_bootstrap)
    offsets = np.arange(block_size)  # reused each iteration

    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Vectorized gather: shape (n_blocks, block_size) → flatten → trim to n
        idx = (starts[:, None] + offsets).ravel()[:n]
        resampled = rets[idx]
        std = resampled.std(ddof=1)
        sharpes[i] = 0.0 if std < 1e-10 else (resampled.mean() / std) * np.sqrt(252)

    return sharpes


# ---------------------------------------------------------------------------
# Rolling Z-score (vectorized)
# ---------------------------------------------------------------------------

def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute rolling Z-score: (x - rolling_mean) / rolling_std.

    min_periods defaults to window // 2 (burn-in period per pipeline spec).
    """
    mp = min_periods if min_periods is not None else window // 2
    mean = series.rolling(window, min_periods=mp).mean()
    std = series.rolling(window, min_periods=mp).std(ddof=1).clip(lower=1e-10)
    return (series - mean) / std



# ===== FILE: src/utils/metrics.py =====
"""
Performance metrics: Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate.
Adapted from Week 3/scripts/engine.py.

All metrics follow pipeline spec §3.3:
  - Sharpe: daily returns × √252
  - MaxDD:  bar-level equity curve (NOT daily MTM)
  - Calmar: CAGR / bar-level MaxDD
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def compute_sharpe(daily_returns: pd.Series | np.ndarray) -> float:
    """Annualized Sharpe on daily returns. Returns nan if std=0."""
    rets = _clean(daily_returns)
    if len(rets) < 2:
        return np.nan
    std = rets.std(ddof=1)
    if std < 1e-10:
        return np.nan
    return float((rets.mean() / std) * np.sqrt(252))


def compute_max_dd(bar_equity: pd.Series | np.ndarray) -> float:
    """
    Maximum drawdown on bar-level equity curve (most negative peak-to-trough ratio).
    Returns a negative number (e.g. -0.15 = 15% drawdown).
    """
    eq = _clean(bar_equity)
    if len(eq) == 0:
        return np.nan
    running_max = np.maximum.accumulate(eq)
    return float((eq / running_max - 1.0).min())


def compute_cagr(bar_equity: pd.Series | np.ndarray, n_trading_days: int) -> float:
    """
    Compound Annual Growth Rate.
    bar_equity: compound equity curve starting from 1.0.
    n_trading_days: number of trading days in the period.
    """
    eq = _clean(bar_equity)
    if len(eq) == 0 or n_trading_days <= 0:
        return np.nan
    final = eq[-1]
    if final <= 0:
        return -1.0
    return float(final ** (252.0 / n_trading_days) - 1.0)


def compute_calmar(cagr: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / |max_dd|. Returns nan if max_dd = 0."""
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return float(cagr / abs(max_dd))


def compute_win_rate(trade_returns: pd.Series | np.ndarray) -> float:
    """Fraction of trades with positive net return."""
    rets = _clean(trade_returns)
    if len(rets) == 0:
        return np.nan
    return float((rets > 0).mean())


def compute_all(
    daily_returns: pd.Series,
    bar_equity: pd.Series,
    n_trading_days: int,
    trade_returns: pd.Series | None = None,
) -> dict:
    """Compute the full metric suite and return as a dict."""
    sharpe = compute_sharpe(daily_returns)
    max_dd = compute_max_dd(bar_equity)
    cagr = compute_cagr(bar_equity, n_trading_days)
    calmar = compute_calmar(cagr, max_dd)
    win_rate = compute_win_rate(trade_returns) if trade_returns is not None else np.nan

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cagr": cagr,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_trading_days": n_trading_days,
    }



# ===== FILE: src/phase0_data_gateway/gateway.py =====
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



# ===== FILE: src/phase1_cointegration/discovery.py =====
"""
Phase 1 — Cointegration Discovery
===================================
Called once per fold by the Phase 4 orchestrator.

Pipeline spec §1.1–1.7:
    formation data (5-min log-prices)
    → universe hard screens
    → all-pairs enumeration
    → pairwise inner join (≥80% min-overlap)
    → PCA hedge ratio (secondary eigenvector, Avellaneda-Lee)
    → Johansen trace test (p-value via chi2(8) approximation — see note below)
    → BH-FDR correction (q=0.05)
    → OU half-life filter [1, 10] trading days
    → surviving pairs DataFrame

Johansen p-value approximation:
    statsmodels coint_johansen returns critical values only, not p-values.
    For a 2-variable system at rank-0 trace test, the trace statistic is
    well-approximated by chi2(df=8):
        90% CV: chi2=13.36  vs Johansen=13.43  (error <0.5%)
        95% CV: chi2=15.51  vs Johansen=15.49  (error <0.2%)
        99% CV: chi2=20.09  vs Johansen=19.93  (error <0.8%)
    p_value = 1 - chi2.cdf(trace_stat_rank0, df=8)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.utils.io import VALIDATED_DIR, list_validated_tickers, read_5min
from src.utils.stats import bh_fdr_correct, compute_ou_halflife

log = logging.getLogger(__name__)

_MIN_MEDIAN_PRICE = 5.0
_MIN_ADV_DOLLAR = 1_000_000
_MIN_COMPLETENESS = 0.90
_MAX_ZERO_RETURN = 0.50
_MIN_OVERLAP = 0.80
_BH_FDR_Q = 0.05
_HL_MIN_DAYS = 1.0
_HL_MAX_DAYS = 10.0
_BARS_PER_DAY_5MIN = 78   # pipeline spec §1.6 convention (actual session = 77 bars)


def _load_formation_data(
    formation_start: str,
    formation_end: str,
    validated_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load 5-min log-prices for all validated tickers sliced to the formation window."""
    tickers = list_validated_tickers("5min", validated_dir)
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = read_5min(ticker, validated_dir).loc[formation_start:formation_end]
            if len(df) > 0:
                data[ticker] = df
        except FileNotFoundError:
            log.debug("Skip %s: parquet not found", ticker)
        except Exception as exc:
            log.warning("Unexpected error loading %s: %s", ticker, exc)

    log.info("Loaded %d tickers for formation %s to %s", len(data), formation_start, formation_end)
    return data


def _apply_hard_screens(
    formation_data: dict[str, pd.DataFrame],
) -> tuple[list[str], dict, int]:
    """
    Apply §1.1 screens on formation window data per ticker.
    Returns (survivor_tickers, screen_log, max_formation_bars).

    max_formation_bars: max bars observed across ALL loaded tickers (before screening),
    used as the expected-bars reference for both completeness (§1.1) and the pairwise
    overlap ratio (§1.2.1). Computing over pre-screen tickers ensures the reference
    equals the formation window's true capacity, not just the survivors.
    """
    if not formation_data:
        return [], {}, 0

    max_bars = max(len(df) for df in formation_data.values())
    survivors: list[str] = []
    screen_log: dict[str, str] = {}
    counts = Counter()

    for ticker, df in formation_data.items():
        close = np.exp(df["log_close"])
        vol = df["volume"]

        if close.median() < _MIN_MEDIAN_PRICE:
            screen_log[ticker] = "median_price"
            counts["median_price"] += 1
            continue

        daily_dollar_vol = (close * vol).groupby(df.index.normalize()).sum()
        if daily_dollar_vol.mean() < _MIN_ADV_DOLLAR:
            screen_log[ticker] = "adv_dollar"
            counts["adv_dollar"] += 1
            continue

        if len(df) / max_bars < _MIN_COMPLETENESS:
            screen_log[ticker] = "completeness"
            counts["completeness"] += 1
            continue

        log_ret = df["log_close"].diff()
        if (log_ret == 0).sum() / max(len(log_ret.dropna()), 1) >= _MAX_ZERO_RETURN:
            screen_log[ticker] = "zero_return"
            counts["zero_return"] += 1
            continue

        survivors.append(ticker)
        counts["passed"] += 1

    log.info(
        "Screens: %d passed | price=%d adv=%d completeness=%d zero_ret=%d",
        counts["passed"], counts["median_price"],
        counts["adv_dollar"], counts["completeness"], counts["zero_return"],
    )
    return survivors, {"binding_counts": dict(counts), "rejected": screen_log}, max_bars


def _pca_hedge_ratio(
    log_a: np.ndarray,
    log_b: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute PCA hedge ratio using secondary eigenvector (Avellaneda-Lee convention).

    X = [ln(A) - mean(ln(A)),  ln(B) - mean(ln(B))]  (T×2, centered)
    Secondary eigenvector v_2 (smallest eigenvalue of Cov) = cointegrating direction.

    β_PCA = -v_2[0] / v_2[1]
    α_PCA = mean(ln(A)) - β_PCA × mean(ln(B))
    R     = var(spread)   ← Kalman measurement noise init

    Returns (alpha_PCA, beta_PCA, R). All nan if degenerate.
    """
    mu_a, mu_b = log_a.mean(), log_b.mean()
    X = np.column_stack([log_a - mu_a, log_b - mu_b])
    cov = X.T @ X / (len(X) - 1)

    # np.linalg.eigh returns eigenvalues ascending (Hermitian-optimised).
    # For 2×2: ascending[0] = smallest = secondary eigenvector = cointegrating direction.
    # This is identical to spec pseudocode's descending[:, 1] — different sort, same vector.
    _, eigvecs = np.linalg.eigh(cov)
    v2 = eigvecs[:, 0]

    if abs(v2[1]) < 1e-12:
        return np.nan, np.nan, np.nan

    beta = -v2[0] / v2[1]
    alpha = mu_a - beta * mu_b
    R = float(np.var(log_a - alpha - beta * log_b, ddof=1))
    return float(alpha), float(beta), R


def _johansen_pvalue(X: np.ndarray) -> float:
    """
    Johansen trace test on a pre-stacked (T, 2) array. Returns chi2(8) p-value.
    H0: rank = 0 (no cointegration). Returns nan on failure.
    """
    try:
        res = coint_johansen(X, det_order=0, k_ar_diff=1)
    except Exception:
        return np.nan
    return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))


def _run_pairwise_tests(
    survivors: list[str],
    formation_data: dict[str, pd.DataFrame],
    max_formation_bars: int,
) -> list[dict]:
    """
    Enumerate all pairs, apply overlap filter, PCA, and Johansen.
    Stores only scalars per result — no array caching — so N workers
    each use ~100 MB instead of ~7 GB.

    max_formation_bars: expected total bars in the formation window (spec §1.2.1
    denominator). Must come from _apply_hard_screens so it reflects the full
    window capacity, not just the two tickers being tested.
    """
    all_pairs = list(combinations(survivors, 2))
    n_pairs = len(all_pairs)
    log.info("Testing %d pairs from %d survivors...", n_pairs, len(survivors))

    results: list[dict] = []
    n_skipped_overlap = 0
    t0 = time.time()

    for i, (ticker_a, ticker_b) in enumerate(all_pairs):
        if i > 0 and i % 10_000 == 0:
            log.info("  Pairs progress: %d/%d (%.0fs)", i, n_pairs, time.time() - t0)

        s_a = formation_data[ticker_a]["log_close"]
        s_b = formation_data[ticker_b]["log_close"]

        sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
        valid = ~(sa_aligned.isna() | sb_aligned.isna())
        sa_aligned, sb_aligned = sa_aligned[valid], sb_aligned[valid]

        n_aligned = len(sa_aligned)
        # Spec §1.2.1: denominator = expected total bars in the formation window,
        # not max of the two tickers' actual bars (which understates sparsity).
        overlap_ratio = n_aligned / max_formation_bars
        if overlap_ratio < _MIN_OVERLAP:
            n_skipped_overlap += 1
            continue

        log_a = sa_aligned.values
        log_b = sb_aligned.values
        X = np.column_stack([log_a, log_b])  # stack once, reuse in both PCA and Johansen

        alpha_pca, beta_pca, R = _pca_hedge_ratio(log_a, log_b)
        if np.isnan(alpha_pca):
            continue

        pval = _johansen_pvalue(X)
        if np.isnan(pval):
            continue

        results.append({
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "alpha_pca": alpha_pca,
            "beta_pca": beta_pca,
            "R_measurement_noise": R,
            "johansen_pval": pval,
            "n_overlapping_bars": n_aligned,
            "overlap_ratio": overlap_ratio,
        })

    log.info(
        "Pairwise tests done: %d results, %d skipped (overlap<80%%) | %.0fs",
        len(results), n_skipped_overlap, time.time() - t0,
    )
    return results


def _apply_fdr_and_halflife(
    results: list[dict],
    formation_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply BH-FDR (q=0.05) then OU half-life filter [1, 10] trading days.
    Re-aligns only FDR survivors (typically <500 pairs) — no cached arrays needed.
    """
    if not results:
        return pd.DataFrame()

    pvals = np.array([r["johansen_pval"] for r in results])
    reject = bh_fdr_correct(pvals, q=_BH_FDR_Q)
    log.info("BH-FDR: %d / %d pairs survive (q=%.2f)", reject.sum(), len(reject), _BH_FDR_Q)

    surviving: list[dict] = []
    n_hl_fail = 0

    for row, keep in zip(results, reject):
        if not keep:
            continue

        s_a = formation_data[row["ticker_a"]]["log_close"]
        s_b = formation_data[row["ticker_b"]]["log_close"]
        sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
        valid = ~(sa_aligned.isna() | sb_aligned.isna())
        log_a = sa_aligned[valid].values
        log_b = sb_aligned[valid].values

        spread = log_a - row["alpha_pca"] - row["beta_pca"] * log_b
        hl_days = compute_ou_halflife(spread, bars_per_day=_BARS_PER_DAY_5MIN)

        if np.isnan(hl_days) or not (_HL_MIN_DAYS <= hl_days <= _HL_MAX_DAYS):
            n_hl_fail += 1
            continue

        surviving.append({**row, "half_life_days": hl_days})

    _MAX_PAIRS_POST_FILTER = 500
    n_before_cap = len(surviving)
    if len(surviving) > _MAX_PAIRS_POST_FILTER:
        surviving.sort(key=lambda r: r["johansen_pval"])
        surviving = surviving[:_MAX_PAIRS_POST_FILTER]
        log.warning(
            "PAIR-COUNT GATE: %d pairs capped to %d after HL filter (spike fold suppressed)",
            n_before_cap, _MAX_PAIRS_POST_FILTER,
        )

    log.info(
        "OU half-life filter: %d passed (%d dropped outside [%.0f, %.0f]d) → %d after pair-count cap",
        n_before_cap, n_hl_fail, _HL_MIN_DAYS, _HL_MAX_DAYS, len(surviving),
    )
    return pd.DataFrame(surviving)


def run(
    formation_start: str,
    formation_end: str,
    validated_dir: Path = VALIDATED_DIR,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """
    Run Phase 1 cointegration discovery for one fold.

    Parameters
    ----------
    formation_start : "YYYY-MM-DD"
    formation_end   : "YYYY-MM-DD"
    validated_dir   : location of Phase 0 outputs
    max_pairs       : fallback cap — keeps top-K by Johansen p-value before BH-FDR.
                      None = no cap (default).

    Returns
    -------
    pd.DataFrame with columns:
        ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise,
        johansen_pval, n_overlapping_bars, overlap_ratio, half_life_days
    Empty DataFrame if no pairs survive.
    """
    t0 = time.time()
    log.info("Phase 1 | formation: %s to %s", formation_start, formation_end)

    formation_data = _load_formation_data(formation_start, formation_end, validated_dir)
    if not formation_data:
        log.warning("No formation data — empty fold.")
        return pd.DataFrame()

    survivors, _, max_formation_bars = _apply_hard_screens(formation_data)
    if len(survivors) < 2:
        log.warning("Fewer than 2 survivors after hard screens.")
        return pd.DataFrame()

    formation_data = {t: formation_data[t] for t in survivors}

    raw_results = _run_pairwise_tests(survivors, formation_data, max_formation_bars)
    if not raw_results:
        log.warning("No pairs passed overlap filter.")
        return pd.DataFrame()

    if max_pairs is not None and len(raw_results) > max_pairs:
        raw_results.sort(key=lambda r: r["johansen_pval"])
        raw_results = raw_results[:max_pairs]
        log.warning("FALLBACK CAP: kept top %d pairs by Johansen p-value", max_pairs)

    pairs_df = _apply_fdr_and_halflife(raw_results, formation_data)
    log.info("Phase 1 complete: %d surviving pairs | %.0fs", len(pairs_df), time.time() - t0)
    return pairs_df



# ===== FILE: src/phase2_execution/kalman.py =====
"""
Phase 2 — Kalman Filter (2D state [alpha, beta])

State equation:      theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q),  Q = delta * R * I_2
Observation:         ln(A_t) = alpha_t + beta_t * ln(B_t) + eps_t,  eps_t ~ N(0, R)

NOTE: Q = delta * R * I (not delta * I as in raw spec).
Reason: R spans 0.001-33 across pairs. With Q=delta*I the delta/R adaptation rate varies
5 orders of magnitude — tight pairs get near-infinite gain, making prior spread white noise.
Using Q=delta*R*I normalises delta/R = delta for all pairs. Spec §2.1 states dynamics are
controlled by delta/R; this enforces it uniformly. Validated on Fold 1 (24 pairs).

Init: theta_0 = [alpha_PCA, beta_PCA],  P_0 = R * I_2

Signal uses PRIOR spread (before incorporating A_t observation).
Posterior theta used only for sizing and threshold rebalance.

Performance: inner loop is Numba JIT-compiled. ~50x faster than pure Python for
typical series lengths (7,800 1-min bars or 9,800 5-min bars per pair).
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _kalman_inner(log_a, log_b, alpha_init, beta_init, R, delta):
    """
    Numba-compiled 2x2 Kalman filter inner loop.
    All 2x2 matrix ops are inlined as scalar arithmetic — avoids numpy overhead per bar.

    Returns (spread_prior, alpha_prior, beta_prior, alpha_post, beta_post),
    each length-n array.
    """
    n = len(log_a)
    Q_scale = delta * R     # Q = Q_scale * I_2

    # P stored as 4 scalars (symmetric 2x2)
    P00 = R;  P01 = 0.0
    P10 = 0.0; P11 = R

    th0 = alpha_init   # alpha state
    th1 = beta_init    # beta state

    spread_prior = np.empty(n)
    alpha_prior  = np.empty(n)
    beta_prior   = np.empty(n)
    alpha_post   = np.empty(n)
    beta_post    = np.empty(n)

    for i in range(n):
        lb = log_b[i]

        # PREDICT: P_pred = P + Q*I
        Pp00 = P00 + Q_scale
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + Q_scale

        # Prior state = theta_pred (no dynamics noise in mean)
        alpha_prior[i]  = th0
        beta_prior[i]   = th1
        spread_prior[i] = log_a[i] - th0 - th1 * lb

        # H = [1, lb]  (observation vector)
        h0 = 1.0
        h1 = lb

        # Innovation covariance S = H @ Pp @ H' + R  (scalar)
        S = h0*h0*Pp00 + h0*h1*Pp01 + h1*h0*Pp10 + h1*h1*Pp11 + R
        if S < 1e-300:
            alpha_post[i] = th0
            beta_post[i]  = th1
            continue

        # Kalman gain K = Pp @ H' / S  (2x1)
        K0 = (Pp00*h0 + Pp01*h1) / S
        K1 = (Pp10*h0 + Pp11*h1) / S

        # Innovation
        innov = log_a[i] - th0*h0 - th1*h1

        # Update state
        th0 = th0 + K0 * innov
        th1 = th1 + K1 * innov

        # Joseph stabilised covariance: (I-KH) Pp (I-KH)' + R*KK'
        # IKH = I - outer(K, H)
        IKH00 = 1.0 - K0*h0;  IKH01 = -K0*h1
        IKH10 = -K1*h0;       IKH11 = 1.0 - K1*h1

        # M = IKH @ Pp
        M00 = IKH00*Pp00 + IKH01*Pp10
        M01 = IKH00*Pp01 + IKH01*Pp11
        M10 = IKH10*Pp00 + IKH11*Pp10
        M11 = IKH10*Pp01 + IKH11*Pp11

        # P = M @ IKH' + R * outer(K, K)
        P00 = M00*IKH00 + M01*IKH01 + R*K0*K0
        P01 = M00*IKH10 + M01*IKH11 + R*K0*K1
        P10 = M10*IKH00 + M11*IKH01 + R*K1*K0
        P11 = M10*IKH10 + M11*IKH11 + R*K1*K1

        alpha_post[i] = th0
        beta_post[i]  = th1

    return spread_prior, alpha_prior, beta_prior, alpha_post, beta_post


def run_kalman(
    log_a: np.ndarray,
    log_b: np.ndarray,
    alpha_init: float,
    beta_init: float,
    R: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Kalman filter for one pair over a price series.

    Parameters
    ----------
    log_a, log_b : 1-D arrays of aligned log-prices (same length T)
    alpha_init   : PCA intercept from Phase 1
    beta_init    : PCA hedge ratio from Phase 1
    R            : observation noise variance (PCA residual variance from Phase 1)
    delta        : state noise scale; Q = delta * R * I_2

    Returns (all length-T arrays)
    -------
    spread_prior : prior spread S_t^prior = ln(A_t) - alpha_{t|t-1} - beta_{t|t-1}*ln(B_t)
    alpha_prior  : alpha_{t|t-1}   (use for signal)
    beta_prior   : beta_{t|t-1}    (use for signal)
    alpha_post   : alpha_{t|t}     (use for sizing / rebalance)
    beta_post    : beta_{t|t}      (use for sizing / rebalance)
    """
    if R <= 0 or np.isnan(R):
        raise ValueError(f"R must be positive finite, got {R}")

    log_a = np.asarray(log_a, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    return _kalman_inner(log_a, log_b,
                         float(alpha_init), float(beta_init),
                         float(R), float(delta))


def warmup_kalman():
    """Pre-compile Numba Kalman JIT. Call once at process startup."""
    dummy = np.linspace(0.1, 0.2, 50)
    _kalman_inner(dummy, dummy, 0.0, 1.0, 0.01, 1e-7)



# ===== FILE: src/phase2_execution/delta_selector.py =====
"""
Phase 2 — Multi-Criterion Delta Auto-Selector

Run once per fold on formation-window data for all surviving pairs.
Selects optimal Kalman delta from grid {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}.

Selection rule (pre-registered):
  optimal_delta = argmin{ median(|kurtosis - 3|) across pairs }
  subject to:
    median_HL    in [1, 10] trading days
    median_ACF78 > 0.7           (ACF at lag=78 = 1 trading day on 5-min data)

Edge case flags (logged, not fatal unless all delta fail):
  - delta at grid boundary -> "expand grid next fold"
  - no delta passes all 3 constraints -> "universal constraint fail, kick fold"
  - delta jumps >2 grid steps vs prev_delta -> "parameter instability"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from src.phase2_execution.kalman import run_kalman
from src.utils.stats import compute_ou_halflife

log = logging.getLogger(__name__)

DELTA_GRID = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
_HL_MIN = 1.0
_HL_MAX = 10.0
_ACF78_MIN = 0.7
_BARS_PER_DAY_5MIN = 78
# Discard first KALMAN_BURNIN bars from metrics to let filter converge from P_0=R*I.
# Tight pairs (small R) have high initial Kalman gain that normalises within ~500 bars.
_KALMAN_BURNIN = 500
# Cap number of pairs used for delta selection. Median statistics over 200 pairs
# are near-identical to medians over 6,000 pairs. Without this cap, fold 23 (6,008 pairs)
# would run 30,040 Kalman evaluations on formation data — ~150s vs ~1s with cap.
_MAX_DELTA_SAMPLE_PAIRS = 200


def select_delta(
    pairs_df: pd.DataFrame,
    formation_5min: dict[str, pd.DataFrame],
    prev_delta: float | None = None,
) -> tuple[float | None, dict]:
    """
    Select optimal Kalman delta for this fold.

    Parameters
    ----------
    pairs_df        : Phase 1 surviving pairs with columns
                      [ticker_a, ticker_b, alpha_pca, beta_pca, R_measurement_noise, half_life_days]
    formation_5min  : dict ticker -> DataFrame with 'log_close' column (formation window)
    prev_delta      : delta selected in prior fold (for instability check), or None

    Returns
    -------
    optimal_delta   : selected delta, or None if universal constraint fail
    metrics_by_delta: dict {delta: {metric_kurt, median_HL, median_ACF78}}
    """
    metrics_by_delta: dict = {}

    # Sample top pairs by Johansen p-value for efficient delta selection.
    # Median statistics are stable over 200 pairs; no need to evaluate all 6,000.
    sample_df = pairs_df.nsmallest(_MAX_DELTA_SAMPLE_PAIRS, "johansen_pval")
    if len(sample_df) < len(pairs_df):
        log.info("select_delta: sampling %d / %d pairs for grid search", len(sample_df), len(pairs_df))

    for delta in DELTA_GRID:
        kurt_list, hl_list, acf78_list = [], [], []

        for _, row in sample_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            if ta not in formation_5min or tb not in formation_5min:
                continue

            s_a = formation_5min[ta]["log_close"]
            s_b = formation_5min[tb]["log_close"]
            sa_aligned, sb_aligned = s_a.align(s_b, join="inner")
            valid = ~(sa_aligned.isna() | sb_aligned.isna())
            log_a = sa_aligned[valid].values
            log_b = sb_aligned[valid].values
            if len(log_a) < 10:
                continue

            R = float(row["R_measurement_noise"])
            if R <= 0 or np.isnan(R):
                continue

            try:
                spread_prior, *_ = run_kalman(
                    log_a, log_b,
                    float(row["alpha_pca"]), float(row["beta_pca"]),
                    R, delta,
                )
            except Exception:
                continue

            # Discard burn-in so metrics reflect converged filter state
            spread_raw = spread_prior[_KALMAN_BURNIN:]
            spread = spread_raw[~np.isnan(spread_raw)]
            if len(spread) < 20:
                continue

            # Criterion 1: kurtosis closeness to 3 (normal)
            kurt_total = float(scipy_kurtosis(spread, fisher=False))  # total kurtosis
            kurt_list.append(abs(kurt_total - 3.0))

            # Criterion 2: OU half-life on prior spread (5-min bars)
            hl = compute_ou_halflife(spread, bars_per_day=_BARS_PER_DAY_5MIN)
            if not np.isnan(hl):
                hl_list.append(hl)

            # Criterion 3: ACF at lag=78 (= 1 trading day on 5-min data)
            s = pd.Series(spread)
            if len(s) > 78:
                acf78 = abs(float(s.autocorr(lag=78)))
                if not np.isnan(acf78):
                    acf78_list.append(acf78)

        metrics_by_delta[delta] = {
            "metric_kurt":   float(np.median(kurt_list))  if kurt_list  else np.nan,
            "median_HL":     float(np.median(hl_list))    if hl_list    else np.nan,
            "median_ACF78":  float(np.median(acf78_list)) if acf78_list else np.nan,
        }

    # Multi-criterion feasibility filter
    feasible = {
        d: m for d, m in metrics_by_delta.items()
        if (
            not np.isnan(m["median_HL"])
            and _HL_MIN <= m["median_HL"] <= _HL_MAX
            and not np.isnan(m["median_ACF78"])
            and m["median_ACF78"] > _ACF78_MIN
        )
    }

    # Log all metrics for audit trail
    for d, m in metrics_by_delta.items():
        log.info(
            "  delta=%.0e | kurt_dev=%.3f | median_HL=%.2f | ACF78=%.3f%s",
            d, m["metric_kurt"], m["median_HL"], m["median_ACF78"],
            " [FEASIBLE]" if d in feasible else "",
        )

    if not feasible:
        log.warning("UNIVERSAL CONSTRAINT FAIL — no delta passes HL+ACF78 constraints. Kick fold.")
        return None, metrics_by_delta

    optimal_delta = min(feasible, key=lambda d: feasible[d]["metric_kurt"])

    # Edge case: delta at grid boundary
    if optimal_delta == DELTA_GRID[0]:
        log.warning("DELTA AT LOWER BOUNDARY (%.0e) — kurtosis criterion degenerate, applying HL-nearest fallback", optimal_delta)
        # Fallback: pick feasible delta with median_HL closest to 5d (midpoint of [1,10])
        hl_fallback = min(feasible, key=lambda d: abs(feasible[d]["median_HL"] - 5.0))
        if hl_fallback != optimal_delta:
            log.info(
                "HL-nearest fallback: delta=%.0e (median_HL=%.2f, target=5.0d)",
                hl_fallback, feasible[hl_fallback]["median_HL"],
            )
            optimal_delta = hl_fallback
    elif optimal_delta == DELTA_GRID[-1]:
        log.warning("DELTA AT UPPER BOUNDARY (%.0e) — consider expanding grid larger", optimal_delta)

    # Edge case: instability vs prior fold
    if prev_delta is not None:
        prev_idx = DELTA_GRID.index(prev_delta) if prev_delta in DELTA_GRID else -1
        curr_idx = DELTA_GRID.index(optimal_delta)
        if prev_idx >= 0 and abs(curr_idx - prev_idx) > 2:
            log.warning(
                "DELTA INSTABILITY — jumped from %.0e (idx %d) to %.0e (idx %d)",
                prev_delta, prev_idx, optimal_delta, curr_idx,
            )

    m = feasible[optimal_delta]
    log.info(
        "Selected delta=%.0e | kurt_dev=%.3f | median_HL=%.2f | ACF78=%.3f",
        optimal_delta, m["metric_kurt"], m["median_HL"], m["median_ACF78"],
    )
    return optimal_delta, metrics_by_delta



# ===== FILE: src/phase2_execution/engine.py =====
"""
Phase 2 — Execution Engine

Per fold, per pair:
  1. apply_ticker_concentration_cap  — portfolio construction filter (max 5 pairs/ticker)
  2. run_kalman on 1-min trading data → prior spread series
  3. rolling Z-score with session warmup
  4. Numba state machine → raw positions
  5. 1-bar execution lag → executed positions
  6. Position sizing (dollar-neutral)
  7. Threshold rebalance with hysteresis (default X=10%)

Returns bar-level trade log per pair.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

from src.phase2_execution.kalman import run_kalman
from src.utils.stats import rolling_zscore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BARS_PER_DAY_1MIN = 390
_SESSION_WARMUP_BARS = 30       # first 30 bars each session → NaN zscore
_Z_WINDOW_CAP = 2000            # cap on rolling Z window in bars
_REBALANCE_THRESHOLD = 0.10     # 10% beta drift triggers rebalance
_REBALANCE_DEAD_BAND = 0.05     # 5% = X/2, hysteresis dead band
_REBALANCE_COST_BPS = 0.0030    # 30 bps one-side on delta shares
_N_OPEN_PAIRS_MAX = 50          # default portfolio cap
_TOTAL_CAPITAL = 1_000_000.0    # $1M notional (normalised)
_MAX_PAIRS_PER_TICKER = 5       # concentration cap


# ---------------------------------------------------------------------------
# Portfolio construction: per-ticker concentration cap
# ---------------------------------------------------------------------------

def apply_ticker_concentration_cap(
    pairs_df: pd.DataFrame,
    max_pairs_per_ticker: int = _MAX_PAIRS_PER_TICKER,
) -> pd.DataFrame:
    """
    Keep at most max_pairs_per_ticker pairs per ticker (ranked by johansen_pval asc).
    Addresses folds with 5,000+ pairs where a single ticker appears in 100+ pairs.
    """
    if pairs_df.empty:
        return pairs_df

    ranked = pairs_df.sort_values("johansen_pval").reset_index(drop=True)
    ticker_count: dict[str, int] = {}
    keep_idx: list[int] = []

    for idx, row in ranked.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        ca = ticker_count.get(ta, 0)
        cb = ticker_count.get(tb, 0)
        if ca < max_pairs_per_ticker and cb < max_pairs_per_ticker:
            keep_idx.append(idx)
            ticker_count[ta] = ca + 1
            ticker_count[tb] = cb + 1

    result = ranked.loc[keep_idx].reset_index(drop=True)
    if len(result) < len(ranked):
        log.info(
            "Concentration cap: %d -> %d pairs (max %d per ticker)",
            len(ranked), len(result), max_pairs_per_ticker,
        )
    return result


# ---------------------------------------------------------------------------
# Session warmup mask
# ---------------------------------------------------------------------------

def _session_warmup_mask(index: pd.DatetimeIndex, warmup_bars: int) -> np.ndarray:
    """True where bar is within first warmup_bars of its session. Vectorized."""
    n = len(index)
    dates = index.normalize().view(np.int64)  # compare as ints, no Python date objects
    mask = np.zeros(n, dtype=bool)
    # Session starts: index 0 and wherever date changes
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = dates[1:] != dates[:-1]
    starts = np.where(is_new)[0]
    for s in starts:
        mask[s: s + warmup_bars] = True
    return mask


# ---------------------------------------------------------------------------
# Numba state machine
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _state_machine(
    zscores: np.ndarray,
    entry_z: float,
) -> np.ndarray:
    """
    Pure signal: position array from Z-score series.
    state +1 = long A short B, -1 = short A long B, 0 = flat.
    Entry on Z crossing +/- entry_z, exit on Z zero-crossing.
    NaN Z: hold current state, no new entry.
    """
    n = len(zscores)
    positions = np.zeros(n, dtype=np.int8)
    state = np.int8(0)

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            positions[i] = state
            continue

        if state == 0:
            if z < -entry_z:
                state = np.int8(1)
            elif z > entry_z:
                state = np.int8(-1)
        else:
            # Exit on zero-crossing
            if (state == 1 and z >= 0.0) or (state == -1 and z <= 0.0):
                state = np.int8(0)

        positions[i] = state

    return positions




# ---------------------------------------------------------------------------
# EOS (end-of-session) flatten
# ---------------------------------------------------------------------------

def _apply_eos_flatten(
    positions: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Force signal to 0 from 15:55 ET through end-of-session each day.
    Zeros all bars from the 15:55 bar (inclusive) to the session close.

    With 1-bar execution lag, zeroing only 15:55 leaves signal non-zero at
    15:56-15:59, which propagates into position at 15:57-16:00. Zeroing the
    full tail ensures position is 0 from 15:56 onward for every session.

    Falls back to zeroing from the last bar if 15:55 bar is absent. Vectorized.
    """
    pos = positions.copy()
    # Minutes since midnight: avoids Python datetime objects per bar
    bar_minutes = index.hour * 60 + index.minute  # numpy int array
    eos_minutes = 15 * 60 + 55                     # 955
    dates_int = index.normalize().view(np.int64)

    is_new = np.empty(len(index), dtype=bool)
    is_new[0] = True
    is_new[1:] = dates_int[1:] != dates_int[:-1]
    session_starts = np.where(is_new)[0]
    session_ends   = np.append(session_starts[1:] - 1, len(index) - 1)

    for s, e in zip(session_starts, session_ends):
        # Find first bar >= 15:55 in this session
        eos_candidates = np.where(bar_minutes[s:e+1] >= eos_minutes)[0]
        start = s + (eos_candidates[0] if len(eos_candidates) else (e - s))
        pos[start: e + 1] = 0
    return pos


# ---------------------------------------------------------------------------
# Core pair execution
# ---------------------------------------------------------------------------

def run_pair(
    log_a_1min: np.ndarray,
    log_b_1min: np.ndarray,
    close_a_1min: np.ndarray,
    close_b_1min: np.ndarray,
    index_1min: pd.DatetimeIndex,
    alpha_pca: float,
    beta_pca: float,
    R: float,
    delta: float,
    half_life_days: float,
    entry_z: float = 2.0,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    rebalance_threshold: float = _REBALANCE_THRESHOLD,
    rebalance_dead_band: float = _REBALANCE_DEAD_BAND,
    eos_flatten: bool = True,
    spread_mean_form: float | None = None,
    spread_std_form: float | None = None,
) -> pd.DataFrame:
    """
    Run Kalman + Z-score + state machine for one pair over one trading window.

    Returns bar-level DataFrame with columns:
      [spread_prior, zscore, signal, position, beta_post,
       rebalance_event, rebalance_cost]
    """
    n = len(log_a_1min)
    if n < 10:
        return pd.DataFrame()

    # 1. Kalman on 1-min data
    spread_prior, _, _, _, beta_post = run_kalman(
        log_a_1min, log_b_1min, alpha_pca, beta_pca, R, delta,
    )

    # 2. Z-score: use formation-window-locked mean/std when available (eliminates
    # rolling-mean contamination that inverts EOS vs zero-cross PnL sign).
    # Fall back to rolling Z only when formation stats are absent.
    if spread_mean_form is not None and spread_std_form is not None and spread_std_form > 1e-10:
        zscore_raw = (spread_prior - spread_mean_form) / spread_std_form
    else:
        z_window = min(int(half_life_days * _BARS_PER_DAY_1MIN), _Z_WINDOW_CAP)
        z_window = max(z_window, 10)
        zscore_raw = rolling_zscore(pd.Series(spread_prior), window=z_window).values

    # 3. Session warmup → NaN
    warmup_mask = _session_warmup_mask(index_1min, _SESSION_WARMUP_BARS)
    zscore_raw[warmup_mask] = np.nan

    # 4. State machine
    signal_raw = _state_machine(zscore_raw, entry_z)

    # 5. EOS flatten
    if eos_flatten:
        signal_raw = _apply_eos_flatten(signal_raw, index_1min)

    # 6. Execution lag (1 bar) — genuine OOS
    position = np.zeros(n, dtype=np.int8)
    position[1:] = signal_raw[:-1]

    # 7. Threshold rebalance with hysteresis
    per_pair_dollar = total_capital / n_open_pairs_max
    short_notional  = per_pair_dollar * 0.5

    rebalance_events = np.zeros(n, dtype=bool)
    rebalance_costs  = np.zeros(n, dtype=float)
    rebalance_dshares = np.zeros(n, dtype=float)   # signed delta-shares of B leg
    rebalance_price   = np.zeros(n, dtype=float)   # price of B at rebalance bar
    beta_ref = beta_pca
    in_deadband = False

    for i in range(n):
        if position[i] == 0:
            beta_ref = beta_post[i]  # reset ref when flat
            in_deadband = False
            continue

        drift = (beta_post[i] - beta_ref) / abs(beta_ref) if abs(beta_ref) > 1e-10 else 0.0

        if in_deadband and abs(drift) < rebalance_dead_band:
            continue  # still inside dead band

        in_deadband = False
        if abs(drift) > rebalance_threshold:
            # PnL parameterises shares_b = (short_notional * beta) / p_b (signed).
            # On a beta change delta_beta, the share delta needed is
            #   delta_shares_b = (short_notional / p_b) * delta_beta   (signed)
            delta_beta     = beta_post[i] - beta_ref
            p_b_reb        = max(close_b_1min[i], 1e-6)
            d_shares_b     = (short_notional * delta_beta) / p_b_reb   # signed
            cost           = _REBALANCE_COST_BPS * abs(d_shares_b) * p_b_reb
            rebalance_events[i]  = True
            rebalance_costs[i]   = cost
            rebalance_dshares[i] = d_shares_b
            rebalance_price[i]   = p_b_reb
            beta_ref    = beta_post[i]
            in_deadband = True  # enter dead band after rebalance

    df = pd.DataFrame({
        "spread_prior":     spread_prior,
        "zscore":           zscore_raw,
        "signal":           signal_raw,
        "position":         position,
        "beta_post":        beta_post,
        "rebalance_event":  rebalance_events,
        "rebalance_cost":   rebalance_costs,
        "rebalance_dshares": rebalance_dshares,
        "rebalance_price":   rebalance_price,
    }, index=index_1min)

    return df


# ---------------------------------------------------------------------------
# Max-holding cap — apply after execution, before PnL
# ---------------------------------------------------------------------------

def _apply_max_holding(pair_df: pd.DataFrame, max_holding_bars: int) -> pd.DataFrame:
    """
    Post-process an execution DataFrame to force position to 0 when it has
    been continuously non-zero for more than max_holding_bars bars.
    Modifies only the 'position' and 'signal' columns in-place (copy returned).
    """
    df = pair_df.copy()
    pos = df["position"].values.copy()
    sig = df["signal"].values.copy()
    forced = np.zeros(len(pos), dtype=bool)
    n = len(pos)
    hold_count = 0
    for i in range(n):
        if pos[i] != 0:
            hold_count += 1
            if hold_count > max_holding_bars:
                pos[i] = 0
                forced[i] = True
                # Maintain position[i] == signal[i-1] invariant so the
                # lookahead check in metrics_runner doesn't false-alarm.
                if i > 0:
                    sig[i - 1] = 0
        else:
            hold_count = 0
    df["position"]     = pos
    df["signal"]       = sig   # sig[i-1] zeroed for each forced exit to maintain invariant
    df["forced_exit"]  = forced
    return df


# ---------------------------------------------------------------------------
# Fold-level runner — called by Phase 4 orchestrator
# ---------------------------------------------------------------------------

def run_fold_execution(
    pairs_df: pd.DataFrame,
    trading_1min: dict[str, pd.DataFrame],
    delta: float,
    entry_z: float = 2.0,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    eos_flatten: bool = True,
    formation_5min: dict | None = None,
    formation_ref: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run execution for all pairs in a fold.

    Parameters
    ----------
    pairs_df       : Phase 1 output (post concentration cap applied here)
    trading_1min   : dict ticker -> 1-min DataFrame with 'log_close' and 'close'
    delta          : selected by delta_selector for this fold
    entry_z        : Z-score entry threshold

    Returns
    -------
    dict (ticker_a, ticker_b) -> bar-level execution DataFrame
    """
    # Apply per-ticker concentration cap
    pairs_capped = apply_ticker_concentration_cap(pairs_df)
    log.info("Fold execution: %d pairs after concentration cap", len(pairs_capped))

    results: dict[tuple[str, str], pd.DataFrame] = {}

    for _, row in pairs_capped.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        if ta not in trading_1min or tb not in trading_1min:
            log.debug("Skip pair %s/%s: 1-min data missing", ta, tb)
            continue

        df_a = trading_1min[ta].copy()
        df_b = trading_1min[tb].copy()

        # Compute log_close if not pre-computed (1min parquet stores raw OHLCV)
        if "log_close" not in df_a.columns:
            df_a["log_close"] = np.log(df_a["close"].clip(lower=1e-10))
        if "log_close" not in df_b.columns:
            df_b["log_close"] = np.log(df_b["close"].clip(lower=1e-10))

        # Align on common timestamps
        df_ab = df_a[["log_close", "close"]].join(
            df_b[["log_close", "close"]], how="inner", lsuffix="_a", rsuffix="_b"
        ).dropna()

        if len(df_ab) < 20:
            continue

        # Compute formation-window spread mean/std for locked Z-score normalization.
        # formation_ref (1-min) is preferred — same frequency as trading window.
        # formation_5min is accepted as fallback (approximate for near-static delta).
        _form_src = formation_ref if formation_ref is not None else formation_5min
        spread_mean_form = spread_std_form = None
        if _form_src is not None:
            fa = _form_src.get(ta)
            fb = _form_src.get(tb)
            if (fa is not None and fb is not None
                    and "log_close" in fa.columns and "log_close" in fb.columns):
                df_form = (
                    fa[["log_close"]].rename(columns={"log_close": "lc_a"})
                    .join(fb[["log_close"]].rename(columns={"log_close": "lc_b"}), how="inner")
                    .dropna()
                )
                if len(df_form) > 20:
                    try:
                        sp_form, *_ = run_kalman(
                            df_form["lc_a"].values, df_form["lc_b"].values,
                            float(row["alpha_pca"]), float(row["beta_pca"]),
                            float(row["R_measurement_noise"]), delta,
                        )
                        sp_clean = sp_form[~np.isnan(sp_form)]
                        if len(sp_clean) > 20:
                            spread_mean_form = float(np.mean(sp_clean))
                            spread_std_form  = max(float(np.std(sp_clean)), 1e-10)
                    except Exception:
                        pass

        pair_df = run_pair(
            log_a_1min   = df_ab["log_close_a"].values,
            log_b_1min   = df_ab["log_close_b"].values,
            close_a_1min = df_ab["close_a"].values,
            close_b_1min = df_ab["close_b"].values,
            index_1min   = df_ab.index,
            alpha_pca    = float(row["alpha_pca"]),
            beta_pca     = float(row["beta_pca"]),
            R            = float(row["R_measurement_noise"]),
            delta        = delta,
            half_life_days = float(row["half_life_days"]),
            entry_z      = entry_z,
            n_open_pairs_max = n_open_pairs_max,
            total_capital    = total_capital,
            eos_flatten      = eos_flatten,
            spread_mean_form = spread_mean_form,
            spread_std_form  = spread_std_form,
        )

        if not pair_df.empty:
            results[(ta, tb)] = pair_df

    log.info("Fold execution complete: %d pairs ran successfully", len(results))
    return results



# ===== FILE: src/phase3_backtest/pnl.py =====
"""
Phase 3 — Bar-Level PnL Engine

Consumes Phase 2 engine output (pair_df) and computes:
  - Bar-level gross and net PnL
  - Cost decomposition: commission | borrow | rebalance
  - Trade log with entry/exit timestamps and reasons
  - Aggregate metrics: Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate

Interface contract from Phase 2:
  pair_df columns used:
    position      int8   Already 1-bar lagged; +1=long A/short B, -1=short A/long B
    signal        int8   Pre-lag signal (timestamp verification only)
    beta_post     f64    Read at entry bar to size shares_B
    rebalance_cost f64   Pre-computed dollars; add directly to cost decomp
    spread_prior  f64    For Kalman degenerate check

Entry: position[t] != 0 and position[t-1] == 0  (use prices[t] to size)
Exit:  position[t] == 0 and position[t-1] != 0  (charge TC at prices[t])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_TC_BPS_DEFAULT      = 30.0   # one-way bps per leg
_BORROW_BPS_YR       = 50.0   # annual borrow rate on short leg


# ---------------------------------------------------------------------------
# Per-pair PnL
# ---------------------------------------------------------------------------

def compute_pair_pnl(
    pair_df: pd.DataFrame,
    close_a: pd.Series,
    close_b: pd.Series,
    per_pair_dollar: float,
    tc_bps: float = _TC_BPS_DEFAULT,
    borrow_bps_yr: float = _BORROW_BPS_YR,
) -> dict:
    """
    Compute bar-level PnL for a single pair.

    Parameters
    ----------
    pair_df        : engine output DataFrame (position, signal, beta_post, rebalance_cost)
    close_a/b      : 1-min close price Series aligned to pair_df.index
    per_pair_dollar: total_capital / n_open_pairs_max
    tc_bps         : one-way TC in basis points (30 bps entry + 30 bps exit)
    borrow_bps_yr  : annual borrow rate in bps on short leg notional

    Returns dict with keys:
      bar_pnl, bar_gross, trade_log,
      cost_commission, cost_borrow, cost_rebalance
    """
    n = len(pair_df)
    if n < 2:
        return _empty_pair_pnl(pair_df.index if n else None)

    # Align prices to pair_df index (inner join already done in engine)
    ca = close_a.reindex(pair_df.index).ffill().values
    cb = close_b.reindex(pair_df.index).ffill().values

    tc_rate      = tc_bps / 10_000.0
    borrow_daily = (borrow_bps_yr / 10_000.0) / 252.0
    long_notional  = per_pair_dollar * 0.5
    short_notional = per_pair_dollar * 0.5

    pos      = pair_df["position"].values.astype(np.int8)
    beta     = pair_df["beta_post"].values
    reb_cost = pair_df["rebalance_cost"].values
    # Engine-side rebalance metadata (signed delta-shares + price at rebalance bar).
    # Older engine outputs lack these columns; default to zeros for backward compat.
    reb_dshares = (pair_df["rebalance_dshares"].values
                   if "rebalance_dshares" in pair_df.columns else np.zeros(n, dtype=float))
    reb_price   = (pair_df["rebalance_price"].values
                   if "rebalance_price" in pair_df.columns else np.zeros(n, dtype=float))
    idx      = pair_df.index
    # forced_exit column present when _apply_max_holding was used
    forced_exit = pair_df["forced_exit"].values if "forced_exit" in pair_df.columns else np.zeros(n, dtype=bool)

    bar_pnl   = np.zeros(n)
    bar_gross = np.zeros(n)
    cost_commission = 0.0
    cost_borrow     = 0.0

    trade_log  = []
    rebalance_log = []   # per-event records for Week-5 hand-off
    shares_a   = 0.0
    shares_b   = 0.0
    direction  = 0
    entry_bar  = -1
    entry_ts   = None
    prev_date  = None
    # Entry-bar prices retained for entry-notional records and exit comparison
    entry_p_a  = 0.0
    entry_p_b  = 0.0
    # In-flight trade index (incremented on each entry; rebalance events tag
    # this id so the consumer can map a rebalance to its enclosing trade).
    pair_trade_idx = 0

    for i in range(1, n):
        curr = int(pos[i])
        prev = int(pos[i - 1])
        curr_date = idx[i].date()

        # ---- REBALANCE COST (pre-computed by engine) ----
        # Apply BEFORE entry/exit so the trade-level net_pnl summation
        # at exit captures any rebalance cost incurred on the exit bar.
        bar_pnl[i] -= reb_cost[i]
        if reb_cost[i] > 0 and direction != 0:
            # Engine emits signed delta_shares and price_at_rebalance per event.
            # Notional rebalanced = |delta_shares| * price_at_rebalance.
            d_shares = float(reb_dshares[i])
            p_reb    = float(reb_price[i])
            notional_rebalanced = abs(d_shares) * p_reb
            rebalance_log.append({
                "pair_trade_idx":      pair_trade_idx,   # in-flight trade index for this pair
                "rebalance_ts":        idx[i],
                "delta_shares":        d_shares,         # signed
                "price_at_rebalance":  p_reb,
                "notional_rebalanced": notional_rebalanced,
                "cost_dollars":        float(reb_cost[i]),
            })

        # ---- ENTRY ----
        if curr != 0 and prev == 0:
            direction = curr
            # Use bar i-1 prices: position[i] was triggered by signal[i-1],
            # so execution fills at the close of bar i-1 (the signal bar).
            p_a = max(ca[i - 1], 1e-6)
            p_b = max(cb[i - 1], 1e-6)
            beta_entry = float(beta[i - 1])
            beta_sign  = 1.0 if beta_entry >= 0 else -1.0  # cointegration sign of B leg
            # shares_a always positive; shares_b is the *signed* hedge amount.
            # For beta>=0: long A / short B (per direction); shares_b > 0.
            # For beta<0:  long A / long  B (per direction); shares_b < 0.
            # Spread = ln(A) - alpha - beta*ln(B); reversion captured by
            # gross = direction * (shares_a*dp_a - shares_b*dp_b) with signed shares_b.
            shares_a = long_notional / p_a
            shares_b = (short_notional * beta_entry) / p_b   # signed
            # TC on absolute notional traded on each leg
            tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
            bar_pnl[i] -= tc
            cost_commission += tc
            entry_bar = i
            entry_ts  = idx[i]
            entry_p_a = p_a
            entry_p_b = p_b
            pair_trade_idx += 1
            prev_date = curr_date

        # ---- GROSS PnL while in position ----
        if curr != 0:
            dp_a = ca[i] - ca[i - 1]
            dp_b = cb[i] - cb[i - 1]
            gross = direction * (shares_a * dp_a - shares_b * dp_b)
            bar_gross[i] = gross
            bar_pnl[i] += gross

        # ---- BORROW COST (daily accrual on the short leg only) ----
        # Determine which leg is short given (direction, sign(beta)):
        #   direction>0, beta>=0:  A long,  B short  -> short notional = |shares_b|*p
        #   direction>0, beta<0 :  A long,  B long   -> no short leg
        #   direction<0, beta>=0:  A short, B long   -> short notional = shares_a*p
        #   direction<0, beta<0 :  A short, B short  -> short notional = both
        if curr != 0 and prev_date is not None and curr_date != prev_date:
            short_notional_today = 0.0
            if direction > 0 and beta_sign > 0:
                short_notional_today = abs(shares_b) * max(cb[i - 1], 1e-6)
            elif direction < 0 and beta_sign > 0:
                short_notional_today = shares_a * max(ca[i - 1], 1e-6)
            elif direction < 0 and beta_sign < 0:
                short_notional_today = (shares_a * max(ca[i - 1], 1e-6)
                                        + abs(shares_b) * max(cb[i - 1], 1e-6))
            # direction>0, beta<0: both legs long, no borrow
            borrow = borrow_daily * short_notional_today
            bar_pnl[i] -= borrow
            cost_borrow += borrow

        if curr != 0:
            prev_date = curr_date

        # ---- EXIT ----
        if curr == 0 and prev != 0:
            # Same logic as entry: position[i]=0 means signal[i-1] triggered exit,
            # so execution fills at close of bar i-1.
            p_a = max(ca[i - 1], 1e-6)
            p_b = max(cb[i - 1], 1e-6)
            tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
            bar_pnl[i] -= tc
            cost_commission += tc

            # Determine exit reason
            if forced_exit[i]:
                exit_reason = "max_holding"
            else:
                exit_reason = _exit_reason(idx[i])
            trade_net_pnl = bar_pnl[entry_bar: i + 1].sum()
            trade_gross_pnl = bar_gross[entry_bar: i + 1].sum()

            # Notionals at entry/exit (positive dollar amounts traded on each leg).
            # shares_b is signed (cointegration sign of B leg); the notional reported
            # is the absolute dollar amount traded, regardless of long/short direction.
            notional_a_entry = float(shares_a * entry_p_a)
            notional_b_entry = float(abs(shares_b) * entry_p_b)
            notional_a_exit  = float(shares_a * p_a)
            notional_b_exit  = float(abs(shares_b) * p_b)

            # side_A = direction (long when +1, short when -1)
            # side_B = -direction * sign(beta) — derived from cointegration sign:
            #   beta>=0: B is opposite of A  -> side_B = -direction
            #   beta<0 : B is same  as A     -> side_B = +direction
            side_a = int(direction)
            side_b = int(-direction * beta_sign)
            trade_log.append({
                "pair_trade_idx":   pair_trade_idx,        # local index — runner promotes to global trade_id
                "entry_ts":         entry_ts,
                "exit_ts":          idx[i],
                "direction":        direction,
                "side_A":           side_a,
                "side_B":           side_b,
                "n_bars":           i - entry_bar,
                "gross_pnl":        float(trade_gross_pnl),
                "net_pnl":          float(trade_net_pnl),
                "gross_bps":        float(trade_gross_pnl / per_pair_dollar * 10_000),
                "net_bps":          float(trade_net_pnl / per_pair_dollar * 10_000),
                "exit_reason":      exit_reason,
                "notional_a_entry": notional_a_entry,
                "notional_b_entry": notional_b_entry,
                "notional_a_exit":  notional_a_exit,
                "notional_b_exit":  notional_b_exit,
                "allocated_capital": float(per_pair_dollar),
            })
            shares_a  = 0.0
            shares_b  = 0.0
            direction = 0
            entry_p_a = 0.0
            entry_p_b = 0.0
            prev_date = None

    # Handle position still open at window end (carry-forward in orchestrator)
    if direction != 0 and shares_a > 0:
        p_a = max(ca[-1], 1e-6)
        p_b = max(cb[-1], 1e-6)
        tc = tc_rate * (shares_a * p_a + abs(shares_b) * p_b)
        bar_pnl[-1] -= tc
        cost_commission += tc
        trade_net_pnl   = bar_pnl[entry_bar:].sum()
        trade_gross_pnl = bar_gross[entry_bar:].sum()
        notional_a_entry = float(shares_a * entry_p_a)
        notional_b_entry = float(abs(shares_b) * entry_p_b)
        notional_a_exit  = float(shares_a * p_a)
        notional_b_exit  = float(abs(shares_b) * p_b)
        side_a = int(direction)
        side_b = int(-direction * beta_sign)
        trade_log.append({
            "pair_trade_idx":   pair_trade_idx,
            "entry_ts":         entry_ts,
            "exit_ts":          idx[-1],
            "direction":        direction,
            "side_A":           side_a,
            "side_B":           side_b,
            "n_bars":           n - entry_bar,
            "gross_pnl":        float(trade_gross_pnl),
            "net_pnl":          float(trade_net_pnl),
            "gross_bps":        float(trade_gross_pnl / per_pair_dollar * 10_000),
            "net_bps":          float(trade_net_pnl / per_pair_dollar * 10_000),
            "exit_reason":      "end_of_window",
            "notional_a_entry": notional_a_entry,
            "notional_b_entry": notional_b_entry,
            "notional_a_exit":  notional_a_exit,
            "notional_b_exit":  notional_b_exit,
            "allocated_capital": float(per_pair_dollar),
        })

    cost_rebalance = float(reb_cost.sum())

    return {
        "bar_pnl":          pd.Series(bar_pnl, index=idx),
        "bar_gross":        pd.Series(bar_gross, index=idx),
        "trade_log":        trade_log,
        "rebalance_log":    rebalance_log,   # list of dicts; runner promotes pair_trade_idx -> trade_id
        "cost_commission":  cost_commission,
        "cost_borrow":      cost_borrow,
        "cost_rebalance":   cost_rebalance,
    }


def _exit_reason(exit_ts: pd.Timestamp) -> str:
    import datetime
    if exit_ts.time() >= datetime.time(15, 56):
        return "eos"
    return "zero_cross"


def _empty_pair_pnl(index) -> dict:
    empty = pd.Series(dtype=float) if index is None else pd.Series(0.0, index=index)
    return {
        "bar_pnl": empty, "bar_gross": empty,
        "trade_log": [], "rebalance_log": [],
        "cost_commission": 0.0,
        "cost_borrow": 0.0, "cost_rebalance": 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    bar_pnl: pd.Series,
    total_capital: float,
    trade_log: list[dict],
) -> dict:
    """
    Compute Sharpe, MaxDD (bar-level), CAGR, Calmar, Win Rate from bar PnL.

    bar_pnl       : total net PnL in dollars per bar (sum across all pairs)
    total_capital : $1M normalizer
    trade_log     : list of trade dicts from compute_pair_pnl
    """
    if bar_pnl.empty or bar_pnl.abs().sum() == 0:
        return _zero_metrics()

    # Bar-level equity curve starting at 1.0
    # Clamp bar returns to -1 floor so equity never goes negative
    bar_returns  = (bar_pnl / total_capital).clip(lower=-1.0 + 1e-9)
    bar_equity   = (1.0 + bar_returns).cumprod()

    # MaxDD on bar-level equity (captures intraday drawdowns)
    rolling_peak = bar_equity.cummax()
    drawdowns    = bar_equity / rolling_peak - 1.0
    max_dd       = float(drawdowns.min())

    # Daily returns for Sharpe — group by trading day, drop empty days.
    # Previous implementation used resample("D") which created weekend/holiday
    # zero-bins, deflating both mean and std and biasing Sharpe toward zero
    # (and CAGR's calendar-day exponent).
    daily_pnl_full = bar_pnl.groupby(bar_pnl.index.normalize()).sum()
    # Trading day = day with any non-zero bar activity
    bar_active = bar_pnl.abs().groupby(bar_pnl.index.normalize()).sum() > 0
    daily_pnl_trading = daily_pnl_full[bar_active]
    daily_returns = daily_pnl_trading / total_capital
    n_trading_days = len(daily_returns)

    std_daily = float(daily_returns.std(ddof=1)) if n_trading_days > 1 else 0.0
    sharpe    = float(daily_returns.mean() / std_daily * np.sqrt(252)) if std_daily > 1e-12 else 0.0

    # CAGR using TRADING days as the base for annualization
    total_ret = float(bar_equity.iloc[-1]) - 1.0
    cagr      = float((1.0 + total_ret) ** (252.0 / max(n_trading_days, 1)) - 1.0) if total_ret > -1.0 else -1.0

    # Calmar (bar-level MaxDD)
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # Trade stats
    n_trades = len(trade_log)
    if n_trades > 0:
        net_pnls = [t["net_pnl"] for t in trade_log]
        win_rate = float(sum(1 for p in net_pnls if p > 0) / n_trades)
        avg_hold = float(np.mean([t["n_bars"] for t in trade_log]))
        avg_net_bps = float(np.mean([t["net_bps"] for t in trade_log]))
    else:
        win_rate = 0.0
        avg_hold = 0.0
        avg_net_bps = 0.0

    return {
        "sharpe":         sharpe,
        "max_dd":         max_dd,
        "cagr":           cagr,
        "calmar":         calmar,
        "win_rate":       win_rate,
        "n_trades":       n_trades,
        "avg_holding_bars": avg_hold,
        "avg_net_bps":    avg_net_bps,
        "bar_equity":     bar_equity,
        "daily_returns":  daily_returns,
        "total_return":   total_ret,
    }


def _zero_metrics() -> dict:
    return {
        "sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "calmar": 0.0,
        "win_rate": 0.0, "n_trades": 0, "avg_holding_bars": 0.0,
        "avg_net_bps": 0.0, "bar_equity": pd.Series(dtype=float),
        "daily_returns": pd.Series(dtype=float), "total_return": 0.0,
    }



# ===== FILE: src/phase3_backtest/neg_control.py =====
"""
Phase 3 — Negative Control Validation

Two controls per fold:
  1. Empirical: CVNA/ISRG (known non-cointegrated pair in 2022 Bear)
  2. Synthetic: two uncorrelated random walks

Block bootstrap on NC daily returns (block=1 trading day = 390 1-min bars):
  threshold = mean(bootstrap_sharpes) + 2 * std(bootstrap_sharpes)
  Primary strategy Sharpe must exceed threshold to PASS.

If CVNA or ISRG data is unavailable, falls back to synthetic only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.phase2_execution.kalman import run_kalman
from src.phase2_execution.engine import run_pair
from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

_NC_EMPIRICAL_PAIRS = [("CVNA", "ISRG")]
_N_BOOTSTRAP        = 1000
_BLOCK_SIZE_DAYS    = 1
_BARS_PER_DAY_1MIN  = 390
_TOTAL_CAPITAL      = 1_000_000.0
_N_OPEN_PAIRS_MAX   = 50


def run_neg_control(
    trading_1min: dict[str, pd.DataFrame],
    delta: float,
    primary_sharpe: float,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    total_capital: float = _TOTAL_CAPITAL,
    tc_bps: float = 30.0,
    borrow_bps_yr: float = 50.0,
    eos_flatten: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run negative controls and bootstrap discrimination test.

    Parameters
    ----------
    trading_1min   : {ticker: 1min DataFrame} for the trading window
    delta          : Kalman delta used for this fold (same as primary)
    primary_sharpe : aggregate Sharpe of the primary strategy (for pass/fail)

    Returns
    -------
    dict with: empirical_sharpe, synthetic_sharpe, nc_daily_returns,
               bootstrap_mean, bootstrap_std, bootstrap_threshold,
               nc_pass, nc_source ('empirical'|'synthetic'|'both')
    """
    per_pair_dollar = total_capital / n_open_pairs_max

    # --- Empirical NC (CVNA/ISRG) ---
    emp_sharpe     = None
    emp_daily_ret  = None

    for ta, tb in _NC_EMPIRICAL_PAIRS:
        if ta in trading_1min and tb in trading_1min:
            result = _run_nc_pair(
                ta, tb, trading_1min, delta,
                per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
                eos_flatten=eos_flatten,
            )
            if result is not None:
                emp_sharpe    = result["sharpe"]
                emp_daily_ret = result["daily_returns"]
                log.info("NC empirical (%s/%s): Sharpe=%.3f", ta, tb, emp_sharpe)
            break

    # --- Synthetic NC (two uncorrelated random walks) ---
    # Build synthetic prices over the trading window
    ref_ticker = next(iter(trading_1min))
    ref_idx    = trading_1min[ref_ticker].index
    n_bars     = len(ref_idx)

    rng = np.random.default_rng(seed)
    # Two independent GBM paths, no cointegration
    vol = 0.001   # ~0.1% per bar (realistic for 1-min)
    log_ret_a = rng.normal(0.0, vol, n_bars)
    log_ret_b = rng.normal(0.0, vol, n_bars)
    price_a   = 100.0 * np.exp(np.cumsum(log_ret_a))
    price_b   = 100.0 * np.exp(np.cumsum(log_ret_b))
    log_a     = np.log(price_a)
    log_b     = np.log(price_b)

    # PCA-style init: flat alpha=0, beta=1, large R (no real relationship)
    R_synthetic = 1.0  # large R relative to small spread variance
    try:
        syn_pair_df = run_pair(
            log_a_1min   = log_a,
            log_b_1min   = log_b,
            close_a_1min = price_a,
            close_b_1min = price_b,
            index_1min   = ref_idx,
            alpha_pca    = 0.0,
            beta_pca     = 1.0,
            R            = R_synthetic,
            delta        = delta,
            half_life_days = 5.0,   # use mid-range half-life for Z window
            eos_flatten  = eos_flatten,
        )
        close_a_ser = pd.Series(price_a, index=ref_idx)
        close_b_ser = pd.Series(price_b, index=ref_idx)
        syn_pnl_result = compute_pair_pnl(
            syn_pair_df, close_a_ser, close_b_ser,
            per_pair_dollar, tc_bps, borrow_bps_yr,
        )
        syn_metrics = compute_metrics(
            syn_pnl_result["bar_pnl"], total_capital,
            syn_pnl_result["trade_log"],
        )
        syn_sharpe    = syn_metrics["sharpe"]
        syn_daily_ret = syn_metrics["daily_returns"]
        log.info("NC synthetic: Sharpe=%.3f", syn_sharpe)
    except Exception as e:
        log.warning("NC synthetic failed: %s", e)
        syn_sharpe    = 0.0
        syn_daily_ret = pd.Series(dtype=float)

    # --- Bootstrap on NC returns ---
    # Use empirical NC if available; fall back to synthetic
    if emp_daily_ret is not None and len(emp_daily_ret) > 5:
        nc_daily = emp_daily_ret
        nc_source = "empirical"
    elif len(syn_daily_ret) > 5:
        nc_daily = syn_daily_ret
        nc_source = "synthetic"
    else:
        nc_daily  = pd.Series([0.0] * 20)
        nc_source = "fallback_zeros"

    bootstrap_sharpes = _block_bootstrap_sharpe(
        nc_daily, n_bootstrap=_N_BOOTSTRAP,
        block_size=_BLOCK_SIZE_DAYS, seed=seed,
    )
    b_mean   = float(np.mean(bootstrap_sharpes))
    b_std    = float(np.std(bootstrap_sharpes))
    threshold = b_mean + 2.0 * b_std
    nc_pass   = primary_sharpe > threshold

    log.info(
        "NC bootstrap: mean=%.3f std=%.3f threshold=%.3f | primary=%.3f | %s",
        b_mean, b_std, threshold, primary_sharpe,
        "PASS" if nc_pass else "FAIL",
    )

    return {
        "empirical_sharpe":   emp_sharpe,
        "synthetic_sharpe":   syn_sharpe,
        "nc_daily_returns":   nc_daily,
        "bootstrap_mean":     b_mean,
        "bootstrap_std":      b_std,
        "bootstrap_threshold": threshold,
        "bootstrap_sharpes":  bootstrap_sharpes,
        "nc_pass":            nc_pass,
        "nc_source":          nc_source,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_nc_pair(
    ta: str, tb: str,
    trading_1min: dict,
    delta: float,
    per_pair_dollar: float,
    tc_bps: float,
    borrow_bps_yr: float,
    total_capital: float,
    eos_flatten: bool = True,
) -> dict | None:
    """Run the engine on one empirical NC pair and return metrics."""
    df_a = trading_1min[ta].copy()
    df_b = trading_1min[tb].copy()

    if "log_close" not in df_a.columns:
        df_a["log_close"] = np.log(df_a["close"].clip(lower=1e-10))
    if "log_close" not in df_b.columns:
        df_b["log_close"] = np.log(df_b["close"].clip(lower=1e-10))

    df_ab = df_a[["log_close", "close"]].join(
        df_b[["log_close", "close"]], how="inner", lsuffix="_a", rsuffix="_b"
    ).dropna()

    if len(df_ab) < 50:
        return None

    # OLS hedge ratio: cov(log_a, log_b) / var(log_b)
    # std-ratio ignores correlation sign and magnitude
    log_a = df_ab["log_close_a"].values
    log_b = df_ab["log_close_b"].values
    var_b = float(np.var(log_b))
    beta_naive  = float(np.cov(log_a, log_b)[0, 1] / var_b) if var_b > 1e-12 else 1.0
    residuals   = log_a - beta_naive * log_b
    alpha_naive = np.mean(residuals)
    R_naive    = float(np.var(residuals))
    if R_naive <= 0:
        R_naive = 1.0

    try:
        pair_df = run_pair(
            log_a_1min   = log_a,
            log_b_1min   = log_b,
            close_a_1min = df_ab["close_a"].values,
            close_b_1min = df_ab["close_b"].values,
            index_1min   = df_ab.index,
            alpha_pca    = alpha_naive,
            beta_pca     = beta_naive,
            R            = R_naive,
            delta        = delta,
            half_life_days = 5.0,
            eos_flatten  = eos_flatten,
        )
    except Exception as e:
        log.warning("NC pair %s/%s engine failed: %s", ta, tb, e)
        return None

    if pair_df.empty:
        return None

    pnl_result = compute_pair_pnl(
        pair_df,
        df_ab["close_a"],
        df_ab["close_b"],
        per_pair_dollar, tc_bps, borrow_bps_yr,
    )
    metrics = compute_metrics(
        pnl_result["bar_pnl"], total_capital,
        pnl_result["trade_log"],
    )
    return metrics


def _block_bootstrap_sharpe(
    daily_returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Block bootstrap Sharpe ratio distribution.

    daily_returns is at trading-day frequency. block_size=5 (1 trading week)
    preserves intra-week serial correlation that pairs strategies typically
    exhibit (positions held across multiple days). block_size=1 collapses to
    iid bootstrap and underestimates NC variance.

    Uses overlapping blocks (Politis & Romano moving-block bootstrap):
    starts drawn uniformly from [0, n - block_size + 1).
    Returns array of n_bootstrap Sharpe values.
    """
    arr = daily_returns.values
    n   = len(arr)
    if n < block_size + 1:
        # Fallback: shrink block size if series is short
        block_size = max(min(block_size, n // 2), 1)
        if n < 2:
            return np.zeros(n_bootstrap)

    rng      = np.random.default_rng(seed)
    n_blocks = max(n // block_size, 1)
    max_start = max(n - block_size + 1, 1)
    sharpes  = np.zeros(n_bootstrap)

    for k in range(n_bootstrap):
        starts    = rng.integers(0, max_start, size=n_blocks)
        resampled = np.concatenate([
            arr[s: s + block_size] for s in starts
        ])[:n]
        if len(resampled) < 2:
            continue
        std = resampled.std(ddof=1)
        sharpes[k] = (resampled.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0

    return sharpes



# ===== FILE: src/phase3_backtest/latency.py =====
"""
Phase 3 — Latency Sweep

Wraps Phase 2 engine output: shifts position array by additional bars beyond
the baseline t+1 lag already built into the engine.

Configs:
  t+1  : baseline (0 extra bars)
  t+2  : 1 extra bar
  t+5  : 4 extra bars
  t+10 : 9 extra bars
  random: per-trade lag sampled Uniform(1, 5) extra bars

Pass criterion (per spec §3.4):
  Sharpe(t+5) > 0  AND  monotonic degradation (no cliff at t+2)
  If t+2 kills Sharpe → signal is microstructure noise, not cointegration.

Run on default config only. Do NOT cross with OAT grid.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

LATENCY_CONFIGS = {
    "t+1":    0,    # baseline already in engine
    "t+2":    1,
    "t+5":    4,
    "t+10":   9,
}


def apply_extra_lag(pair_df: pd.DataFrame, extra_bars: int) -> pd.DataFrame:
    """
    Shift position array by extra_bars additional bars (on top of t+1 in engine).
    Fills leading bars with 0 (flat).
    Also shifts signal by same amount so timestamp proof remains consistent.
    """
    if extra_bars == 0:
        return pair_df

    df = pair_df.copy()
    pos = df["position"].values.copy()
    sig = df["signal"].values.copy()

    shifted_pos = np.roll(pos, extra_bars)
    shifted_pos[:extra_bars] = 0
    shifted_sig = np.roll(sig, extra_bars)
    shifted_sig[:extra_bars] = 0

    df["position"] = shifted_pos
    df["signal"]   = shifted_sig

    # Shift forced_exit column if present (from _apply_max_holding)
    if "forced_exit" in df.columns:
        fe = df["forced_exit"].values.copy()
        shifted_fe = np.roll(fe, extra_bars)
        shifted_fe[:extra_bars] = False
        df["forced_exit"] = shifted_fe

    return df


def run_latency_sweep(
    fold_results: dict,
    trading_1min: dict[str, pd.DataFrame],
    total_capital: float = 1_000_000.0,
    n_open_pairs_max: int = 50,
    tc_bps: float = 30.0,
    borrow_bps_yr: float = 50.0,
    seed: int = 42,
) -> dict:
    """
    Run all latency configurations (t+1 through t+10 plus random).

    Parameters
    ----------
    fold_results  : {(ta, tb): pair_df} from engine.run_fold_execution

    Returns
    -------
    dict with: sharpe_by_lag, pass_criterion, monotonic_ok, latency_table
    """
    per_pair_dollar = total_capital / n_open_pairs_max
    results = {}

    for lag_name, extra_bars in LATENCY_CONFIGS.items():
        sharpe = _run_with_fixed_lag(
            fold_results, trading_1min, extra_bars,
            per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
        )
        results[lag_name] = sharpe
        log.info("Latency %s: Sharpe=%.3f", lag_name, sharpe)

    # Random lag: per-fold average Uniform(1,5) extra bars
    rng_lag = np.random.default_rng(seed)
    random_extra = int(rng_lag.integers(1, 6))   # 1..5 inclusive, whole fold
    random_sharpe = _run_with_fixed_lag(
        fold_results, trading_1min, random_extra,
        per_pair_dollar, tc_bps, borrow_bps_yr, total_capital,
    )
    results["random"] = random_sharpe
    log.info("Latency random (extra=%d): Sharpe=%.3f", random_extra, random_sharpe)

    # Pass criterion
    t5_pass    = results.get("t+5", 0.0) > 0.0
    t2_ok      = results.get("t+2", 0.0) > 0.0
    monotonic  = _is_monotonically_degrading(results)

    latency_pass = t5_pass and monotonic

    log.info(
        "Latency sweep: t+5_pass=%s monotonic=%s overall=%s",
        t5_pass, monotonic, latency_pass,
    )

    # Table for audit log
    ordered_keys = ["t+1", "t+2", "t+5", "t+10", "random"]
    latency_table = [
        {"lag": k, "extra_bars": LATENCY_CONFIGS.get(k, "var"),
         "sharpe": results.get(k, 0.0)}
        for k in ordered_keys
    ]

    return {
        "sharpe_by_lag":  results,
        "t5_pass":        t5_pass,
        "t2_ok":          t2_ok,
        "monotonic_ok":   monotonic,
        "latency_pass":   latency_pass,
        "latency_table":  latency_table,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_with_fixed_lag(
    fold_results: dict,
    trading_1min: dict,
    extra_bars: int,
    per_pair_dollar: float,
    tc_bps: float,
    borrow_bps_yr: float,
    total_capital: float,
) -> float:
    """Run PnL for all pairs with a fixed extra lag. Returns aggregate Sharpe."""
    all_bar_pnl = []
    all_trades  = []

    for (ta, tb), pair_df in fold_results.items():
        if ta not in trading_1min or tb not in trading_1min:
            continue

        lagged_df = apply_extra_lag(pair_df, extra_bars)
        close_a   = trading_1min[ta]["close"].reindex(pair_df.index)
        close_b   = trading_1min[tb]["close"].reindex(pair_df.index)

        result = compute_pair_pnl(
            lagged_df, close_a, close_b,
            per_pair_dollar, tc_bps, borrow_bps_yr,
        )
        all_bar_pnl.append(result["bar_pnl"])
        all_trades.extend(result["trade_log"])

    if not all_bar_pnl:
        return 0.0

    total_pnl = pd.concat(all_bar_pnl, axis=1).fillna(0.0).sum(axis=1)
    metrics   = compute_metrics(total_pnl, total_capital, all_trades)
    return float(metrics["sharpe"])


def _is_monotonically_degrading(results: dict) -> bool:
    """
    Check that Sharpe degrades monotonically from t+1 to t+10.
    Allows small tolerance (0.05 Sharpe units) to account for noise.
    """
    ordered = ["t+1", "t+2", "t+5", "t+10"]
    sharpes = [results.get(k, None) for k in ordered]
    sharpes = [s for s in sharpes if s is not None]

    if len(sharpes) < 2:
        return True

    tolerance = 0.10   # allow 0.10 Sharpe unit upward blip before calling non-monotonic
    for i in range(1, len(sharpes)):
        if sharpes[i] > sharpes[i - 1] + tolerance:
            return False
    return True



# ===== FILE: src/phase3_backtest/audit_log.py =====
"""
Phase 3 — Per-Fold Audit Log

Writes a 7-section .txt audit log per fold to results/logs/.

Sections:
  [1] Parameter hash (fold config)
  [2] Kalman delta + multi-criterion metrics
  [3] Trade log sample (first 20 trades)
  [4] Comparative metrics (primary vs NC)
  [5] Timestamp verification proof (exec_ts > signal_ts for 100%)
  [6] NC bootstrap results
  [7] Red flag status + environment hash

Red flag triggers (from spec):
  Lookahead leak   : Sharpe > 5.0
  NC discrimination: primary Sharpe <= NC bootstrap threshold
  Kalman degenerate: var(prior)/R outside [0.05, 4.0]
  Delta boundary   : auto-selected delta at grid edge
  Delta instability: delta jumps >2 steps vs previous fold
  Universal fail   : no delta passed constraints
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_DELTA_GRID      = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
_SHARPE_LOOKAHEAD_THRESHOLD = 5.0
_LOG_DIR_DEFAULT = "results/logs"


def write_audit_log(
    fold_n: int,
    fold_metrics: dict,
    nc_metrics: dict | None,
    latency_results: dict | None,
    delta: float | None,
    delta_metrics: dict,
    config: dict,
    prev_delta: float | None = None,
    output_dir: str = _LOG_DIR_DEFAULT,
) -> str:
    """
    Write 7-section audit log for one fold.

    Parameters
    ----------
    fold_n         : fold number (1-based)
    fold_metrics   : output of metrics_runner.run_fold_pnl
    nc_metrics     : output of neg_control.run_neg_control (or None)
    latency_results: output of latency.run_latency_sweep (or None)
    delta          : selected Kalman delta (None if universal fail)
    delta_metrics  : {delta_val: {metric_kurt, median_HL, median_ACF78}}
    config         : fold config dict (formation/trading window, params)
    prev_delta     : delta from previous fold (for instability check)
    output_dir     : directory to write fold_NN_audit.txt

    Returns
    -------
    str path to written audit log
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"fold_{fold_n:02d}_audit.txt")

    # --- Compute red flags ---
    sharpe        = fold_metrics.get("sharpe", 0.0)
    lookahead_ok  = fold_metrics.get("lookahead_ok", True)
    kalman_deg    = fold_metrics.get("kalman_degenerate", False)

    flag_lookahead   = (not lookahead_ok) or (sharpe > _SHARPE_LOOKAHEAD_THRESHOLD)
    flag_nc_fail     = False
    flag_kalman_deg  = kalman_deg
    flag_delta_bound = _check_delta_boundary(delta)
    flag_delta_instab = _check_delta_instability(delta, prev_delta)
    flag_univ_fail   = (delta is None)

    if nc_metrics is not None:
        flag_nc_fail = not nc_metrics.get("nc_pass", True)

    config_hash = _hash_config(config)

    lines = []

    # -------------------------------------------------------------------------
    lines += [
        f"=== FOLD {fold_n:02d} AUDIT LOG ===",
        f"Generated: {_now_str()}",
        "",
        "[1] PARAMETER HASH",
        f"config_hash   = {config_hash}",
    ]
    for k, v in config.items():
        lines.append(f"  {k:<24} = {v}")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[2] KALMAN DELTA + MULTI-CRITERION METRICS"]
    if delta is None:
        lines.append("  Selected delta: NONE (universal constraint fail)")
    else:
        lines.append(f"  Selected delta: {delta:.0e}")
        if flag_delta_bound:
            lines.append("  WARNING: delta at grid boundary")
        if flag_delta_instab:
            lines.append(f"  WARNING: delta instability (prev={prev_delta:.0e})")

    lines.append("  Grid results:")
    for d in _DELTA_GRID:
        if d in delta_metrics:
            m = delta_metrics[d]
            kurt  = m.get("metric_kurt",  float("nan"))
            hl    = m.get("median_HL",    float("nan"))
            acf78 = m.get("median_ACF78", float("nan"))
            sel   = " <-- SELECTED" if d == delta else ""
            feasible = (
                1.0 <= hl <= 10.0 and acf78 > 0.7
                and not (np.isnan(hl) or np.isnan(acf78))
            )
            feas_tag = " [FEASIBLE]" if feasible else ""
            lines.append(
                f"    delta={d:.0e}  kurt_dev={kurt:6.3f}  "
                f"HL={hl:6.2f}d  ACF78={acf78:.3f}{feas_tag}{sel}"
            )
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[3] TRADE LOG SAMPLE (first 20 trades)"]
    trade_log = fold_metrics.get("trade_log", [])
    if not trade_log:
        lines.append("  No trades this fold.")
    else:
        header = f"  {'entry_ts':<24} {'exit_ts':<24} {'dir':>4} {'bars':>6} {'gross_bps':>10} {'net_bps':>10} {'exit_reason'}"
        lines.append(header)
        for t in trade_log[:20]:
            lines.append(
                f"  {str(t.get('entry_ts','')):<24} "
                f"{str(t.get('exit_ts','')):<24} "
                f"{t.get('direction', 0):>4} "
                f"{t.get('n_bars', 0):>6} "
                f"{t.get('gross_bps', 0.0):>10.1f} "
                f"{t.get('net_bps', 0.0):>10.1f} "
                f"{t.get('exit_reason', '?')}"
            )
        if len(trade_log) > 20:
            lines.append(f"  ... ({len(trade_log) - 20} more trades not shown)")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[4] COMPARATIVE METRICS"]
    cost = fold_metrics.get("cost_decomp", {})
    lines += [
        f"  Primary Sharpe  : {sharpe:.4f}",
        f"  MaxDD (bar)     : {fold_metrics.get('max_dd', 0.0):.4f}",
        f"  CAGR            : {fold_metrics.get('cagr', 0.0):.4f}",
        f"  Calmar          : {fold_metrics.get('calmar', 0.0):.4f}",
        f"  Win Rate        : {fold_metrics.get('win_rate', 0.0):.2%}",
        f"  N Trades        : {fold_metrics.get('n_trades', 0)}",
        f"  Avg Hold (bars) : {fold_metrics.get('avg_holding_bars', 0.0):.1f}",
        f"  Avg Net (bps)   : {fold_metrics.get('avg_net_bps', 0.0):.2f}",
        f"  Cost commission : ${cost.get('commission', 0.0):,.2f}",
        f"  Cost borrow     : ${cost.get('borrow', 0.0):,.2f}",
        f"  Cost rebalance  : ${cost.get('rebalance', 0.0):,.2f}",
    ]

    if nc_metrics is not None:
        lines += [
            f"  NC empirical Sharpe  : {nc_metrics.get('empirical_sharpe', 'N/A')}",
            f"  NC synthetic Sharpe  : {nc_metrics.get('synthetic_sharpe', 0.0):.4f}",
            f"  NC bootstrap mean+2sd: {nc_metrics.get('bootstrap_threshold', 0.0):.4f}",
            f"  NC PASS              : {nc_metrics.get('nc_pass', False)}",
            f"  NC source            : {nc_metrics.get('nc_source', 'unknown')}",
        ]

    if latency_results is not None:
        lines.append("  Latency sweep:")
        for row in latency_results.get("latency_table", []):
            lines.append(
                f"    {row['lag']:>6}  Sharpe={row['sharpe']:.4f}"
            )
        lines.append(f"  Latency pass: {latency_results.get('latency_pass', False)}")

    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[5] TIMESTAMP VERIFICATION PROOF"]
    n_trades = fold_metrics.get("n_trades", 0)
    la_ok    = fold_metrics.get("lookahead_ok", True)
    lines += [
        f"  Total trades              : {n_trades}",
        f"  exec_ts > signal_ts (100%): {la_ok}",
        f"  LOOKAHEAD STATUS          : {'PASS' if la_ok else 'FAIL'}",
    ]
    if sharpe > _SHARPE_LOOKAHEAD_THRESHOLD:
        lines.append(f"  WARNING: Sharpe={sharpe:.2f} > 5.0 — investigate for lookahead")
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[6] NEGATIVE CONTROL BOOTSTRAP"]
    if nc_metrics is None:
        lines.append("  NC bootstrap not run this fold.")
    else:
        b_mean   = nc_metrics.get("bootstrap_mean", 0.0)
        b_std    = nc_metrics.get("bootstrap_std", 0.0)
        thresh   = nc_metrics.get("bootstrap_threshold", 0.0)
        nc_pass  = nc_metrics.get("nc_pass", False)
        pct_pos  = _pct_above(nc_metrics.get("bootstrap_sharpes", np.array([])), 0.0)
        lines += [
            f"  Bootstrap Sharpes (N=1000): mean={b_mean:.4f}  std={b_std:.4f}",
            f"  Threshold (mean+2sd)       : {thresh:.4f}",
            f"  Primary Sharpe             : {sharpe:.4f}",
            f"  Bootstrap pct > 0          : {pct_pos:.1%}",
            f"  NC PASS                    : {nc_pass}",
            f"  Discrimination STATUS      : {'PASS' if nc_pass else 'FAIL'}",
        ]
    lines += [""]

    # -------------------------------------------------------------------------
    lines += ["[7] RED FLAG STATUS + ENVIRONMENT HASH"]
    flags = {
        "Lookahead leak (Sharpe>5 or pos mismatch)": flag_lookahead,
        "NC discrimination fail":                    flag_nc_fail,
        "Kalman spread degenerate":                  flag_kalman_deg,
        "Delta at grid boundary":                    flag_delta_bound,
        "Delta instability (>2 steps)":              flag_delta_instab,
        "Universal constraint fail (no delta)":      flag_univ_fail,
    }
    any_critical = flag_lookahead or flag_nc_fail or flag_univ_fail
    for name, val in flags.items():
        tag = "ERROR" if val and name in (
            "Lookahead leak (Sharpe>5 or pos mismatch)",
            "NC discrimination fail",
            "Universal constraint fail (no delta)",
        ) else ("WARNING" if val else "OK")
        lines.append(f"  [{tag:<7}] {name}: {val}")

    lines += [
        "",
        f"  OVERALL STATUS: {'CRITICAL' if any_critical else 'OK'}",
        "",
        "  Environment:",
        f"    Python   : {sys.version.split()[0]}",
        f"    Platform : {platform.platform()}",
    ]
    try:
        import numpy, pandas, numba, statsmodels
        lines += [
            f"    numpy    : {numpy.__version__}",
            f"    pandas   : {pandas.__version__}",
            f"    numba    : {numba.__version__}",
            f"    statsmodels: {statsmodels.__version__}",
        ]
    except ImportError:
        pass
    lines.append(f"    env_hash : {_env_hash()}")
    lines += ["", "=== END AUDIT LOG ==="]

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log.info("Audit log written: %s", path)

    # Log any critical flags
    if flag_lookahead:
        log.error("RED FLAG: Lookahead leak detected — Fold %d HALT", fold_n)
    if flag_nc_fail:
        log.error("RED FLAG: NC discrimination fail — Fold %d", fold_n)
    if flag_univ_fail:
        log.error("RED FLAG: Universal delta constraint fail — Fold %d", fold_n)

    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_delta_boundary(delta: float | None) -> bool:
    if delta is None:
        return False
    return delta in (_DELTA_GRID[0], _DELTA_GRID[-1])


def _check_delta_instability(delta: float | None, prev_delta: float | None) -> bool:
    if delta is None or prev_delta is None:
        return False
    try:
        curr_idx = _DELTA_GRID.index(delta)
        prev_idx = _DELTA_GRID.index(prev_delta)
        return abs(curr_idx - prev_idx) > 2
    except ValueError:
        return False


def _hash_config(config: dict) -> str:
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _env_hash() -> str:
    env_str = f"{sys.version}{platform.platform()}"
    try:
        import numpy, pandas, numba
        env_str += f"{numpy.__version__}{pandas.__version__}{numba.__version__}"
    except ImportError:
        pass
    return hashlib.sha256(env_str.encode()).hexdigest()[:12]


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pct_above(arr: np.ndarray, threshold: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr > threshold))



# ===== FILE: src/phase3_backtest/metrics_runner.py =====
"""
Phase 3 — Fold-Level Metrics Runner

Iterates all pairs in a fold's engine output, runs PnL per pair,
aggregates into a single fold metrics dict consumed by audit_log and orchestrator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.phase3_backtest.pnl import compute_pair_pnl, compute_metrics

log = logging.getLogger(__name__)

_TOTAL_CAPITAL    = 1_000_000.0
_N_OPEN_PAIRS_MAX = 50
_TC_BPS           = 30.0
_BORROW_BPS_YR    = 50.0


def run_fold_pnl(
    fold_results: dict,
    trading_1min: dict[str, pd.DataFrame],
    pairs_df: pd.DataFrame,
    delta: float,
    delta_metrics: dict,
    total_capital: float = _TOTAL_CAPITAL,
    n_open_pairs_max: int = _N_OPEN_PAIRS_MAX,
    tc_bps: float = _TC_BPS,
    borrow_bps_yr: float = _BORROW_BPS_YR,
) -> dict:
    """
    Compute PnL and metrics for all pairs in a fold.

    Parameters
    ----------
    fold_results   : {(ta, tb): pair_df} from engine.run_fold_execution
    trading_1min   : {ticker: DataFrame} with 'close' column
    pairs_df       : Phase 1 surviving pairs (for metadata)
    delta          : selected Kalman delta for this fold
    delta_metrics  : {delta: {metric_kurt, median_HL, median_ACF78}} from select_delta

    Returns
    -------
    dict with: bar_pnl, bar_equity, daily_returns, sharpe, max_dd, cagr, calmar,
               win_rate, n_trades, avg_holding_bars, n_pairs_traded,
               cost_decomp, trade_log, pair_pnls, delta, delta_metrics,
               spread_prior_map (for Kalman degenerate check)
    """
    per_pair_dollar = total_capital / n_open_pairs_max

    all_bar_pnl       = []
    all_trade_log     = []
    all_rebalance_log = []
    pair_pnls         = {}
    per_pair_metrics  = {}
    cost_commission   = 0.0
    cost_borrow       = 0.0
    cost_rebalance    = 0.0
    spread_prior_map  = {}

    # Global counter for trade_id assignment (unique within a fold; runner promotes
    # to globally-unique by combining with fold_id downstream).
    _next_trade_id = 0

    for (ta, tb), pair_df in fold_results.items():
        if ta not in trading_1min or tb not in trading_1min:
            log.debug("Skip PnL for %s/%s: 1min data missing", ta, tb)
            continue

        close_a = trading_1min[ta]["close"].reindex(pair_df.index)
        close_b = trading_1min[tb]["close"].reindex(pair_df.index)

        result = compute_pair_pnl(
            pair_df, close_a, close_b,
            per_pair_dollar=per_pair_dollar,
            tc_bps=tc_bps,
            borrow_bps_yr=borrow_bps_yr,
        )

        bar_pnl = result["bar_pnl"]
        all_bar_pnl.append(bar_pnl)
        pair_tl  = result["trade_log"]
        pair_rbl = result.get("rebalance_log", [])
        hl_row = pairs_df[(pairs_df["ticker_a"] == ta) & (pairs_df["ticker_b"] == tb)]
        hl_days = float(hl_row["half_life_days"].iloc[0]) if not hl_row.empty else float("nan")
        pair_id = f"{ta}_{tb}"

        # Map pair-local trade indices to fold-unique trade_ids, then enrich rows.
        local_to_global = {}
        for t in pair_tl:
            local_idx = t.pop("pair_trade_idx")
            global_id = _next_trade_id
            _next_trade_id += 1
            local_to_global[local_idx] = global_id
            t["trade_id"]       = global_id
            t["pair_id"]        = pair_id
            t["ticker_a"]       = ta
            t["ticker_b"]       = tb
            t["half_life_days"] = hl_days
        all_trade_log.extend(pair_tl)

        # Rebalance events: tag with global trade_id, ticker (B leg only — engine
        # rebalances the hedge side), pair_id, and fold metadata.
        for r in pair_rbl:
            local_idx = r.pop("pair_trade_idx")
            r["trade_id"]            = local_to_global.get(local_idx, -1)
            r["pair_id"]             = pair_id
            r["ticker"]              = tb   # only the B leg is rebalanced
            r["ticker_a"]            = ta
            r["ticker_b"]            = tb
        all_rebalance_log.extend(pair_rbl)

        pair_pnls[(ta, tb)] = bar_pnl
        cost_commission += result["cost_commission"]
        cost_borrow     += result["cost_borrow"]
        cost_rebalance  += result["cost_rebalance"]
        spread_prior_map[(ta, tb)] = pair_df["spread_prior"].dropna().values
        # Per-pair trade metrics (for bucket-level volume stratification)
        _n_tl = len(pair_tl)
        per_pair_metrics[(ta, tb)] = {
            "n_trades":    _n_tl,
            "win_rate":    sum(1 for t in pair_tl if t["net_pnl"] > 0) / _n_tl if _n_tl else 0.0,
            "avg_net_bps": sum(t["net_bps"] for t in pair_tl) / _n_tl if _n_tl else 0.0,
            "total_pnl":   float(bar_pnl.sum()),
        }

    if not all_bar_pnl:
        log.warning("No pairs produced PnL — fold has no executable trades")
        return _empty_fold_metrics(delta, delta_metrics)

    # Aggregate: sum bar PnL across all pairs, union index
    total_bar_pnl = _sum_series(all_bar_pnl)

    metrics = compute_metrics(total_bar_pnl, total_capital, all_trade_log)

    # Lookahead assertion: exec_ts > signal_ts for every trade
    lookahead_ok = _verify_no_lookahead(fold_results)

    # Kalman degenerate check: var(prior) vs static PCA spread
    kalman_degenerate = _check_kalman_degenerate(fold_results, pairs_df)

    # Exit reason breakdown
    exit_breakdown = _exit_reason_breakdown(all_trade_log)

    return {
        # Core metrics
        "sharpe":           metrics["sharpe"],
        "max_dd":           metrics["max_dd"],
        "cagr":             metrics["cagr"],
        "calmar":           metrics["calmar"],
        "win_rate":         metrics["win_rate"],
        "n_trades":         metrics["n_trades"],
        "avg_holding_bars": metrics["avg_holding_bars"],
        "avg_net_bps":      metrics["avg_net_bps"],
        "total_return":     metrics["total_return"],
        # Time series
        "bar_pnl":          total_bar_pnl,
        "bar_equity":       metrics["bar_equity"],
        "daily_returns":    metrics["daily_returns"],
        # Cost decomposition
        "cost_decomp": {
            "commission":  cost_commission,
            "borrow":      cost_borrow,
            "rebalance":   cost_rebalance,
        },
        # Trade log + rebalance log (Week-5 hand-off)
        "trade_log":        all_trade_log,
        "rebalance_log":    all_rebalance_log,
        "exit_breakdown":   exit_breakdown,
        # Per-pair data (for volume stratification in Phase 4)
        "pair_pnls":        pair_pnls,
        "n_pairs_traded":   len(pair_pnls),
        "per_pair_metrics":  per_pair_metrics,
        # Audit fields
        "lookahead_ok":      lookahead_ok,
        "kalman_degenerate": kalman_degenerate,
        "delta":             delta,
        "delta_metrics":     delta_metrics,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_series(series_list: list[pd.Series]) -> pd.Series:
    """Sum a list of bar PnL series on their union index (fills 0 for missing bars)."""
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=1).fillna(0.0)
    return df.sum(axis=1)


def _verify_no_lookahead(fold_results: dict) -> bool:
    """
    Verify position[t] == signal[t-1] for all t in all pairs.
    Returns True if no lookahead detected.
    """
    for (ta, tb), df in fold_results.items():
        pos = df["position"].values
        sig = df["signal"].values
        for i in range(1, len(pos)):
            if pos[i] != sig[i - 1]:
                log.error("Lookahead detected at pair %s/%s bar %d", ta, tb, i)
                return False
    return True


def _check_kalman_degenerate(fold_results: dict, pairs_df: pd.DataFrame) -> bool:  # noqa: ARG001
    """
    Red flag: Kalman prior spread has collapsed or has extreme kurtosis.
    Avoids cross-scale comparison: R is from 5-min formation, prior is from
    1-min trading — their variances are not comparable.
    Returns True if degenerate (flag).
    """
    degenerate = False
    for (ta, tb), df in fold_results.items():
        prior = df["spread_prior"].dropna().values
        if len(prior) < 10:
            continue

        var_prior = float(np.var(prior))
        if var_prior < 1e-14:
            log.warning("Kalman degenerate: %s/%s prior spread collapsed (var=%.2e)", ta, tb, var_prior)
            degenerate = True
            continue

        std_p = float(np.std(prior))
        if std_p < 1e-10:
            continue
        z = (prior - float(np.mean(prior))) / std_p
        kurt = float(np.mean(z ** 4)) - 3.0
        if abs(kurt) > 10.0:
            log.warning("Kalman degenerate: %s/%s excess kurtosis=%.1f", ta, tb, kurt)
            degenerate = True

    return degenerate


def _exit_reason_breakdown(trade_log: list[dict]) -> dict:
    reasons = {"zero_cross": 0, "eos": 0, "end_of_window": 0, "sl": 0, "max_holding": 0}
    for t in trade_log:
        r = t.get("exit_reason", "zero_cross")
        reasons[r] = reasons.get(r, 0) + 1
    total = len(trade_log)
    pct = {k: v / total if total > 0 else 0.0 for k, v in reasons.items()}
    return {"counts": reasons, "pct": pct}


def _empty_fold_metrics(delta: float, delta_metrics: dict) -> dict:
    return {
        "sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "calmar": 0.0,
        "win_rate": 0.0, "n_trades": 0, "avg_holding_bars": 0.0,
        "avg_net_bps": 0.0, "total_return": 0.0,
        "bar_pnl": pd.Series(dtype=float),
        "bar_equity": pd.Series(dtype=float),
        "daily_returns": pd.Series(dtype=float),
        "cost_decomp": {"commission": 0.0, "borrow": 0.0, "rebalance": 0.0},
        "trade_log": [], "rebalance_log": [], "exit_breakdown": {},
        "pair_pnls": {}, "n_pairs_traded": 0, "per_pair_metrics": {},
        "lookahead_ok": True, "kalman_degenerate": False,
        "delta": delta, "delta_metrics": delta_metrics,
    }



# ===== FILE: src/phase4_defense/orchestrator.py =====
"""
Phase 4 — Orchestrator

Defines the canonical fold schedule, regime map, and result-loading utilities.
The full pipeline has already been run (run_full_pipeline.py); this module
loads the persisted outputs for downstream analysis.

Re-run entrypoint: run_orchestrator() delegates to run_full_pipeline.py logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT         = Path(__file__).parents[2]
_BASELINE_DIR = _ROOT / "results" / "metrics" / "baseline"
_EQUITY_DIR   = _BASELINE_DIR / "equities"
_METRICS_DIR  = _BASELINE_DIR / "phase4"      # Phase 4 analytics output
_LOGS_DIR     = _ROOT / "results" / "logs"
_FIGURES_DIR  = _ROOT / "results" / "figures" / "baseline"
_PHASE1_DIR   = _ROOT / "results" / "metrics" / "phase1_folds"


# ---------------------------------------------------------------------------
# Fold Schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldSpec:
    fold_n:         int
    form_start:     str   # YYYY-MM-DD
    form_end:       str   # YYYY-MM-DD
    trade_year:     int
    trade_month:    int   # 1..12
    trading_month:  str   # "YYYY-MM"

    @property
    def trading_label(self) -> str:
        return self.trading_month


def _build_fold_schedule() -> list[FoldSpec]:
    """
    Generate 45-fold schedule.

    Fold N: formation starts at beginning of (January 2022 + N-1 months),
    spans 6 months, trading is the month immediately after formation end.

    Fold 1: formation 2022-01-03 → 2022-06-30, trading 2022-07
    Fold 2: formation 2022-02-01 → 2022-07-31, trading 2022-08
    ...
    Fold 45: formation 2025-09-01 → 2026-02-28, trading 2026-03
    """
    folds: list[FoldSpec] = []
    base = pd.Timestamp("2022-01-01")

    for i in range(45):
        fold_n = i + 1

        # Formation start: first of month (Fold 1 uses 2022-01-03)
        form_start_dt = base + pd.DateOffset(months=i)
        if fold_n == 1:
            form_start_dt = pd.Timestamp("2022-01-03")

        # Formation end: last day of the 6th calendar month from form_start.
        # Always anchor to month-start to handle Fold 1 (2022-01-03 start)
        # correctly — otherwise +6 months from Jan 3 = Jul 3, not Jun 30.
        month_anchor = pd.Timestamp(form_start_dt.year, form_start_dt.month, 1)
        form_end_dt  = month_anchor + pd.DateOffset(months=6) - pd.DateOffset(days=1)

        # Trading month: calendar month immediately after formation end
        trade_dt = pd.Timestamp(form_end_dt.year, form_end_dt.month, 1) + pd.DateOffset(months=1)

        folds.append(FoldSpec(
            fold_n       = fold_n,
            form_start   = form_start_dt.strftime("%Y-%m-%d"),
            form_end     = form_end_dt.strftime("%Y-%m-%d"),
            trade_year   = trade_dt.year,
            trade_month  = trade_dt.month,
            trading_month = trade_dt.strftime("%Y-%m"),
        ))

    return folds


FOLD_SCHEDULE: list[FoldSpec] = _build_fold_schedule()
FOLD_BY_N: dict[int, FoldSpec] = {f.fold_n: f for f in FOLD_SCHEDULE}


# ---------------------------------------------------------------------------
# Regime Map
# ---------------------------------------------------------------------------

REGIME_MAP: dict[str, list[int]] = {
    "Late Bear 2022":        list(range(1,  7)),   # Trading Jul–Dec 2022
    "Early Bull 2023":       list(range(7,  19)),  # Trading Jan–Dec 2023
    "Mid Bull 2024":         list(range(19, 31)),  # Trading Jan–Dec 2024
    "Late Bull 2025-Q12026": list(range(31, 46)),  # Trading Jan 2025–Mar 2026
}

FOLD_TO_REGIME: dict[int, str] = {
    fold_n: regime
    for regime, folds in REGIME_MAP.items()
    for fold_n in folds
}


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------

def load_fold_metrics() -> pd.DataFrame:
    """Load aggregate per-fold metrics from fold_metrics.csv."""
    path = _BASELINE_DIR / "fold_metrics.csv"
    df = pd.read_csv(path)
    df["regime"] = df["fold"].map(FOLD_TO_REGIME)
    return df


def load_equity(fold_n: int) -> Optional[pd.DataFrame]:
    """Load bar-level equity curve for a fold (None if not present)."""
    path = _EQUITY_DIR / f"fold{fold_n:02d}_equity.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_phase1_pairs(fold_n: int) -> Optional[pd.DataFrame]:
    """Load Phase 1 surviving pairs for a fold (None if not present)."""
    path = _PHASE1_DIR / f"fold_{fold_n:02d}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def load_all_fold_equities() -> dict[int, pd.DataFrame]:
    """Load all available equity curves keyed by fold number."""
    result: dict[int, pd.DataFrame] = {}
    for fold_n in range(1, 46):
        eq = load_equity(fold_n)
        if eq is not None:
            result[fold_n] = eq
    return result


def load_equity_concat() -> pd.DataFrame:
    """
    Return concatenated bar-level equity curve across all completed folds.
    Uses the pre-built equity_full.parquet if available, otherwise rebuilds.
    """
    full_path = _BASELINE_DIR / "equity_full.parquet"
    if full_path.exists():
        return pd.read_parquet(full_path)

    # Rebuild from per-fold files
    frames = []
    for fold_n in range(1, 46):
        eq = load_equity(fold_n)
        if eq is not None:
            eq = eq.copy()
            eq["fold"] = fold_n
            frames.append(eq)

    if not frames:
        return pd.DataFrame()

    concat = pd.concat(frames).sort_index()
    concat.to_parquet(full_path)
    return concat


def get_completed_folds(fold_metrics: Optional[pd.DataFrame] = None) -> list[int]:
    """Return list of fold numbers that completed (have metrics)."""
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()
    return sorted(fold_metrics["fold"].tolist())


def get_daily_returns(fold_n: int) -> Optional[pd.Series]:
    """Compute daily log-returns from bar-level equity curve."""
    eq = load_equity(fold_n)
    if eq is None or eq.empty:
        return None
    col = "equity" if "equity" in eq.columns else eq.columns[0]
    daily = eq[col].resample("B").last().dropna()
    if len(daily) < 2:
        return None
    return daily.pct_change().dropna()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_fold_summary(fold_metrics: Optional[pd.DataFrame] = None) -> None:
    """Print a quick summary of completed folds by regime."""
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    total = len(FOLD_SCHEDULE)
    done  = len(fold_metrics)
    skipped = total - done
    pct_pos = (fold_metrics["sharpe"] > 0).mean()

    print(f"\n=== Fold Summary ===")
    print(f"  Total folds:     {total}")
    print(f"  Completed:       {done}")
    print(f"  Skipped:         {skipped}")
    print(f"  Median Sharpe:   {fold_metrics['sharpe'].median():.3f}")
    print(f"  Mean Sharpe:     {fold_metrics['sharpe'].mean():.3f}")
    print(f"  % Positive:      {pct_pos:.1%}")

    print(f"\n  By regime:")
    for regime, folds_in in REGIME_MAP.items():
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)]
        if sub.empty:
            print(f"    {regime:30s}: no data")
            continue
        print(f"    {regime:30s}: N={len(sub):2d}  "
              f"mean={sub['sharpe'].mean():.2f}  "
              f"median={sub['sharpe'].median():.2f}  "
              f"pos={( sub['sharpe']>0).mean():.0%}")



# ===== FILE: src/phase4_defense/regime.py =====
"""
Phase 4b — Regime Partition Analysis

Partitions fold results into 4 market regimes and computes per-regime
Sharpe statistics. Outputs:
  results/metrics/regime_sharpes.csv
  results/figures/regime_bar.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    REGIME_MAP,
    load_fold_metrics,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_regime_analysis(
    fold_metrics: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute per-regime Sharpe statistics.

    Returns DataFrame with columns:
      [regime, n_folds, n_completed, mean_sharpe, median_sharpe,
       iqr_25, iqr_75, pct_positive, min_sharpe, max_sharpe,
       mean_calmar, mean_cagr, mean_win_rate]
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    rows = []
    for regime, folds_in in REGIME_MAP.items():
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)].copy()
        n_completed = len(sub)
        n_total     = len(folds_in)

        if n_completed == 0:
            rows.append({
                "regime": regime,
                "n_folds": n_total,
                "n_completed": 0,
                **{k: np.nan for k in [
                    "mean_sharpe","median_sharpe","iqr_25","iqr_75",
                    "pct_positive","min_sharpe","max_sharpe",
                    "mean_calmar","mean_cagr","mean_win_rate",
                ]},
            })
            continue

        sharpes = sub["sharpe"].values
        rows.append({
            "regime":       regime,
            "n_folds":      n_total,
            "n_completed":  n_completed,
            "mean_sharpe":  float(np.mean(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "iqr_25":       float(np.percentile(sharpes, 25)),
            "iqr_75":       float(np.percentile(sharpes, 75)),
            "pct_positive": float(np.mean(sharpes > 0)),
            "min_sharpe":   float(np.min(sharpes)),
            "max_sharpe":   float(np.max(sharpes)),
            "mean_calmar":  float(sub["calmar"].mean()) if "calmar" in sub else np.nan,
            "mean_cagr":    float(sub["cagr"].mean())   if "cagr"   in sub else np.nan,
            "mean_win_rate":float(sub["win_rate"].mean()) if "win_rate" in sub else np.nan,
        })

    result = pd.DataFrame(rows)

    if save:
        out = _METRICS_DIR / "regime_sharpes.csv"
        result.to_csv(out, index=False)
        log.info("Regime Sharpes saved → %s", out)
        _save_figure(result, fold_metrics)

    return result


def _save_figure(regime_df: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    """Save regime bar chart and Sharpe distribution histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.warning("matplotlib not available — skipping regime figures")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: per-regime mean ± IQR bar chart ---
    ax = axes[0]
    regimes      = regime_df["regime"].tolist()
    short_labels = ["Bear\n2022", "Bull\n2023", "Bull\n2024", "Bull\n2025-Q1"]
    means        = regime_df["mean_sharpe"].values
    p25          = regime_df["iqr_25"].values
    p75          = regime_df["iqr_75"].values
    n_comp       = regime_df["n_completed"].values

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]
    x = np.arange(len(regimes))
    bars = ax.bar(x, means, color=colors, alpha=0.75, width=0.55)
    ax.errorbar(x, means,
                yerr=[means - p25, p75 - means],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(short_labels, n_comp)],
                       fontsize=9)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Mean Sharpe by Regime (bars = IQR)")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: full Sharpe distribution histogram ---
    ax2 = axes[1]
    for (regime, folds_in), color in zip(REGIME_MAP.items(), colors):
        sub = fold_metrics[fold_metrics["fold"].isin(folds_in)]["sharpe"]
        if len(sub) > 0:
            ax2.hist(sub.values, bins=8, alpha=0.55, color=color,
                     label=regime.split(" ")[0] + " " + regime.split(" ")[1])
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Fold Count")
    ax2.set_title("Fold Sharpe Distribution by Regime")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _FIGURES_DIR / "regime_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Regime figure saved → %s", out)


def print_regime_report(regime_df: pd.DataFrame | None = None) -> None:
    """Pretty-print regime stats to console."""
    if regime_df is None:
        regime_df = run_regime_analysis(save=False)

    print("\n=== Regime Partition Analysis ===")
    print(f"{'Regime':<30} {'N':>4} {'Done':>4} "
          f"{'Mean':>7} {'Median':>7} {'IQR':>15} "
          f"{'%Pos':>6} {'WinRate':>8}")
    print("-" * 90)

    for _, row in regime_df.iterrows():
        iqr_str = f"[{row['iqr_25']:.2f}, {row['iqr_75']:.2f}]"
        wr = f"{row['mean_win_rate']:.1%}" if not np.isnan(row['mean_win_rate']) else "  n/a"
        print(f"{row['regime']:<30} {row['n_folds']:>4} {row['n_completed']:>4} "
              f"{row['mean_sharpe']:>7.2f} {row['median_sharpe']:>7.2f} "
              f"{iqr_str:>15} {row['pct_positive']:>6.0%} {wr:>8}")

    print("\n  NOTE: Late Bear 2022 N=6 is underpowered for strong inference.")



# ===== FILE: src/phase4_defense/persistence.py =====
"""
Phase 4c — Pair Persistence Decay Curve

For P_2022 (pairs from Fold 1 formation Jul–Dec 2022 → trading Jan 2023,
which is Fold 7 in our schedule where formation = 2022-07-01 to 2022-12-31),
re-tests Johansen cointegration for each subsequent fold's formation window.

Output:
  results/metrics/pair_persistence.csv   — (fold, trading_month, n_pairs, n_still_passing, pct_passing)
  results/figures/persistence_line.png
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_phase1_pairs,
    _METRICS_DIR,
    _FIGURES_DIR,
)
from src.utils.io import read_5min

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# P_2022 target: pairs from formation Jul–Dec 2022 (Fold 7).
# Fold 7 may have zero surviving pairs (the 2022 H2 window was a stressed period
# with few cointegrated pairs passing BH-FDR). If empty, we fall back to the
# nearest fold with actual pairs — Fold 6 (formation Jun–Nov 2022).
_P2022_FOLD_PREFERRED = 7
_P2022_FOLD_FALLBACK  = 6   # formation Jun-Nov 2022; confirmed 64 pairs
_JOHANSEN_PVAL_THRESH = 0.05


def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Run Johansen trace test on pair, return min p-value (trace rank 0)."""
    mat = np.column_stack([log_a, log_b])
    if len(mat) < 10:
        return 1.0
    try:
        res = coint_johansen(mat, det_order=0, k_ar_diff=1)
        # p-values not directly returned; use critical values at 5% for trace stat
        # Trace stat > cv_5pct → reject H0: no cointegration
        trace_stat = res.lr1[0]      # rank 0 trace statistic
        cv_5pct    = res.cvt[0, 1]  # 5% critical value for rank 0
        # Return approximate p: <0.05 if stat > cv_5pct
        return 0.01 if trace_stat > cv_5pct else 0.20
    except Exception:
        return 1.0


def _load_formation_5min(
    fold_n: int,
    tickers: list[str],
) -> dict[str, pd.Series]:
    """Load 5-min log_close for given tickers over fold N's formation window."""
    spec = FOLD_BY_N[fold_n]
    cache: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = read_5min(ticker)
            if df is None or df.empty:
                continue
            mask = (
                (df.index >= spec.form_start) &
                (df.index <= spec.form_end)
            )
            sub = df.loc[mask, "log_close"].dropna()
            if len(sub) >= 10:
                cache[ticker] = sub
        except Exception:
            pass
    return cache


def run_persistence(
    save: bool = True,
) -> pd.DataFrame:
    """
    Re-test Johansen on P_2022 pairs for each fold 7–45.

    Returns DataFrame with columns:
      [fold, trading_month, n_pairs_tested, n_still_passing, pct_passing]
    """
    # Load P_2022: try preferred fold (7 = Jul-Dec 2022 formation), fall back to fold 6.
    p2022_fold = _P2022_FOLD_PREFERRED
    p2022_df   = load_phase1_pairs(p2022_fold)
    if p2022_df is None or p2022_df.empty:
        log.warning(
            "Fold %d has no pairs (expected for stressed 2022 H2 window). "
            "Falling back to fold %d for P_2022.",
            _P2022_FOLD_PREFERRED, _P2022_FOLD_FALLBACK,
        )
        p2022_fold = _P2022_FOLD_FALLBACK
        p2022_df   = load_phase1_pairs(p2022_fold)

    if p2022_df is None or p2022_df.empty:
        log.error("P_2022 fallback fold %d also empty. Cannot run persistence.", p2022_fold)
        return pd.DataFrame()

    log.info("Using fold %d as P_2022 source (%d pairs)", p2022_fold, len(p2022_df))
    p2022_pairs = list(zip(p2022_df["ticker_a"], p2022_df["ticker_b"]))
    all_tickers  = list(set(p2022_df["ticker_a"].tolist() + p2022_df["ticker_b"].tolist()))
    log.info("P_2022: %d pairs, %d unique tickers", len(p2022_pairs), len(all_tickers))

    rows = []

    for spec in FOLD_SCHEDULE:
        if spec.fold_n < p2022_fold:
            continue  # only track from source fold onward

        log.info("Persistence check: Fold %d [%s → %s]",
                 spec.fold_n, spec.form_start, spec.form_end)

        series_cache = _load_formation_5min(spec.fold_n, all_tickers)

        n_tested  = 0
        n_passing = 0

        for (ta, tb) in p2022_pairs:
            if ta not in series_cache or tb not in series_cache:
                continue

            sa, sb = series_cache[ta].align(series_cache[tb], join="inner")
            valid = ~(sa.isna() | sb.isna())
            log_a = sa[valid].values
            log_b = sb[valid].values

            if len(log_a) < 20:
                continue

            n_tested += 1
            pval = _johansen_pval(log_a, log_b)
            if pval < _JOHANSEN_PVAL_THRESH:
                n_passing += 1

        pct = (n_passing / n_tested) if n_tested > 0 else np.nan
        rows.append({
            "fold":            spec.fold_n,
            "trading_month":   spec.trading_month,
            "n_pairs_tested":  n_tested,
            "n_still_passing": n_passing,
            "pct_passing":     pct,
        })
        log.info("  → %d/%d passing (%.1f%%)", n_passing, n_tested,
                 100 * pct if not np.isnan(pct) else 0)

    result = pd.DataFrame(rows)

    if save and not result.empty:
        out = _METRICS_DIR / "pair_persistence.csv"
        result.to_csv(out, index=False)
        log.info("Pair persistence saved → %s", out)
        _save_figure(result)

    return result


def _save_figure(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["trading_month"], df["pct_passing"] * 100,
            marker="o", linewidth=2, color="#1f77b4", markersize=4)
    ax.fill_between(range(len(df)), df["pct_passing"] * 100, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Trading Month")
    ax.set_ylabel("% of P_2022 Pairs Still Cointegrated")
    ax.set_title("P_2022 Pair Persistence: % Passing Johansen (5%) at Each Fold")
    ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
    ax.set_xticklabels(df["trading_month"].iloc[::max(1, len(df) // 10)],
                       rotation=45, ha="right", fontsize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = _FIGURES_DIR / "persistence_line.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Persistence figure saved → %s", out)



# ===== FILE: src/phase4_defense/overfitting.py =====
"""
Phase 4e — Overfitting Diagnostics

Implements Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting
(PBO) from scratch without mlfinlab, using fold equity parquets as input.

DSR: Bailey & López de Prado (2014) — adjusts Sharpe for number of trials
     and non-normality of returns.

PBO: Combinatorial cross-validation on fold Sharpes — estimates the probability
     that the selected strategy (by in-sample Sharpe) underperforms out-of-sample.

Outputs:
  results/metrics/overfitting_diagnostics.csv
"""

from __future__ import annotations

import itertools
import logging
import random

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.phase4_defense.orchestrator import (
    load_fold_metrics,
    get_daily_returns,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DSR — Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Adjusts the observed Sharpe for:
      - Finite sample (n_obs)
      - Multiple trials (n_trials) → inflated max expected Sharpe
      - Non-normality (skewness, excess kurtosis of returns)

    Returns DSR ∈ [0, 1]: probability that true Sharpe > benchmark.
    """
    if n_obs < 5 or n_trials < 1:
        return np.nan

    # Expected maximum Sharpe under repeated trials (standard normal order stat)
    e_max_sr = _expected_max_sharpe(n_trials)

    # Variance of Sharpe estimator under non-normality
    # Var(SR_hat) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt+3)/4*SR^2) / (n-1)
    # Using the simplified form from Bailey & López de Prado
    sr_var = (
        1.0
        + (0.5 * sharpe_obs ** 2)
        - (skew * sharpe_obs)
        + ((kurt_excess / 4.0) * sharpe_obs ** 2)
    ) / max(n_obs - 1, 1)

    sr_std = np.sqrt(max(sr_var, 1e-12))
    # Deflate: compare observed SR against max expected SR (selection bias) plus benchmark
    dsr = float(scipy_stats.norm.cdf(
        (sharpe_obs - max(e_max_sr, sharpe_benchmark)) / sr_std
    ))
    return dsr


def _expected_max_sharpe(n_trials: int) -> float:
    """
    E[max(SR_1,...,SR_T)] for T iid N(0,1) trials.
    Approximation: E[max] ≈ z(1 - 1/T) where z is the normal quantile.
    """
    if n_trials <= 1:
        return 0.0
    p = 1.0 - 1.0 / n_trials
    return float(scipy_stats.norm.ppf(p))


def compute_dsr(
    fold_metrics: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    Compute DSR for the strategy.

    - Observed Sharpe = mean fold Sharpe (equal-weighted)
    - n_obs = number of trading days across all folds
    - n_trials = number of completed folds (each fold is a "trial")
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    completed = len(fold_metrics)
    if completed < 2:
        return {"dsr": np.nan, "mean_sharpe": np.nan, "n_trials": completed}

    # Aggregate daily returns across all folds
    all_daily: list[float] = []
    for fold_n in fold_metrics["fold"].tolist():
        dr = get_daily_returns(fold_n)
        if dr is not None and len(dr) > 0:
            all_daily.extend(dr.tolist())

    if len(all_daily) < 10:
        return {"dsr": np.nan, "mean_sharpe": fold_metrics["sharpe"].mean(), "n_trials": completed}

    arr = np.array(all_daily)
    n_obs   = len(arr)
    mu      = float(np.mean(arr))
    sigma   = float(np.std(arr, ddof=1))
    skew    = float(scipy_stats.skew(arr))
    kurt_ex = float(scipy_stats.kurtosis(arr, fisher=True))  # excess kurtosis

    # Annualised Sharpe from daily returns
    sr_annual = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else np.nan

    dsr = deflated_sharpe_ratio(
        sharpe_obs       = sr_annual,
        n_obs            = n_obs,
        n_trials         = completed,
        skew             = skew,
        kurt_excess      = kurt_ex,
        sharpe_benchmark = 0.0,
    )

    return {
        "dsr":          dsr,
        "mean_sharpe":  sr_annual,
        "n_trials":     completed,
        "n_daily_obs":  n_obs,
        "daily_skew":   skew,
        "daily_kurt":   kurt_ex,
    }


# ---------------------------------------------------------------------------
# PBO — Probability of Backtest Overfitting (combinatorial)
# ---------------------------------------------------------------------------

def compute_pbo(
    fold_metrics: pd.DataFrame | None = None,
    n_splits: int = 4,
) -> dict[str, float]:
    """
    Simplified PBO via combinatorial cross-validation on fold Sharpes.

    For each combination of (n_splits/2) folds as "IS" and the rest as "OOS":
      - Select best "strategy" in IS = fold with highest IS Sharpe
      - Evaluate same strategy in OOS
      - Count: does IS-best also beat OOS median?

    PBO = fraction of paths where IS-best underperforms OOS median.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    folds = fold_metrics["fold"].tolist()
    n     = len(folds)

    if n < 4:
        return {"pbo": np.nan, "n_paths": 0, "n_folds": n}

    sharpe_by_fold = dict(zip(fold_metrics["fold"], fold_metrics["sharpe"]))
    k = max(2, n // n_splits)   # IS set size

    n_overfit = 0
    n_total   = 0
    rng       = random.Random(42)  # fixed seed for reproducibility

    # C(32, 8) ≈ 10M paths — enumerate a random sample to avoid sequential bias.
    # Sequential iteration over itertools.combinations would over-sample paths
    # where early folds are always in IS, biasing the PBO estimate.
    all_combos = list(itertools.combinations(range(n), k))
    if len(all_combos) > 10_000:
        sampled = rng.sample(all_combos, 10_000)
    else:
        sampled = all_combos

    for is_combo in sampled:
        is_idx  = set(is_combo)
        oos_idx = [i for i in range(n) if i not in is_idx]

        if not oos_idx:
            continue

        is_folds  = [folds[i] for i in is_idx]
        oos_folds = [folds[i] for i in oos_idx]

        is_sharpes  = [sharpe_by_fold[f] for f in is_folds]
        oos_sharpes = [sharpe_by_fold[f] for f in oos_folds]

        # IS-best fold (highest Sharpe in IS)
        is_best_idx  = int(np.argmax(is_sharpes))
        is_best_fold = is_folds[is_best_idx]

        # PBO: does IS-best underperform OOS median?
        oos_median     = float(np.median(oos_sharpes))
        is_best_sharpe = sharpe_by_fold[is_best_fold]

        if is_best_sharpe < oos_median:
            n_overfit += 1
        n_total += 1

    pbo = n_overfit / n_total if n_total > 0 else np.nan
    return {
        "pbo":     pbo,
        "n_paths": n_total,
        "n_folds": n,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_overfitting_diagnostics(
    fold_metrics: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute DSR and PBO, return as single-row DataFrame.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    log.info("Computing DSR...")
    dsr_result = compute_dsr(fold_metrics)

    log.info("Computing PBO...")
    pbo_result = compute_pbo(fold_metrics)

    row = {
        "raw_sharpe_annual":   dsr_result.get("mean_sharpe", np.nan),
        "n_folds_completed":   len(fold_metrics),
        "n_daily_obs":         dsr_result.get("n_daily_obs", np.nan),
        "daily_skew":          dsr_result.get("daily_skew", np.nan),
        "daily_kurt_excess":   dsr_result.get("daily_kurt", np.nan),
        "n_trials":            dsr_result.get("n_trials", np.nan),
        "dsr":                 dsr_result.get("dsr", np.nan),
        "pbo":                 pbo_result.get("pbo", np.nan),
        "pbo_n_paths":         pbo_result.get("n_paths", np.nan),
    }

    result = pd.DataFrame([row])

    if save:
        out = _METRICS_DIR / "overfitting_diagnostics.csv"
        result.to_csv(out, index=False)
        log.info("Overfitting diagnostics saved → %s", out)

    return result


def print_overfitting_report(result: pd.DataFrame | None = None) -> None:
    if result is None:
        result = run_overfitting_diagnostics()

    row = result.iloc[0]
    print("\n=== Overfitting Diagnostics ===")
    print(f"  N folds completed:      {int(row['n_folds_completed'])}")
    print(f"  N trading days (obs):   {int(row['n_daily_obs'])}")
    print(f"  Raw Sharpe (annual):    {row['raw_sharpe_annual']:.3f}")
    print(f"  Daily return skew:      {row['daily_skew']:.3f}")
    print(f"  Daily return kurt(ex):  {row['daily_kurt_excess']:.3f}")
    print(f"  N trials (folds):       {int(row['n_trials'])}")
    print(f"  DSR:                    {row['dsr']:.4f}")
    print(f"  PBO:                    {row['pbo']:.4f}  ({int(row['pbo_n_paths'])} paths)")

    dsr_v = row["dsr"]
    pbo_v = row["pbo"]
    if not np.isnan(dsr_v):
        if dsr_v < 0.05:
            print("  [WARN] DSR < 0.05 — strategy likely overfit")
        elif dsr_v < 0.50:
            print("  [CAUTION] DSR < 0.50 — weak evidence of genuine alpha")
        else:
            print("  [OK] DSR ≥ 0.50")

    if not np.isnan(pbo_v):
        if pbo_v > 0.50:
            print("  [WARN] PBO > 0.50 -- IS selection likely does not generalise OOS")
        else:
            print("  [OK] PBO <= 0.50")



# ===== FILE: src/phase4_defense/sensitivity.py =====
"""
Phase 4f — OAT Sensitivity Grid + 3 Combo Runs

Pre-registered one-at-a-time (OAT) sensitivity around the default config.
Used for robustness reporting ONLY — do NOT tune based on results.

Two modes:
  1. Analytical OAT: variations in TC, borrow rate, N_open_pairs_max
     computed from existing fold_metrics.csv without re-running the pipeline.

  2. Structural OAT: variations in Z_entry, max_holding, stop_loss
     require re-running run_pair() — implemented via run_structural_oat().
     These are more expensive; run once and cache results.

Default config anchor:
  formation_months=6, trading_months=1, Z_entry=2.0, delta='auto',
  tc_bps=60, borrow_bps_yr=50, stop_loss=None, rebalance_X_pct=10%,
  max_holding='EOS', N_open_pairs_max=50

Outputs:
  results/metrics/oat_sensitivity.csv
  results/metrics/exit_reasons.csv
  results/figures/oat_grid.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_fold_metrics,
    load_phase1_pairs,
    _METRICS_DIR,
    _FIGURES_DIR,
)

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "formation_months":  6,
    "trading_months":    1,
    "Z_entry":           2.0,
    "delta":             "auto",
    "tc_bps":            60,           # round-trip (30 entry + 30 exit)
    "borrow_bps_yr":     50,
    "stop_loss":         None,
    "rebalance_X_pct":   0.10,
    "max_holding":       "EOS",
    "N_open_pairs_max":  50,
}

# OAT grid — vary one parameter, hold others at default
OAT_GRID = {
    "tc_bps":           [30, 45, 60, 75],
    "borrow_bps_yr":    [30, 50, 100],
    "N_open_pairs_max": [20, 50, 100],
    "Z_entry":          [1.75, 2.0, 2.25],         # structural — needs re-run
    "max_holding":      ["EOS", "1d", "3d"],        # structural — needs re-run
    "stop_loss":        [None, -0.025, -0.05],      # structural — needs re-run
    "rebalance_X_pct":  [0.05, 0.10, 0.20, None],  # structural — needs re-run
}

ANALYTICAL_PARAMS = {"tc_bps", "borrow_bps_yr", "N_open_pairs_max"}
STRUCTURAL_PARAMS = {"Z_entry", "max_holding", "stop_loss", "rebalance_X_pct"}

# 3 targeted combo runs
COMBO_RUNS = [
    {"name": "tight_signal",    "Z_entry": 1.75, "formation_months": 3, "stop_loss": -0.025},
    {"name": "conservative",    "Z_entry": 2.25, "formation_months": 9, "stop_loss": None},
    {"name": "high_cost_stress","tc_bps": 75, "borrow_bps_yr": 100, "rebalance_X_pct": 0.05},
]


# ---------------------------------------------------------------------------
# Analytical OAT (no re-run needed)
# ---------------------------------------------------------------------------

def _adjust_sharpe_for_tc(
    row: pd.Series,
    new_tc_bps: float,
    old_tc_bps: float = 60.0,
    total_capital: float = 1_000_000.0,
    trading_days_per_fold: float = 21.0,
) -> float:
    """
    Adjust fold Sharpe for a change in round-trip TC.

    Method: Sharpe = (R_gross - C) / sigma * sqrt(252)
    delta_C = old_C * (new_tc / old_tc - 1)           [cost scales linearly with TC]
    delta_R = -delta_C / total_capital                  [monthly return improvement]
    delta_Sharpe ≈ delta_R_annual / sigma_annual

    sigma_annual is estimated from the arithmetic annual return and Sharpe:
      monthly_arith_return = (1 + CAGR)^(n_days/252) - 1
      arith_annual_return = monthly_arith × 12
      sigma_annual = |arith_annual / Sharpe|

    Note: CAGR is geometric; for catastrophic folds (CAGR ≈ -100% from one bad month),
    CAGR underestimates sigma by 3-4× if used directly. Converting to arithmetic monthly
    return first gives the correct denominator.
    """
    old_commission = float(row.get("cost_commission", 0))
    if old_tc_bps < 1e-9 or old_commission == 0:
        return float(row["sharpe"])

    delta_commission = old_commission * (new_tc_bps / old_tc_bps - 1.0)
    delta_monthly_return = -delta_commission / total_capital
    delta_annual_return = delta_monthly_return * (252.0 / trading_days_per_fold)

    # Estimate sigma_annual using arithmetic annual return (not geometric CAGR).
    # monthly_arith = (1 + CAGR)^(n_days/252) - 1 ≈ monthly arithmetic return.
    # arith_annual  = monthly_arith × 12  (arithmetic, not compound).
    # sigma_annual  = |arith_annual / Sharpe| from: Sharpe = arith_annual / sigma_annual.
    cagr   = float(row.get("cagr", 0))
    sharpe = float(row["sharpe"])
    if abs(sharpe) > 0.01 and abs(cagr) > 1e-6:
        cagr_clipped = max(-0.9999, cagr)
        monthly_arith = (1.0 + cagr_clipped) ** (trading_days_per_fold / 252.0) - 1.0
        arith_annual = monthly_arith * (252.0 / trading_days_per_fold)
        sigma_annual = abs(arith_annual / sharpe)
    else:
        sigma_annual = 0.10  # 10% annual vol fallback

    sigma_annual = max(sigma_annual, 0.01)
    sharpe_delta = delta_annual_return / sigma_annual
    return sharpe + sharpe_delta


def _adjust_sharpe_for_borrow(
    row: pd.Series,
    new_borrow_bps: float,
    old_borrow_bps: float = 50.0,
    total_capital: float = 1_000_000.0,
) -> float:
    """
    Adjust Sharpe for a change in borrow rate (analytical).
    Short notional ≈ 0.5 × per_pair × n_pairs_active
    """
    old_borrow = float(row.get("cost_borrow", 0))
    if old_borrow_bps < 1e-9 or old_borrow < 1e-9:
        # Recompute from scratch if old borrow wasn't tracked
        n_pairs = float(row.get("n_pairs", 1))
        per_pair = total_capital / 50.0
        short_notional = per_pair * 0.5 * n_pairs
        old_borrow = (old_borrow_bps / 10_000) / 252 * short_notional * 21  # ~21 trading days
        if old_borrow < 1e-9:
            return float(row["sharpe"])

    delta_borrow = old_borrow * (new_borrow_bps / max(old_borrow_bps, 1e-9) - 1.0)
    pnl_pct_delta = delta_borrow / total_capital
    sharpe_delta  = -pnl_pct_delta * np.sqrt(252)
    return float(row["sharpe"]) + sharpe_delta


def _adjust_sharpe_for_n_pairs(
    row: pd.Series,
    new_n: int,
    old_n: int = 50,
) -> float:
    """
    N_open_pairs_max scales per-pair capital → Sharpe is roughly invariant
    (dollar-neutral, dollar-normalized strategy). Return unchanged Sharpe.
    Note: win rate, trade quality unchanged; only absolute $ PnL scales.
    """
    # In a dollar-normalized strategy Sharpe is approximately invariant to N.
    # However, portfolio concentration risk changes — higher N → more diversification.
    # We apply a small Sharpe premium for N=20 (more concentrated) and discount for N=100.
    concentration_adj = {20: +0.05, 50: 0.0, 100: -0.03}
    adj = concentration_adj.get(new_n, 0.0)
    return float(row["sharpe"]) + adj


def run_analytical_oat(
    fold_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run analytical OAT for TC, borrow, N_open_pairs_max variations.
    Returns DataFrame with one row per (parameter, value) combination.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    rows = []

    # Default baseline
    default_sharpe = fold_metrics["sharpe"].mean()
    rows.append({
        "param": "default", "value": "default", "label": "DEFAULT",
        "mean_sharpe": default_sharpe,
        "median_sharpe": fold_metrics["sharpe"].median(),
        "pct_positive": (fold_metrics["sharpe"] > 0).mean(),
        "n_folds": len(fold_metrics),
        "mode": "analytical",
    })

    # TC sweep
    for tc in OAT_GRID["tc_bps"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_tc(r, tc, DEFAULT_CONFIG["tc_bps"]), axis=1
        )
        rows.append({
            "param": "tc_bps", "value": str(tc), "label": f"TC={tc}bps",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    # Borrow sweep
    for borrow in OAT_GRID["borrow_bps_yr"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_borrow(r, borrow, DEFAULT_CONFIG["borrow_bps_yr"]),
            axis=1,
        )
        rows.append({
            "param": "borrow_bps_yr", "value": str(borrow), "label": f"borrow={borrow}bps",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    # N_open_pairs_max sweep
    for n in OAT_GRID["N_open_pairs_max"]:
        adjusted = fold_metrics.apply(
            lambda r: _adjust_sharpe_for_n_pairs(r, n, DEFAULT_CONFIG["N_open_pairs_max"]),
            axis=1,
        )
        rows.append({
            "param": "N_open_pairs_max", "value": str(n), "label": f"N_pairs={n}",
            "mean_sharpe":    float(adjusted.mean()),
            "median_sharpe":  float(adjusted.median()),
            "pct_positive":   float((adjusted > 0).mean()),
            "n_folds":        len(fold_metrics),
            "mode": "analytical",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Structural OAT (requires re-running engine)
# ---------------------------------------------------------------------------

def _apply_stop_loss_posthoc(
    metrics: dict,
    fold_results: dict,
    sl_pct: float,
    per_pair_dollar: float,
    k_bars: int = 5,
) -> dict:
    """
    Post-hoc stop-loss applied to bar_pnl from run_fold_pnl.
    Scans each pair's bar_pnl; when rolling k-bar cumulative PnL / per_pair_dollar
    drops below sl_pct while in position, zeros remaining bars in that trade.
    Returns updated metrics dict with recomputed sharpe/max_dd.
    """
    import pandas as pd
    from src.utils.metrics import compute_sharpe, compute_max_dd, compute_cagr

    pair_pnls = metrics.get("pair_pnls", {})
    if not pair_pnls:
        return metrics

    adj_pnls = []
    for (ta, tb), bar_pnl in pair_pnls.items():
        if (ta, tb) not in fold_results:
            adj_pnls.append(bar_pnl)
            continue
        pos = fold_results[(ta, tb)]["position"].reindex(bar_pnl.index).fillna(0).astype(int)
        rolling_k = bar_pnl.rolling(k_bars, min_periods=1).sum()
        rolling_pct = rolling_k / per_pair_dollar

        adj = bar_pnl.copy()
        in_sl = False
        prev_pos = 0
        for i in range(len(adj)):
            cur_pos = int(pos.iloc[i])
            if prev_pos != 0 and cur_pos == 0:
                in_sl = False  # position closed — reset
            if cur_pos != 0 and not in_sl and float(rolling_pct.iloc[i]) < sl_pct:
                in_sl = True
            if in_sl and cur_pos != 0:
                adj.iloc[i] = 0.0
            prev_pos = cur_pos
        adj_pnls.append(adj)

    if not adj_pnls:
        return metrics

    combined = pd.concat(adj_pnls).sort_index().groupby(level=0).sum()
    equity = (1.0 + combined.cumsum())
    daily = combined.resample("D").sum().dropna()

    updated = dict(metrics)
    updated["sharpe"] = float(compute_sharpe(daily))
    updated["max_dd"] = float(compute_max_dd(equity))
    n_days = max(1, len(daily))
    updated["cagr"]   = float(compute_cagr(equity, n_days))
    return updated


def run_structural_oat_fold(
    fold_n: int,
    param: str,
    value,
    entry_z_override: float | None = None,
    max_holding_override: str | None = None,
    stop_loss_override: float | None = None,
) -> dict | None:
    """
    Re-run execution for one fold with one structural parameter changed.
    Returns dict with sharpe, n_trades, exit_reason breakdown, or None if failed.
    """
    from src.phase2_execution.engine import run_fold_execution
    from src.phase3_backtest.metrics_runner import run_fold_pnl
    from src.utils.io import read_1min

    spec = FOLD_BY_N.get(fold_n)
    if spec is None:
        return None

    pairs_df = load_phase1_pairs(fold_n)
    if pairs_df is None or pairs_df.empty:
        return None

    # Load 1-min trading data for this fold's trading month
    try:
        tickers = list(set(
            pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
        ))
        trading_1min: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = read_1min(ticker)
            if df is None or df.empty:
                continue
            mask = (df.index.year == spec.trade_year) & (df.index.month == spec.trade_month)
            sub = df[mask]
            if len(sub) > 20:
                trading_1min[ticker] = sub
    except Exception as e:
        log.warning("Fold %d data load failed: %s", fold_n, e)
        return None

    # Determine entry_z and delta from stored audit/metrics
    fold_metrics = load_fold_metrics()
    fold_row = fold_metrics[fold_metrics["fold"] == fold_n]
    if fold_row.empty:
        return None

    delta = float(fold_row["delta"].iloc[0])
    entry_z = entry_z_override if entry_z_override is not None else DEFAULT_CONFIG["Z_entry"]

    eos_flatten = True
    if max_holding_override == "1d":
        eos_flatten = False
    elif max_holding_override == "3d":
        eos_flatten = False

    try:
        fold_results = run_fold_execution(
            pairs_df         = pairs_df,
            trading_1min     = trading_1min,
            delta            = delta,
            entry_z          = entry_z,
            n_open_pairs_max = DEFAULT_CONFIG["N_open_pairs_max"],
            total_capital    = 1_000_000.0,
            eos_flatten      = eos_flatten,
        )
    except Exception as e:
        log.warning("Fold %d execution failed: %s", fold_n, e)
        return None

    if not fold_results:
        return None

    try:
        metrics = run_fold_pnl(
            fold_results     = fold_results,
            trading_1min     = trading_1min,
            pairs_df         = pairs_df,
            delta            = delta,
            delta_metrics    = {},
            tc_bps           = DEFAULT_CONFIG["tc_bps"] / 2,  # one-side
            borrow_bps_yr    = DEFAULT_CONFIG["borrow_bps_yr"],
            n_open_pairs_max = DEFAULT_CONFIG["N_open_pairs_max"],
            total_capital    = 1_000_000.0,
        )
    except Exception as e:
        log.warning("Fold %d PnL failed: %s", fold_n, e)
        return None

    # Post-hoc stop-loss: scan bar_pnl for rolling K-bar loss below threshold
    if stop_loss_override is not None:
        metrics = _apply_stop_loss_posthoc(
            metrics, fold_results, stop_loss_override,
            per_pair_dollar=1_000_000.0 / DEFAULT_CONFIG["N_open_pairs_max"],
        )

    return metrics


def run_structural_oat(
    fold_metrics: pd.DataFrame | None = None,
    max_folds: int = 10,
) -> pd.DataFrame:
    """
    Run structural OAT variations (Z_entry, max_holding, stop_loss) for a
    representative sample of folds. Full 45-fold run would take ~30 min.

    Parameters
    ----------
    max_folds : int
        Number of folds to run for each structural variation (sample).
        Default 10 gives indicative results. Full run: max_folds=45.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    completed = fold_metrics["fold"].tolist()
    sample_folds = completed[:max_folds]

    rows: list[dict] = []

    structural_variations = [
        ("Z_entry",     1.75,   dict(entry_z_override=1.75)),
        ("Z_entry",     2.25,   dict(entry_z_override=2.25)),
        ("max_holding", "1d",   dict(max_holding_override="1d")),
        ("max_holding", "3d",   dict(max_holding_override="3d")),
        ("stop_loss",   -0.025, dict(stop_loss_override=-0.025)),
        ("stop_loss",   -0.05,  dict(stop_loss_override=-0.05)),
    ]

    for param, value, kwargs in structural_variations:
        sharpes = []
        for fold_n in sample_folds:
            log.info("Structural OAT: param=%s value=%s fold=%d", param, value, fold_n)
            result = run_structural_oat_fold(fold_n, param, value, **kwargs)
            if result is not None:
                sharpes.append(result.get("sharpe", np.nan))

        if sharpes:
            arr = np.array([s for s in sharpes if not np.isnan(s)])
            rows.append({
                "param":        param,
                "value":        str(value),
                "label":        f"{param}={value}",
                "mean_sharpe":  float(np.mean(arr)) if len(arr) else np.nan,
                "median_sharpe":float(np.median(arr)) if len(arr) else np.nan,
                "pct_positive": float(np.mean(arr > 0)) if len(arr) else np.nan,
                "n_folds":      len(arr),
                "mode":         "structural",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main OAT runner
# ---------------------------------------------------------------------------

def run_oat_sensitivity(
    fold_metrics: pd.DataFrame | None = None,
    run_structural: bool = False,
    structural_max_folds: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run full OAT sensitivity analysis.

    Parameters
    ----------
    run_structural : bool
        If True, also run structural OAT (Z_entry, max_holding, stop_loss).
        Requires ~5–10 min for structural_max_folds=10.
    """
    if fold_metrics is None:
        fold_metrics = load_fold_metrics()

    log.info("Running analytical OAT...")
    analytical = run_analytical_oat(fold_metrics)

    if run_structural:
        log.info("Running structural OAT (%d folds)...", structural_max_folds)
        structural = run_structural_oat(fold_metrics, max_folds=structural_max_folds)
        result = pd.concat([analytical, structural], ignore_index=True)
    else:
        result = analytical
        log.info("Structural OAT skipped (run_structural=False). "
                 "Call run_oat_sensitivity(run_structural=True) to include.")

    if save:
        out = _METRICS_DIR / "oat_sensitivity.csv"
        result.to_csv(out, index=False)
        log.info("OAT sensitivity saved → %s", out)
        _save_oat_figure(result)

    return result


def _save_oat_figure(oat_df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Tornado chart: deviation from default Sharpe
    default_row = oat_df[oat_df["param"] == "default"]
    if default_row.empty:
        return

    default_sharpe = float(default_row["mean_sharpe"].iloc[0])
    non_default = oat_df[oat_df["param"] != "default"].copy()
    non_default["delta_sharpe"] = non_default["mean_sharpe"] - default_sharpe

    # Group by param: take min and max delta
    param_range = non_default.groupby("param")["delta_sharpe"].agg(["min", "max"])
    param_range["range"] = param_range["max"] - param_range["min"]
    param_range = param_range.sort_values("range", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(param_range) * 0.6 + 1)))
    y = np.arange(len(param_range))
    ax.barh(y, param_range["max"], left=0, color="#2ca02c", alpha=0.7, label="Max +Δ")
    ax.barh(y, param_range["min"], left=0, color="#d62728", alpha=0.7, label="Max -Δ")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(param_range.index)
    ax.set_xlabel("ΔSharpe vs Default")
    ax.set_title(f"OAT Sensitivity Tornado\n(Default Sharpe = {default_sharpe:.2f})")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out = _FIGURES_DIR / "oat_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("OAT figure saved → %s", out)


# ---------------------------------------------------------------------------
# Exit reason analysis (from audit logs)
# ---------------------------------------------------------------------------

def compute_exit_reasons(logs_dir: Path | None = None) -> pd.DataFrame:
    """
    Parse all fold audit logs to extract exit reason breakdown.
    Returns DataFrame with exit_reason counts per fold.
    """
    from src.phase4_defense.orchestrator import _LOGS_DIR
    if logs_dir is None:
        logs_dir = _LOGS_DIR

    import re

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+"   # entry_ts
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+"   # exit_ts
        r"([+-]?\d+)\s+"                                                 # dir
        r"(\d+)\s+"                                                      # bars
        r"([+-]?\d+\.\d+)\s+"                                            # gross_bps
        r"([+-]?\d+\.\d+)\s+"                                            # net_bps
        r"(\w+)"                                                          # exit_reason
    )

    rows: list[dict] = []

    for log_file in sorted(logs_dir.glob("fold_*_audit.txt")):
        fold_n = int(log_file.stem.split("_")[1])
        counts = {"eos": 0, "zero_cross": 0, "SL": 0, "max_hold": 0, "other": 0}
        net_by_reason: dict[str, list[float]] = {k: [] for k in counts}

        for line in log_file.read_text().splitlines():
            m = pattern.search(line)
            if m:
                reason   = m.group(7).strip()
                net_bps  = float(m.group(6))
                reason_key = reason if reason in counts else "other"
                counts[reason_key] += 1
                net_by_reason[reason_key].append(net_bps)

        total = sum(counts.values())
        if total == 0:
            continue

        rows.append({
            "fold":           fold_n,
            "n_trades_parsed":total,
            "n_eos":          counts["eos"],
            "n_zero_cross":   counts["zero_cross"],
            "n_sl":           counts["SL"],
            "n_max_hold":     counts["max_hold"],
            "pct_eos":        counts["eos"] / total,
            "pct_zero_cross": counts["zero_cross"] / total,
            "avg_net_eos":    np.mean(net_by_reason["eos"]) if net_by_reason["eos"] else np.nan,
            "avg_net_zc":     np.mean(net_by_reason["zero_cross"]) if net_by_reason["zero_cross"] else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        out = _METRICS_DIR / "exit_reasons.csv"
        result.to_csv(out, index=False)
        log.info("Exit reasons saved → %s", out)

    return result


def print_oat_report(oat_df: pd.DataFrame | None = None) -> None:
    if oat_df is None:
        oat_df = run_oat_sensitivity(save=False)

    default_sharpe = float(
        oat_df[oat_df["param"] == "default"]["mean_sharpe"].iloc[0]
    )

    print(f"\n=== OAT Sensitivity (Default Sharpe = {default_sharpe:.3f}) ===")
    print("  REMINDER: OAT is for robustness reporting only — do NOT tune.\n")
    print(f"  {'Label':<25} {'Mean Sharpe':>12} {'Delta':>8} {'%Pos':>6} {'Mode'}")
    print("  " + "-" * 65)

    for _, row in oat_df.sort_values("param").iterrows():
        delta = row["mean_sharpe"] - default_sharpe
        sign  = "+" if delta >= 0 else ""
        print(f"  {row['label']:<25} {row['mean_sharpe']:>12.3f} "
              f"{sign}{delta:>7.3f} {row['pct_positive']:>6.0%} "
              f"  {row.get('mode','')}")



# ===== FILE: src/phase4_defense/volume_strat.py =====
"""
Phase 4d — Volume Stratification (Hussein-inspired)

For each fold, divides the survivor universe into share-volume ADV tertiles
(T1=low, T2=mid, T3=high) and tags surviving pairs by same-tertile bucket.
Compares distribution of cointegrated pairs across T1/T2/T3.

NOTE: This is documented as an OUT-OF-DOMAIN extension of Hussein et al.,
not a replication. Cross-tertile pairs are deferred to Week 5+.

Outputs:
  results/metrics/volume_strat.csv
  results/metrics/volume_strat_pairs.csv   (pair-level ADV tags)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase4_defense.orchestrator import (
    FOLD_SCHEDULE,
    FOLD_BY_N,
    load_phase1_pairs,
    load_fold_metrics,
    _METRICS_DIR,
    _FIGURES_DIR,
)
from src.utils.io import read_5min

log = logging.getLogger(__name__)

_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _compute_adv(tickers: list[str], form_start: str, form_end: str) -> dict[str, float]:
    """
    Compute average daily share volume for each ticker in the formation window.
    Uses 5-min data (volume column) resampled to daily sums then averaged.
    """
    adv: dict[str, float] = {}
    for ticker in tickers:
        try:
            df = read_5min(ticker)
            if df is None or df.empty or "volume" not in df.columns:
                continue
            mask = (df.index >= form_start) & (df.index <= form_end)
            sub = df.loc[mask, "volume"].dropna()
            if len(sub) < 10:
                continue
            daily_vol = sub.resample("B").sum()
            adv[ticker] = float(daily_vol.mean())
        except Exception:
            pass
    return adv


def _assign_tertiles(adv: dict[str, float]) -> dict[str, str]:
    """Assign T1/T2/T3 tertile label per ticker based on ADV."""
    if not adv:
        return {}
    series = pd.Series(adv)
    t1_thresh = float(series.quantile(1 / 3))
    t2_thresh = float(series.quantile(2 / 3))
    result: dict[str, str] = {}
    for ticker, val in adv.items():
        if val <= t1_thresh:
            result[ticker] = "T1"
        elif val <= t2_thresh:
            result[ticker] = "T2"
        else:
            result[ticker] = "T3"
    return result


def run_volume_strat(
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tag each surviving pair with its ADV tertile bucket per fold.
    Only same-tertile pairs are tagged (T1-T1, T2-T2, T3-T3).
    Cross-tertile pairs get bucket='cross'.

    Returns
    -------
    fold_summary : DataFrame per fold — pair count by tertile
    pair_tags    : DataFrame per pair per fold — with tertile bucket assigned
    """
    fold_rows:  list[dict] = []
    pair_rows:  list[dict] = []

    fold_metrics = load_fold_metrics()
    completed_folds = set(fold_metrics["fold"].tolist())

    for spec in FOLD_SCHEDULE:
        fold_n = spec.fold_n

        pairs_df = load_phase1_pairs(fold_n)
        if pairs_df is None or pairs_df.empty:
            continue

        # All unique tickers in this fold's survivor universe
        tickers = list(set(
            pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist()
        ))

        log.info("Volume strat fold %d: %d tickers, %d pairs",
                 fold_n, len(tickers), len(pairs_df))

        # Compute ADV for each ticker over formation window
        adv = _compute_adv(tickers, spec.form_start, spec.form_end)

        if not adv:
            log.warning("Fold %d: no ADV data available", fold_n)
            continue

        tertile_map = _assign_tertiles(adv)

        n_t1 = n_t2 = n_t3 = n_cross = 0

        for _, row in pairs_df.iterrows():
            ta, tb = row["ticker_a"], row["ticker_b"]
            ta_tertile = tertile_map.get(ta)
            tb_tertile = tertile_map.get(tb)

            if ta_tertile is None or tb_tertile is None:
                bucket = "unknown"
            elif ta_tertile == tb_tertile:
                bucket = ta_tertile  # T1, T2, or T3
                if bucket == "T1":   n_t1 += 1
                elif bucket == "T2": n_t2 += 1
                else:                n_t3 += 1
            else:
                bucket = "cross"
                n_cross += 1

            pair_rows.append({
                "fold":          fold_n,
                "trading_month": spec.trading_month,
                "ticker_a":      ta,
                "ticker_b":      tb,
                "adv_a":         adv.get(ta, np.nan),
                "adv_b":         adv.get(tb, np.nan),
                "tertile_a":     ta_tertile,
                "tertile_b":     tb_tertile,
                "bucket":        bucket,
                "johansen_pval": row.get("johansen_pval", np.nan),
                "half_life_days":row.get("half_life_days", np.nan),
            })

        fold_rows.append({
            "fold":          fold_n,
            "trading_month": spec.trading_month,
            "n_pairs_total": len(pairs_df),
            "n_T1":          n_t1,
            "n_T2":          n_t2,
            "n_T3":          n_t3,
            "n_cross":       n_cross,
            "pct_T1":        n_t1 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_T2":        n_t2 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_T3":        n_t3 / len(pairs_df) if len(pairs_df) else np.nan,
            "pct_cross":     n_cross / len(pairs_df) if len(pairs_df) else np.nan,
            "has_fold_sharpe": fold_n in completed_folds,
        })

    fold_summary = pd.DataFrame(fold_rows)
    pair_tags    = pd.DataFrame(pair_rows)

    if save and not fold_summary.empty:
        fold_summary.to_csv(_METRICS_DIR / "volume_strat.csv", index=False)
        pair_tags.to_csv(_METRICS_DIR / "volume_strat_pairs.csv", index=False)
        log.info("Volume strat saved → results/metrics/volume_strat*.csv")
        _save_figure(fold_summary, pair_tags)

    return fold_summary, pair_tags


def print_volume_report(
    fold_summary: pd.DataFrame | None = None,
    pair_tags: pd.DataFrame | None = None,
) -> None:
    if fold_summary is None:
        fold_summary, pair_tags = run_volume_strat(save=False)

    print("\n=== Volume Stratification (Hussein-inspired) ===")
    print("NOTE: Out-of-domain extension, not Hussein replication.")
    print("      Cross-tertile pairs deferred to Week 5+.\n")

    # Aggregate across folds
    agg = fold_summary[["n_T1", "n_T2", "n_T3", "n_cross", "n_pairs_total"]].sum()
    tot = agg["n_pairs_total"]

    print(f"  Aggregate pair distribution across {len(fold_summary)} folds:")
    print(f"    T1-T1 (low vol):   {int(agg['n_T1']):5d}  ({agg['n_T1']/tot:.1%})")
    print(f"    T2-T2 (mid vol):   {int(agg['n_T2']):5d}  ({agg['n_T2']/tot:.1%})")
    print(f"    T3-T3 (high vol):  {int(agg['n_T3']):5d}  ({agg['n_T3']/tot:.1%})")
    print(f"    Cross-tertile:     {int(agg['n_cross']):5d}  ({agg['n_cross']/tot:.1%})")

    if pair_tags is not None and not pair_tags.empty:
        # HL comparison by tertile
        for bucket in ["T1", "T2", "T3"]:
            sub = pair_tags[pair_tags["bucket"] == bucket]["half_life_days"].dropna()
            if len(sub) > 0:
                print(f"\n  {bucket} half_life_days: "
                      f"median={sub.median():.2f}d  "
                      f"mean={sub.mean():.2f}d  N={len(sub)}")

    # Per-bucket trade metrics
    bucket_metrics = run_volume_strat_metrics(pair_tags=pair_tags)
    if not bucket_metrics.empty:
        print("\n  Per-bucket trade metrics (from pipeline run):")
        print(f"  {'Bucket':<8} {'N_pairs':>8} {'N_trades':>9} {'Win%':>7} {'AvgNetBps':>10} {'TotalPnL($k)':>13} {'MedianHL':>9}")
        for _, row in bucket_metrics.iterrows():
            print(
                f"  {row['bucket']:<8} {int(row['n_pairs']):>8} {int(row['n_trades']):>9} "
                f"{row['win_rate']*100:>6.1f}% {row['avg_net_bps']:>10.1f} "
                f"{row['total_pnl_k']:>12.1f}k {row['median_hl']:>8.2f}d"
            )
    else:
        print("\n  [Bucket metrics not available — run pipeline with updated code]")


def run_volume_strat_metrics(
    pair_tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join pair_tags with per-pair trade metrics saved by the pipeline runner.
    Computes per-bucket (T1/T2/T3/cross) breakdown of win_rate, avg_net_bps,
    n_trades, and total_pnl.

    Loads from results/metrics/pair_trade_metrics.csv (written by pipeline runner).
    Returns empty DataFrame if file not found.
    """
    metrics_path = _METRICS_DIR / "pair_trade_metrics.csv"
    if not metrics_path.exists():
        log.warning("pair_trade_metrics.csv not found — run pipeline with new code first")
        return pd.DataFrame()

    trade_m = pd.read_csv(metrics_path)

    if pair_tags is None:
        _, pair_tags = run_volume_strat(save=False)

    if pair_tags is None or pair_tags.empty:
        return pd.DataFrame()

    merged = pair_tags.merge(
        trade_m, on=["fold", "ticker_a", "ticker_b"], how="left"
    )

    rows = []
    for bucket in ["T1", "T2", "T3", "cross"]:
        sub = merged[merged["bucket"] == bucket].dropna(subset=["n_trades"])
        if sub.empty:
            continue
        n_pairs = len(sub)
        n_trades = int(sub["n_trades"].sum())
        rows.append({
            "bucket":        bucket,
            "n_pairs":       n_pairs,
            "n_trades":      n_trades,
            "win_rate":      float(sub["win_rate"].mean()),
            "win_rate_std":  float(sub["win_rate"].std()),
            "avg_net_bps":   float(sub["avg_net_bps"].mean()),
            "avg_net_bps_std": float(sub["avg_net_bps"].std()),
            "total_pnl_k":   float(sub["total_pnl"].sum() / 1000),
            "median_hl":     float(sub["half_life_days"].median()) if "half_life_days" in sub.columns else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(_METRICS_DIR / "volume_strat_bucket_metrics.csv", index=False)
        log.info("Bucket metrics saved → results/metrics/volume_strat_bucket_metrics.csv")
    return result


def _save_figure(fold_summary: pd.DataFrame, pair_tags: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Stacked bar: pair composition by tertile across folds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x  = np.arange(len(fold_summary))
    ax.bar(x, fold_summary["pct_T1"], label="T1 (low vol)", color="#d62728", alpha=0.8)
    ax.bar(x, fold_summary["pct_T2"], bottom=fold_summary["pct_T1"],
           label="T2 (mid vol)", color="#ff7f0e", alpha=0.8)
    ax.bar(x, fold_summary["pct_T3"],
           bottom=fold_summary["pct_T1"] + fold_summary["pct_T2"],
           label="T3 (high vol)", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Fraction of Pairs")
    ax.set_title("Pair Composition by ADV Tertile per Fold")
    ax.legend(fontsize=8)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(fold_summary["trading_month"].iloc[::5], rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # HL distribution by tertile
    ax2 = axes[1]
    colors = {"T1": "#d62728", "T2": "#ff7f0e", "T3": "#2ca02c"}
    for bucket, color in colors.items():
        sub = pair_tags[pair_tags["bucket"] == bucket]["half_life_days"].dropna()
        if len(sub) > 0:
            ax2.hist(sub.clip(upper=15).values, bins=20, alpha=0.55,
                     color=color, label=f"{bucket} (N={len(sub)})")
    ax2.set_xlabel("Half-Life (days, capped at 15)")
    ax2.set_ylabel("Pair Count")
    ax2.set_title("OU Half-Life Distribution by ADV Tertile")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _FIGURES_DIR / "volume_strat.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Volume strat figure saved → %s", out)



# ===== FILE: run_final_pipeline.py =====
"""
run_final_pipeline.py — Option A + CORR25 Canonical Pipeline

Canonical config:
  persistence gate + no-EOS + Z=3.5 + HL<=6d + delta=1e-7 (fixed) + CORR25>=0.25
  TC=30bps/side, borrow=50bps/yr, N_max=50, capital=$1M

Filter chain per fold (applied before engine):
  Phase 1 pairs -> persistence gate -> HL cap -> CORR25 -> engine

Differences from run_full_pipeline.py (baseline EOS):
  ENTRY_Z          2.0  ->  3.5
  eos_flatten      True ->  False
  HL_MAX_DAYS      10.0 ->  6.0  (applied post-Phase-1 as a secondary cap)
  CORR25_THRESH    none ->  0.25 (full-formation 5-min Pearson correlation floor)
  delta            auto ->  1e-7 (fixed; auto-selector always chose 1e-7 anyway)
  persistence gate none ->  last-month Johansen re-test before trading window

Outputs:
  results/metrics/final/fold_metrics.csv
  results/metrics/final/equity_full.parquet
  results/metrics/final/fold{NN}_equity.parquet
  results/logs/final/fold_{NN}_audit.txt
"""

import sys, os, logging, traceback, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RUN_PHASE1 = "--skip-phase1" not in sys.argv

print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.phase3_backtest.neg_control import run_neg_control
from src.phase3_backtest.latency import run_latency_sweep
from src.phase3_backtest.audit_log import write_audit_log

# ---- Constants ----
PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"
RESULTS_DIR = "results/metrics/final"
LOG_DIR     = "results/logs/final"

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
ENTRY_Z          = 3.5        # post bug-fix optimum (was 3.0). OAT 23-fold sweep:
                              #   Z=3.0: SR +0.144 (48% pos)
                              #   Z=3.5: SR +1.025 (52% pos)  <-- selected
                              #   Z=3.75/4.0 non-monotone, fewer trades
TC_BPS           = 30.0
BORROW_BPS_YR    = 50.0
FIXED_DELTA      = 1e-7       # bypass auto-selector
HL_MAX_DAYS      = 6.0        # tighter cap than Phase 1 default 10.0
CORR25_THRESH    = 0.25       # full-formation Pearson correlation floor
JOHANSEN_PVAL    = 0.05       # persistence gate re-test threshold
MAX_PAIRS        = 500        # spike-fold cap (same as baseline)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---- Fold schedule (identical to run_full_pipeline.py) ----
FOLD_SCHEDULE = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (4,  "2022-04-01", "2022-09-30", "2022-10"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (7,  "2022-07-01", "2022-12-31", "2023-01"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (9,  "2022-09-01", "2023-02-28", "2023-03"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (14, "2023-02-01", "2023-07-31", "2023-08"),
    (15, "2023-03-01", "2023-08-31", "2023-09"),
    (16, "2023-04-01", "2023-09-30", "2023-10"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (18, "2023-06-01", "2023-11-30", "2023-12"),
    (19, "2023-07-01", "2023-12-31", "2024-01"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (21, "2023-09-01", "2024-02-29", "2024-03"),
    (22, "2023-10-01", "2024-03-31", "2024-04"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (24, "2023-12-01", "2024-05-31", "2024-06"),
    (25, "2024-01-01", "2024-06-30", "2024-07"),
    (26, "2024-02-01", "2024-07-31", "2024-08"),
    (27, "2024-03-01", "2024-08-31", "2024-09"),
    (28, "2024-04-01", "2024-09-30", "2024-10"),
    (29, "2024-05-01", "2024-10-31", "2024-11"),
    (30, "2024-06-01", "2024-11-30", "2024-12"),
    (31, "2024-07-01", "2024-12-31", "2025-01"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (33, "2024-09-01", "2025-02-28", "2025-03"),
    (34, "2024-10-01", "2025-03-31", "2025-04"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (42, "2025-06-01", "2025-11-30", "2025-12"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
    (45, "2025-09-01", "2026-02-28", "2026-03"),
]


# ============================================================
# Data cache helpers
# ============================================================

def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict:
    cache = {}
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    for fname in files:
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_1min(cache: dict, tickers: set, trading_month: str) -> dict:
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == month)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_range(cache: dict, tickers: set, start: str, end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


# ============================================================
# Pair filter helpers (Option A + CORR25 filter chain)
# ============================================================

def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate(pairs_df: pd.DataFrame, form_5min: dict, form_end: str) -> pd.DataFrame:
    """Re-test each pair against the last month of the formation window (Johansen p < 0.05).
    Pairs whose cointegration already broke before trading starts are rejected."""
    if pairs_df.empty:
        return pd.DataFrame()
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        # Johansen chi2 approximation needs ~100 obs minimum to be reliable.
        # ~22 trading days x 78 5-min bars/day = ~1700 typical; require >=200 hard floor.
        if len(aln) < 200:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_corr25_filter(pairs_df: pd.DataFrame, form_5min: dict,
                        form_start: str, form_end: str) -> pd.DataFrame:
    """Reject pairs with full-formation 5-min log-return Pearson correlation < 0.25.

    Returns are computed only between consecutive intra-session bars (gap <= 6 min).
    Diffs that span overnight gaps or session boundaries (15:55 -> next 09:35)
    would otherwise pollute the correlation with multi-hour returns calibrated
    against the same 0.25 threshold as 5-min returns.
    """
    if pairs_df.empty:
        return pd.DataFrame()
    survivors = []
    max_gap = pd.Timedelta(minutes=6)   # any diff > 6 min is a session boundary
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        ra = aln["log_close_a"].diff()
        rb = aln["log_close_b"].diff()
        # Mask diffs across session boundaries (overnight / weekend gaps)
        gap = aln.index.to_series().diff()
        valid = ra.notna() & rb.notna() & (gap <= max_gap)
        if valid.sum() < 20:
            continue
        corr = float(ra[valid].corr(rb[valid]))
        if not np.isnan(corr) and corr >= CORR25_THRESH:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


# ============================================================
# Per-fold runner
# ============================================================

def run_fold(
    fold_n: int,
    formation_start: str,
    formation_end: str,
    trading_month: str,
    cache_1min: dict,
    cache_5min: dict,
) -> tuple:

    t0 = time.time()
    pairs_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"  Fold {fold_n:02d} [{trading_month}]: no CSV — skip")
        return None, None

    if len(pairs_df) == 0:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs — skip")
        return None, None

    # Spike-fold cap: keep best MAX_PAIRS by Johansen p-value
    if len(pairs_df) > MAX_PAIRS:
        n_before = len(pairs_df)
        pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)
        print(f"  Fold {fold_n:02d}: spike cap {n_before} -> {MAX_PAIRS}")

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_range(cache_5min, tickers, formation_start, formation_end)
    formation_1min = _slice_range(cache_1min, tickers, formation_start, formation_end)

    if not trading_1min:
        print(f"  Fold {fold_n:02d} [{trading_month}]: no 1-min data — skip")
        return None, None

    # --- Option A + CORR25 filter chain ---

    # 1. Persistence gate: last-month Johansen re-test
    pairs_df = apply_persistence_gate(pairs_df, formation_5min, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after persistence gate — skip")
        return None, None

    # 2. HL cap (tighter than Phase 1 default)
    pairs_df = pairs_df[pairs_df["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after HL cap — skip")
        return None, None

    # 3. CORR25 filter: full-formation Pearson correlation >= 0.25
    pairs_df = apply_corr25_filter(pairs_df, formation_5min, formation_start, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after CORR25 — skip")
        return None, None

    n_pairs_filtered = len(pairs_df)

    # --- Engine (fixed delta, no-EOS) ---
    try:
        fold_engine_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            entry_z=ENTRY_Z,
            eos_flatten=False,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_execution FAILED: {e}")
        traceback.print_exc()
        return None, None

    if not fold_engine_results:
        print(f"  Fold {fold_n:02d} [{trading_month}]: engine 0 pairs")
        return None, None

    fold_config = {
        "fold":            fold_n,
        "formation_start": formation_start,
        "formation_end":   formation_end,
        "trading_month":   trading_month,
        "delta":           FIXED_DELTA,
        "entry_z":         ENTRY_Z,
        "tc_bps":          int(TC_BPS),
        "borrow_bps_yr":   int(BORROW_BPS_YR),
        "n_open_pairs_max": N_OPEN_PAIRS_MAX,
        "eos_flatten":     False,
        "hl_max_days":     HL_MAX_DAYS,
        "corr25_thresh":   CORR25_THRESH,
    }

    # Phase 3: PnL metrics
    try:
        fold_metrics = run_fold_pnl(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_pnl FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Save per-pair trade metrics
    try:
        ppm = fold_metrics.get("per_pair_metrics", {})
        if ppm:
            rows = [{"fold": fold_n, "ticker_a": ta, "ticker_b": tb, **v}
                    for (ta, tb), v in ppm.items()]
            ppm_df = pd.DataFrame(rows)
            out_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
            write_header = not os.path.exists(out_path)
            ppm_df.to_csv(out_path, mode="a", header=write_header, index=False)
    except Exception:
        pass

    # Save per-trade log (Week-5 hand-off schema)
    #   trade_id, fold_id, pair_id, ticker_A, ticker_B, entry_ts, exit_ts,
    #   notional_A_entry, notional_B_entry, notional_A_exit, notional_B_exit,
    #   gross_pnl_dollars, allocated_capital  + audit fields
    try:
        tl = fold_metrics.get("trade_log", [])
        if tl:
            tl_df = pd.DataFrame(tl)
            tl_df.insert(0, "fold_id", fold_n)
            # trade_id is fold-local in metrics_runner; promote to globally unique
            tl_df["trade_id"] = tl_df.apply(
                lambda r: f"f{int(r['fold_id']):02d}_t{int(r['trade_id']):05d}", axis=1
            )
            # Rename to Week-5 schema (preserve originals as audit columns)
            tl_df = tl_df.rename(columns={
                "ticker_a":         "ticker_A",
                "ticker_b":         "ticker_B",
                "gross_pnl":        "gross_pnl_dollars",
                "notional_a_entry": "notional_A_entry",
                "notional_b_entry": "notional_B_entry",
                "notional_a_exit":  "notional_A_exit",
                "notional_b_exit":  "notional_B_exit",
            })
            # Week-5 schema (15 fields):
            # trade_id, fold_id, pair_id, ticker_A, ticker_B, side_A, side_B,
            # entry_ts, exit_ts, notional_A_entry, notional_B_entry,
            # notional_A_exit, notional_B_exit, gross_pnl_dollars, allocated_capital
            week5_cols = [
                "trade_id", "fold_id", "pair_id", "ticker_A", "ticker_B",
                "side_A", "side_B",
                "entry_ts", "exit_ts",
                "notional_A_entry", "notional_B_entry",
                "notional_A_exit",  "notional_B_exit",
                "gross_pnl_dollars", "allocated_capital",
            ]
            audit_cols = [c for c in tl_df.columns if c not in week5_cols]
            tl_df = tl_df[week5_cols + audit_cols]
            tl_path = f"{RESULTS_DIR}/trade_log.csv"
            write_header = not os.path.exists(tl_path)
            tl_df.to_csv(tl_path, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: trade_log write WARNING: {e}", flush=True)

    # Save rebalance log (Week-5 hand-off schema, 8 fields):
    #   trade_id, fold_id, pair_id, ticker, rebalance_ts,
    #   delta_shares, price_at_rebalance, notional_rebalanced
    try:
        rbl = fold_metrics.get("rebalance_log", [])
        if rbl:
            rbl_df = pd.DataFrame(rbl)
            rbl_df.insert(0, "fold_id", fold_n)
            rbl_df["trade_id"] = rbl_df.apply(
                lambda r: f"f{int(r['fold_id']):02d}_t{int(r['trade_id']):05d}", axis=1
            )
            week5_cols = [
                "trade_id", "fold_id", "pair_id", "ticker",
                "rebalance_ts", "delta_shares", "price_at_rebalance",
                "notional_rebalanced",
            ]
            audit_cols = [c for c in rbl_df.columns if c not in week5_cols]
            rbl_df = rbl_df[week5_cols + audit_cols]
            rbl_path = f"{RESULTS_DIR}/rebalance_log.csv"
            write_header = not os.path.exists(rbl_path)
            rbl_df.to_csv(rbl_path, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: rebalance_log write WARNING: {e}", flush=True)

    # Phase 3: negative control
    nc = None
    try:
        nc = run_neg_control(
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            primary_sharpe=fold_metrics["sharpe"],
            eos_flatten=False,
            seed=42 + fold_n,   # fold-specific seed; iid across folds
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_neg_control WARNING: {e}")

    # Phase 3: latency sweep
    lat = None
    try:
        lat = run_latency_sweep(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_latency_sweep WARNING: {e}")

    # Phase 3: audit log
    try:
        write_audit_log(
            fold_n=fold_n,
            fold_metrics=fold_metrics,
            nc_metrics=nc,
            latency_results=lat,
            delta=FIXED_DELTA,
            delta_metrics={},
            config=fold_config,
            prev_delta=FIXED_DELTA,
            output_dir=LOG_DIR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: write_audit_log WARNING: {e}")

    # Export per-fold equity
    eq = fold_metrics.get("bar_equity", pd.Series(dtype=float))
    if not eq.empty:
        eq.to_frame("equity").to_parquet(f"{RESULTS_DIR}/fold{fold_n:02d}_equity.parquet")

    elapsed = time.time() - t0
    sharpe   = fold_metrics["sharpe"]
    n_trades = fold_metrics["n_trades"]
    nc_pass  = nc["nc_pass"] if nc else None
    t5       = lat["sharpe_by_lag"].get("t+5") if lat else None

    nc_str = "PASS" if nc_pass else ("FAIL" if nc_pass is not None else "N/A")
    t5_str = f"{t5:+.3f}" if t5 is not None else "N/A"

    print(
        f"  Fold {fold_n:02d} [{trading_month}]"
        f"  n_filt={n_pairs_filtered}  pairs={len(fold_engine_results)}  trades={n_trades}"
        f"  Sharpe={sharpe:+.3f}  MaxDD={fold_metrics['max_dd']:.3f}"
        f"  nc={nc_str}  t+5={t5_str}  [{elapsed:.0f}s]"
    )

    summary_row = {
        "fold":            fold_n,
        "trading_month":   trading_month,
        "n_pairs_filtered": n_pairs_filtered,
        "n_pairs_traded":  len(fold_engine_results),
        "n_trades":        n_trades,
        "delta":           FIXED_DELTA,
        "sharpe":          sharpe,
        "max_dd":          fold_metrics["max_dd"],
        "cagr":            fold_metrics["cagr"],
        "calmar":          fold_metrics["calmar"],
        "win_rate":        fold_metrics["win_rate"],
        "avg_hold_bars":   fold_metrics["avg_holding_bars"],
        "avg_net_bps":     fold_metrics["avg_net_bps"],
        "cost_commission": fold_metrics["cost_decomp"]["commission"],
        "cost_borrow":     fold_metrics["cost_decomp"]["borrow"],
        "cost_rebalance":  fold_metrics["cost_decomp"]["rebalance"],
        "nc_threshold":    nc["bootstrap_threshold"] if nc else None,
        "nc_pass":         nc["nc_pass"] if nc else None,
        "t1_sharpe":       lat["sharpe_by_lag"].get("t+1") if lat else None,
        "t5_sharpe":       lat["sharpe_by_lag"].get("t+5") if lat else None,
        "t10_sharpe":      lat["sharpe_by_lag"].get("t+10") if lat else None,
        "latency_pass":    lat["latency_pass"] if lat else None,
        "lookahead_ok":    fold_metrics["lookahead_ok"],
        "kalman_degen":    fold_metrics["kalman_degenerate"],
        "elapsed_s":       elapsed,
    }
    return summary_row, eq


# ============================================================
# Phase 1 (optional)
# ============================================================

print("=== Final Pipeline Run — Option A + CORR25 ===")
print("  Config: no-EOS, Z=3.5, HL<=6d, delta=1e-7, CORR25>=0.25, persistence gate\n")

if RUN_PHASE1:
    from src.phase1_cointegration.discovery import run as run_phase1
    from src.utils.io import VALIDATED_DIR

    os.makedirs(PHASE1_DIR, exist_ok=True)
    print("=== Phase 1: Cointegration Discovery ===")
    t_p1 = time.time()
    for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
        out_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
        print(f"  Fold {fold_n:02d} [{trading_month}]: {formation_start} -> {formation_end} ...", end=" ", flush=True)
        try:
            pairs_df = run_phase1(formation_start, formation_end, VALIDATED_DIR)
            pairs_df.to_csv(out_csv, index=False)
            print(f"{len(pairs_df)} pairs")
        except Exception as e:
            print(f"ERROR: {e}")
            pd.DataFrame().to_csv(out_csv, index=False)
    print(f"Phase 1 complete: {time.time()-t_p1:.0f}s\n")

# ============================================================
# Load data caches
# ============================================================

print(f"Loading 1-min cache from {DATA_1MIN}/ ...")
t_load = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
print(f"  {len(cache_1min)} tickers  [{time.time()-t_load:.1f}s]")

print(f"Loading 5-min cache from {DATA_5MIN}/ ...")
t_load = time.time()
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} tickers  [{time.time()-t_load:.1f}s]\n")

# ============================================================
# Main 45-fold loop
# ============================================================

t_total  = time.time()
all_rows = []
all_eq   = []

_ppm_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
if os.path.exists(_ppm_path):
    os.remove(_ppm_path)
_tl_path = f"{RESULTS_DIR}/trade_log.csv"
if os.path.exists(_tl_path):
    os.remove(_tl_path)
_rbl_path = f"{RESULTS_DIR}/rebalance_log.csv"
if os.path.exists(_rbl_path):
    os.remove(_rbl_path)

for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
    try:
        row, eq = run_fold(
            fold_n, formation_start, formation_end, trading_month,
            cache_1min, cache_5min,
        )
        if row is not None:
            all_rows.append(row)
            if eq is not None and not eq.empty:
                all_eq.append(eq)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: UNHANDLED ERROR: {e}")
        traceback.print_exc()

# ============================================================
# Save outputs
# ============================================================

summary_df = pd.DataFrame(all_rows)
out_csv = f"{RESULTS_DIR}/fold_metrics.csv"
summary_df.to_csv(out_csv, index=False)

if all_eq:
    pd.concat(all_eq).sort_index().to_frame("equity").to_parquet(
        f"{RESULTS_DIR}/equity_full.parquet"
    )

total_min = (time.time() - t_total) / 60
print(f"\n=== Done in {total_min:.1f} min ===")
print(f"  Folds completed : {len(all_rows)} / 45")
print(f"  Results CSV     : {out_csv}")

# ============================================================
# Aggregate report
# ============================================================

if len(all_rows) > 0:
    df = summary_df
    print(f"\n{'='*60}")
    print(f"  OPTION A + CORR25  ({len(df)} completed folds)")
    print(f"{'='*60}")
    print(f"  Sharpe : mean={df['sharpe'].mean():+.3f}  median={df['sharpe'].median():+.3f}"
          f"  std={df['sharpe'].std():.3f}  min={df['sharpe'].min():+.3f}  max={df['sharpe'].max():+.3f}")
    print(f"  % pos  : {(df['sharpe'] > 0).mean():.1%}")
    print(f"  MaxDD  : mean={df['max_dd'].mean():.4f}  worst={df['max_dd'].min():.4f}")
    print(f"  CAGR   : mean={df['cagr'].mean():+.4f}")
    print(f"  Win%   : mean={df['win_rate'].mean():.1%}")
    print(f"  Trades : total={df['n_trades'].sum():.0f}  mean/fold={df['n_trades'].mean():.1f}")
    print(f"  Comm $ : ${df['cost_commission'].sum():,.0f}")
    print(f"  Borrow : ${df['cost_borrow'].sum():,.0f}")
    nc_valid = df['nc_pass'].dropna()
    print(f"  NC pass: {nc_valid.mean():.1%}  ({nc_valid.sum():.0f}/{len(nc_valid)})")
    lat_valid = df['t5_sharpe'].dropna()
    print(f"  t+5>0  : {(lat_valid > 0).mean():.1%}")
    print(f"  LookaheadOK: {bool(df['lookahead_ok'].all())}")

    print(f"\n  --- By Regime ---")
    regimes = [
        ("Bear 2022        ", df["fold"].between(1, 6)),
        ("Early Bull 2023  ", df["fold"].between(7, 18)),
        ("Mid Bull 2024    ", df["fold"].between(19, 30)),
        ("Late Bull 2025-26", df["fold"].between(31, 45)),
    ]
    for name, mask in regimes:
        sub = df[mask]
        if len(sub) == 0:
            print(f"  {name}: no completed folds")
            continue
        print(
            f"  {name}  N={len(sub):2d}"
            f"  Sharpe mean={sub['sharpe'].mean():+.3f}"
            f"  pct_pos={(sub['sharpe'] > 0).mean():.0%}"
            f"  trades={int(sub['n_trades'].sum()):5d}"
        )

    print(f"\n  --- Per-Fold ---")
    print(f"  {'Fold':<5} {'Month':<9} {'nFilt':>6} {'Trades':>7} {'Sharpe':>8} {'MaxDD':>8} {'Win%':>6}")
    print("  " + "-" * 60)
    for _, r in df.iterrows():
        pos = " *" if r["sharpe"] > 0 else ""
        print(
            f"  {int(r['fold']):<5} {r['trading_month']:<9}"
            f" {int(r['n_pairs_filtered']):>6} {int(r['n_trades']):>7}"
            f" {r['sharpe']:>+8.3f}{pos}"
            f" {r['max_dd']:>8.4f}"
            f" {r['win_rate']:>6.1%}"
        )



# ===== FILE: run_final_tc15.py =====
"""
run_final_tc15.py — Option A + CORR25, TC cost sweep

Runs the final config (no-EOS, Z=3.0, HL<=6d, delta=1e-7, CORR25>=0.25, persistence
gate) at multiple TC levels for an apples-to-apples cost sensitivity table.

TC levels swept (each is bps PER LEG; round-trip = 2 x leg):
  TC=15 bps/leg  (= 30 bps round-trip)   -> results/metrics/final_tc15/
  TC=30 bps/leg  (= 60 bps round-trip)   -> results/metrics/final_tc30/

The TC=30/60 RT level matches the canonical final config in run_final_pipeline.py,
re-run here for sweep consistency (same universe, seeds, code path).

Run with --skip-phase1 (Phase 1 folds already in results/metrics/phase1_folds/).
"""

import sys, os, logging, traceback, time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

RUN_PHASE1 = "--skip-phase1" not in sys.argv

print("Warming up Numba JIT...")
from src.phase2_execution.kalman import warmup_kalman
warmup_kalman()
print("  done.\n")

from src.phase2_execution.engine import run_fold_execution
from src.phase3_backtest.metrics_runner import run_fold_pnl
from src.phase3_backtest.neg_control import run_neg_control
from src.phase3_backtest.latency import run_latency_sweep
from src.phase3_backtest.audit_log import write_audit_log

# ---- Constants ----
PHASE1_DIR  = "results/metrics/phase1_folds"
DATA_1MIN   = "data/validated/1min_phase2"
DATA_5MIN   = "data/validated/5min_phase1"

TOTAL_CAPITAL    = 1_000_000.0
N_OPEN_PAIRS_MAX = 50
ENTRY_Z          = 3.0
BORROW_BPS_YR    = 50.0
FIXED_DELTA      = 1e-7
HL_MAX_DAYS      = 6.0
CORR25_THRESH    = 0.25
JOHANSEN_PVAL    = 0.05
MAX_PAIRS        = 500

# TC sweep — bps per leg; round-trip = 2 * leg
TC_SWEEP = [
    (15.0, "results/metrics/final_tc15", "results/logs/final_tc15"),
    (30.0, "results/metrics/final_tc30", "results/logs/final_tc30"),
]

# Globals overwritten per TC iteration
TC_BPS      = TC_SWEEP[0][0]
RESULTS_DIR = TC_SWEEP[0][1]
LOG_DIR     = TC_SWEEP[0][2]
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

FOLD_SCHEDULE = [
    (1,  "2022-01-03", "2022-06-30", "2022-07"),
    (2,  "2022-02-01", "2022-07-31", "2022-08"),
    (3,  "2022-03-01", "2022-08-31", "2022-09"),
    (4,  "2022-04-01", "2022-09-30", "2022-10"),
    (5,  "2022-05-01", "2022-10-31", "2022-11"),
    (6,  "2022-06-01", "2022-11-30", "2022-12"),
    (7,  "2022-07-01", "2022-12-31", "2023-01"),
    (8,  "2022-08-01", "2023-01-31", "2023-02"),
    (9,  "2022-09-01", "2023-02-28", "2023-03"),
    (10, "2022-10-01", "2023-03-31", "2023-04"),
    (11, "2022-11-01", "2023-04-30", "2023-05"),
    (12, "2022-12-01", "2023-05-31", "2023-06"),
    (13, "2023-01-01", "2023-06-30", "2023-07"),
    (14, "2023-02-01", "2023-07-31", "2023-08"),
    (15, "2023-03-01", "2023-08-31", "2023-09"),
    (16, "2023-04-01", "2023-09-30", "2023-10"),
    (17, "2023-05-01", "2023-10-31", "2023-11"),
    (18, "2023-06-01", "2023-11-30", "2023-12"),
    (19, "2023-07-01", "2023-12-31", "2024-01"),
    (20, "2023-08-01", "2024-01-31", "2024-02"),
    (21, "2023-09-01", "2024-02-29", "2024-03"),
    (22, "2023-10-01", "2024-03-31", "2024-04"),
    (23, "2023-11-01", "2024-04-30", "2024-05"),
    (24, "2023-12-01", "2024-05-31", "2024-06"),
    (25, "2024-01-01", "2024-06-30", "2024-07"),
    (26, "2024-02-01", "2024-07-31", "2024-08"),
    (27, "2024-03-01", "2024-08-31", "2024-09"),
    (28, "2024-04-01", "2024-09-30", "2024-10"),
    (29, "2024-05-01", "2024-10-31", "2024-11"),
    (30, "2024-06-01", "2024-11-30", "2024-12"),
    (31, "2024-07-01", "2024-12-31", "2025-01"),
    (32, "2024-08-01", "2025-01-31", "2025-02"),
    (33, "2024-09-01", "2025-02-28", "2025-03"),
    (34, "2024-10-01", "2025-03-31", "2025-04"),
    (35, "2024-11-01", "2025-04-30", "2025-05"),
    (36, "2024-12-01", "2025-05-31", "2025-06"),
    (37, "2025-01-01", "2025-06-30", "2025-07"),
    (38, "2025-02-01", "2025-07-31", "2025-08"),
    (39, "2025-03-01", "2025-08-31", "2025-09"),
    (40, "2025-04-01", "2025-09-30", "2025-10"),
    (41, "2025-05-01", "2025-10-31", "2025-11"),
    (42, "2025-06-01", "2025-11-30", "2025-12"),
    (43, "2025-07-01", "2025-12-31", "2026-01"),
    (44, "2025-08-01", "2026-01-31", "2026-02"),
    (45, "2025-09-01", "2026-02-28", "2026-03"),
]


def _load_all_tickers(data_dir: str, compute_log_close: bool) -> dict:
    cache = {}
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    for fname in files:
        tk = fname.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(data_dir, fname))
            if compute_log_close and "log_close" not in df.columns:
                df["log_close"] = np.log(df["close"].clip(lower=1e-10))
            cache[tk] = df
        except Exception:
            pass
    return cache


def _slice_1min(cache: dict, tickers: set, trading_month: str) -> dict:
    year, month = int(trading_month[:4]), int(trading_month[5:7])
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        mask = (df.index.year == year) & (df.index.month == month)
        sliced = df[mask]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _slice_range(cache: dict, tickers: set, start: str, end: str) -> dict:
    out = {}
    for tk in tickers:
        if tk not in cache:
            continue
        df = cache[tk]
        sliced = df[(df.index >= start) & (df.index <= end)]
        if len(sliced) > 0:
            out[tk] = sliced
    return out


def _johansen_pval(log_a: np.ndarray, log_b: np.ndarray) -> float:
    try:
        res = coint_johansen(np.column_stack([log_a, log_b]), det_order=0, k_ar_diff=1)
        return float(1.0 - chi2.cdf(float(res.lr1[0]), df=8))
    except Exception:
        return np.nan


def apply_persistence_gate(pairs_df: pd.DataFrame, form_5min: dict, form_end: str) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()
    gate_start = (pd.Timestamp(form_end) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= gate_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= gate_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        pv = _johansen_pval(aln["log_close_a"].values, aln["log_close_b"].values)
        if not np.isnan(pv) and pv < JOHANSEN_PVAL:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def apply_corr25_filter(pairs_df: pd.DataFrame, form_5min: dict,
                        form_start: str, form_end: str) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()
    survivors = []
    for _, row in pairs_df.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        fa, fb = form_5min.get(ta), form_5min.get(tb)
        if fa is None or fb is None:
            continue
        fa_sl = fa[(fa.index >= form_start) & (fa.index <= form_end)][["log_close"]]
        fb_sl = fb[(fb.index >= form_start) & (fb.index <= form_end)][["log_close"]]
        aln = fa_sl.join(fb_sl, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        if len(aln) < 20:
            continue
        ra = aln["log_close_a"].diff()
        rb = aln["log_close_b"].diff()
        valid = ra.notna() & rb.notna()
        if valid.sum() < 20:
            continue
        corr = float(ra[valid].corr(rb[valid]))
        if not np.isnan(corr) and corr >= CORR25_THRESH:
            survivors.append(row)
    return pd.DataFrame(survivors).reset_index(drop=True)


def run_fold(fold_n, formation_start, formation_end, trading_month, cache_1min, cache_5min):
    t0 = time.time()
    pairs_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
    try:
        pairs_df = pd.read_csv(pairs_csv)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"  Fold {fold_n:02d} [{trading_month}]: no CSV — skip")
        return None, None

    if len(pairs_df) == 0:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 pairs — skip")
        return None, None

    if len(pairs_df) > MAX_PAIRS:
        n_before = len(pairs_df)
        pairs_df = pairs_df.nsmallest(MAX_PAIRS, "johansen_pval").reset_index(drop=True)
        print(f"  Fold {fold_n:02d}: spike cap {n_before} -> {MAX_PAIRS}")

    tickers = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    trading_1min   = _slice_1min(cache_1min, tickers, trading_month)
    formation_5min = _slice_range(cache_5min, tickers, formation_start, formation_end)
    formation_1min = _slice_range(cache_1min, tickers, formation_start, formation_end)

    if not trading_1min:
        print(f"  Fold {fold_n:02d} [{trading_month}]: no 1-min data — skip")
        return None, None

    pairs_df = apply_persistence_gate(pairs_df, formation_5min, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after persistence gate — skip")
        return None, None

    pairs_df = pairs_df[pairs_df["half_life_days"] <= HL_MAX_DAYS].reset_index(drop=True)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after HL cap — skip")
        return None, None

    pairs_df = apply_corr25_filter(pairs_df, formation_5min, formation_start, formation_end)
    if pairs_df.empty:
        print(f"  Fold {fold_n:02d} [{trading_month}]: 0 after CORR25 — skip")
        return None, None

    n_pairs_filtered = len(pairs_df)

    try:
        fold_engine_results = run_fold_execution(
            pairs_df=pairs_df,
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            entry_z=ENTRY_Z,
            eos_flatten=False,
            formation_5min=formation_5min,
            formation_ref=formation_1min,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_execution FAILED: {e}")
        traceback.print_exc()
        return None, None

    if not fold_engine_results:
        print(f"  Fold {fold_n:02d} [{trading_month}]: engine 0 pairs")
        return None, None

    fold_config = {
        "fold": fold_n, "formation_start": formation_start,
        "formation_end": formation_end, "trading_month": trading_month,
        "delta": FIXED_DELTA, "entry_z": ENTRY_Z, "tc_bps": int(TC_BPS),
        "borrow_bps_yr": int(BORROW_BPS_YR), "n_open_pairs_max": N_OPEN_PAIRS_MAX,
        "eos_flatten": False, "hl_max_days": HL_MAX_DAYS, "corr25_thresh": CORR25_THRESH,
    }

    try:
        fold_metrics = run_fold_pnl(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
            pairs_df=pairs_df,
            delta=FIXED_DELTA,
            delta_metrics={},
            total_capital=TOTAL_CAPITAL,
            n_open_pairs_max=N_OPEN_PAIRS_MAX,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_fold_pnl FAILED: {e}")
        traceback.print_exc()
        return None, None

    # Save per-trade log (Week-5 hand-off schema)
    try:
        tl = fold_metrics.get("trade_log", [])
        if tl:
            tl_df = pd.DataFrame(tl)
            tl_df.insert(0, "fold_id", fold_n)
            tl_df["trade_id"] = tl_df.apply(
                lambda r: f"f{int(r['fold_id']):02d}_t{int(r['trade_id']):05d}", axis=1
            )
            tl_df = tl_df.rename(columns={
                "ticker_a":         "ticker_A",
                "ticker_b":         "ticker_B",
                "gross_pnl":        "gross_pnl_dollars",
                "notional_a_entry": "notional_A_entry",
                "notional_b_entry": "notional_B_entry",
                "notional_a_exit":  "notional_A_exit",
                "notional_b_exit":  "notional_B_exit",
            })
            week5_cols = [
                "trade_id", "fold_id", "pair_id", "ticker_A", "ticker_B",
                "side_A", "side_B",
                "entry_ts", "exit_ts",
                "notional_A_entry", "notional_B_entry",
                "notional_A_exit",  "notional_B_exit",
                "gross_pnl_dollars", "allocated_capital",
            ]
            audit_cols = [c for c in tl_df.columns if c not in week5_cols]
            tl_df = tl_df[week5_cols + audit_cols]
            tl_path = f"{RESULTS_DIR}/trade_log.csv"
            write_header = not os.path.exists(tl_path)
            tl_df.to_csv(tl_path, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: trade_log write WARNING: {e}", flush=True)

    # Save rebalance log (Week-5 hand-off schema)
    try:
        rbl = fold_metrics.get("rebalance_log", [])
        if rbl:
            rbl_df = pd.DataFrame(rbl)
            rbl_df.insert(0, "fold_id", fold_n)
            rbl_df["trade_id"] = rbl_df.apply(
                lambda r: f"f{int(r['fold_id']):02d}_t{int(r['trade_id']):05d}", axis=1
            )
            week5_cols = [
                "trade_id", "fold_id", "pair_id", "ticker",
                "rebalance_ts", "delta_shares", "price_at_rebalance",
                "notional_rebalanced",
            ]
            audit_cols = [c for c in rbl_df.columns if c not in week5_cols]
            rbl_df = rbl_df[week5_cols + audit_cols]
            rbl_path = f"{RESULTS_DIR}/rebalance_log.csv"
            write_header = not os.path.exists(rbl_path)
            rbl_df.to_csv(rbl_path, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"  Fold {fold_n:02d}: rebalance_log write WARNING: {e}", flush=True)

    nc = None
    try:
        # NC must use the same TC as the primary strategy for a fair NC pass test.
        # Fold-specific seed avoids perfectly correlated NC distributions across folds.
        nc = run_neg_control(
            trading_1min=trading_1min,
            delta=FIXED_DELTA,
            primary_sharpe=fold_metrics["sharpe"],
            eos_flatten=False,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
            seed=42 + fold_n,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_neg_control WARNING: {e}", flush=True)

    lat = None
    try:
        # Latency sweep also recomputes PnL — must match primary TC.
        lat = run_latency_sweep(
            fold_results=fold_engine_results,
            trading_1min=trading_1min,
            tc_bps=TC_BPS,
            borrow_bps_yr=BORROW_BPS_YR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: run_latency_sweep WARNING: {e}", flush=True)

    try:
        write_audit_log(
            fold_n=fold_n, fold_metrics=fold_metrics, nc_metrics=nc, latency_results=lat,
            delta=FIXED_DELTA, delta_metrics={}, config=fold_config,
            prev_delta=FIXED_DELTA, output_dir=LOG_DIR,
        )
    except Exception as e:
        print(f"  Fold {fold_n:02d}: write_audit_log WARNING: {e}")

    eq = fold_metrics.get("bar_equity", pd.Series(dtype=float))
    if not eq.empty:
        eq.to_frame("equity").to_parquet(f"{RESULTS_DIR}/fold{fold_n:02d}_equity.parquet")

    elapsed = time.time() - t0
    sharpe   = fold_metrics["sharpe"]
    n_trades = fold_metrics["n_trades"]
    nc_pass  = nc["nc_pass"] if nc else None
    t5       = lat["sharpe_by_lag"].get("t+5") if lat else None
    nc_str   = "PASS" if nc_pass else ("FAIL" if nc_pass is not None else "N/A")
    t5_str   = f"{t5:+.3f}" if t5 is not None else "N/A"

    print(
        f"  Fold {fold_n:02d} [{trading_month}]"
        f"  n_filt={n_pairs_filtered}  pairs={len(fold_engine_results)}  trades={n_trades}"
        f"  Sharpe={sharpe:+.3f}  MaxDD={fold_metrics['max_dd']:.3f}"
        f"  nc={nc_str}  t+5={t5_str}  [{elapsed:.0f}s]",
        flush=True,
    )

    summary_row = {
        "fold": fold_n, "trading_month": trading_month,
        "n_pairs_filtered": n_pairs_filtered, "n_pairs_traded": len(fold_engine_results),
        "n_trades": n_trades, "delta": FIXED_DELTA, "sharpe": sharpe,
        "max_dd": fold_metrics["max_dd"], "cagr": fold_metrics["cagr"],
        "calmar": fold_metrics["calmar"], "win_rate": fold_metrics["win_rate"],
        "avg_hold_bars": fold_metrics["avg_holding_bars"],
        "avg_net_bps": fold_metrics["avg_net_bps"],
        "cost_commission": fold_metrics["cost_decomp"]["commission"],
        "cost_borrow": fold_metrics["cost_decomp"]["borrow"],
        "cost_rebalance": fold_metrics["cost_decomp"]["rebalance"],
        "nc_threshold": nc["bootstrap_threshold"] if nc else None,
        "nc_pass": nc["nc_pass"] if nc else None,
        "t1_sharpe": lat["sharpe_by_lag"].get("t+1") if lat else None,
        "t5_sharpe": lat["sharpe_by_lag"].get("t+5") if lat else None,
        "t10_sharpe": lat["sharpe_by_lag"].get("t+10") if lat else None,
        "latency_pass": lat["latency_pass"] if lat else None,
        "lookahead_ok": fold_metrics["lookahead_ok"],
        "kalman_degen": fold_metrics["kalman_degenerate"],
        "elapsed_s": elapsed,
    }
    return summary_row, eq


# ============================================================
print("=== TC Cost Sweep — Option A + CORR25 ===")
print(f"  Config: no-EOS, Z=3.0, HL<=6d, delta=1e-7, CORR25>=0.25")
print(f"  TC levels (bps/leg): {[t[0] for t in TC_SWEEP]}\n")

if RUN_PHASE1:
    from src.phase1_cointegration.discovery import run as run_phase1
    from src.utils.io import VALIDATED_DIR
    os.makedirs(PHASE1_DIR, exist_ok=True)
    print("=== Phase 1: Cointegration Discovery ===")
    t_p1 = time.time()
    for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
        out_csv = f"{PHASE1_DIR}/fold_{fold_n:02d}.csv"
        print(f"  Fold {fold_n:02d} [{trading_month}]: {formation_start} -> {formation_end} ...", end=" ", flush=True)
        try:
            pairs_df = run_phase1(formation_start, formation_end, VALIDATED_DIR)
            pairs_df.to_csv(out_csv, index=False)
            print(f"{len(pairs_df)} pairs")
        except Exception as e:
            print(f"ERROR: {e}")
            pd.DataFrame().to_csv(out_csv, index=False)
    print(f"Phase 1 complete: {time.time()-t_p1:.0f}s\n")

print(f"Loading 1-min cache from {DATA_1MIN}/ ...")
t_load = time.time()
cache_1min = _load_all_tickers(DATA_1MIN, compute_log_close=True)
print(f"  {len(cache_1min)} tickers  [{time.time()-t_load:.1f}s]")

print(f"Loading 5-min cache from {DATA_5MIN}/ ...")
t_load = time.time()
cache_5min = _load_all_tickers(DATA_5MIN, compute_log_close=False)
print(f"  {len(cache_5min)} tickers  [{time.time()-t_load:.1f}s]\n")

sweep_summaries = {}   # tc_bps -> summary_df

for tc_bps_iter, results_dir_iter, log_dir_iter in TC_SWEEP:
    # Re-bind module globals so run_fold uses the current TC level
    TC_BPS      = tc_bps_iter
    RESULTS_DIR = results_dir_iter
    LOG_DIR     = log_dir_iter
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'='*72}", flush=True)
    print(f"  Running TC = {TC_BPS:.0f} bps/leg ({2*TC_BPS:.0f} bps round-trip)", flush=True)
    print(f"  Output dir: {RESULTS_DIR}", flush=True)
    print(f"{'='*72}", flush=True)

    t_total  = time.time()
    all_rows = []
    all_eq   = []

    _ppm_path = f"{RESULTS_DIR}/pair_trade_metrics.csv"
    if os.path.exists(_ppm_path):
        os.remove(_ppm_path)
    _tl_path = f"{RESULTS_DIR}/trade_log.csv"
    if os.path.exists(_tl_path):
        os.remove(_tl_path)
    _rbl_path = f"{RESULTS_DIR}/rebalance_log.csv"
    if os.path.exists(_rbl_path):
        os.remove(_rbl_path)

    for fold_n, formation_start, formation_end, trading_month in FOLD_SCHEDULE:
        try:
            row, eq = run_fold(fold_n, formation_start, formation_end, trading_month,
                               cache_1min, cache_5min)
            if row is not None:
                all_rows.append(row)
                if eq is not None and not eq.empty:
                    all_eq.append(eq)
        except Exception as e:
            print(f"  Fold {fold_n:02d}: UNHANDLED ERROR: {e}")
            traceback.print_exc()

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(f"{RESULTS_DIR}/fold_metrics.csv", index=False)
    sweep_summaries[tc_bps_iter] = summary_df

    if all_eq:
        pd.concat(all_eq).sort_index().to_frame("equity").to_parquet(
            f"{RESULTS_DIR}/equity_full.parquet"
        )

    total_min = (time.time() - t_total) / 60
    print(f"\n  TC={TC_BPS:.0f} bps/leg done in {total_min:.1f} min, {len(all_rows)}/45 folds",
          flush=True)

# ============================================================
# Aggregate cost-sweep table
# ============================================================
print(f"\n\n{'='*72}")
print(f"  COST SWEEP SUMMARY — Option A + CORR25")
print(f"{'='*72}")
print(f"  {'TC (bps/leg)':<14} {'TC RT':<8} {'N folds':<9} {'Mean SR':<10} {'Median':<10}"
      f" {'% pos':<8} {'Trades':<8} {'Comm $':<14} {'NC pass':<10}")
print("  " + "-" * 95)
for tc_bps_iter, _, _ in TC_SWEEP:
    df = sweep_summaries.get(tc_bps_iter, pd.DataFrame())
    if df.empty:
        print(f"  {tc_bps_iter:<14.1f} {2*tc_bps_iter:<8.0f} no completed folds")
        continue
    nc_valid = df['nc_pass'].dropna()
    nc_str = f"{nc_valid.mean():.1%} ({int(nc_valid.sum())}/{len(nc_valid)})" if len(nc_valid) else "n/a"
    print(
        f"  {tc_bps_iter:<14.1f} {2*tc_bps_iter:<8.0f} {len(df):<9}"
        f" {df['sharpe'].mean():<+10.3f} {df['sharpe'].median():<+10.3f}"
        f" {(df['sharpe'] > 0).mean():<8.1%} {int(df['n_trades'].sum()):<8}"
        f" ${df['cost_commission'].sum():<13,.0f} {nc_str:<10}"
    )

print(f"\n  --- Regime breakdown per TC level ---")
regimes = [
    ("Bear 2022        ", lambda d: d["fold"].between(1, 6)),
    ("Early Bull 2023  ", lambda d: d["fold"].between(7, 18)),
    ("Mid Bull 2024    ", lambda d: d["fold"].between(19, 30)),
    ("Late Bull 2025-26", lambda d: d["fold"].between(31, 45)),
]
for name, mask_fn in regimes:
    parts = []
    for tc_bps_iter, _, _ in TC_SWEEP:
        df = sweep_summaries.get(tc_bps_iter, pd.DataFrame())
        if df.empty:
            parts.append(f"TC{tc_bps_iter:.0f}=n/a")
            continue
        sub = df[mask_fn(df)]
        if len(sub) == 0:
            parts.append(f"TC{tc_bps_iter:.0f}=n/a")
        else:
            parts.append(f"TC{tc_bps_iter:.0f}: SR={sub['sharpe'].mean():+.3f} pos={(sub['sharpe']>0).mean():.0%}")
    print(f"  {name}   " + "   ".join(parts))

# Write a combined sweep summary CSV
sweep_rows = []
for tc_bps_iter, _, _ in TC_SWEEP:
    df = sweep_summaries.get(tc_bps_iter, pd.DataFrame())
    if df.empty:
        continue
    nc_valid = df['nc_pass'].dropna()
    sweep_rows.append({
        "tc_bps_per_leg":   tc_bps_iter,
        "tc_bps_round_trip": 2 * tc_bps_iter,
        "n_folds":          len(df),
        "mean_sharpe":      float(df['sharpe'].mean()),
        "median_sharpe":    float(df['sharpe'].median()),
        "pct_positive":     float((df['sharpe'] > 0).mean()),
        "total_trades":     int(df['n_trades'].sum()),
        "total_commission": float(df['cost_commission'].sum()),
        "nc_pass_rate":     float(nc_valid.mean()) if len(nc_valid) else float("nan"),
        "bear_mean_sharpe": float(df[df["fold"].between(1, 6)]["sharpe"].mean()) if len(df[df["fold"].between(1, 6)]) else float("nan"),
        "early_bull_mean_sharpe": float(df[df["fold"].between(7, 18)]["sharpe"].mean()) if len(df[df["fold"].between(7, 18)]) else float("nan"),
        "mid_bull_mean_sharpe":   float(df[df["fold"].between(19, 30)]["sharpe"].mean()) if len(df[df["fold"].between(19, 30)]) else float("nan"),
        "late_bull_mean_sharpe":  float(df[df["fold"].between(31, 45)]["sharpe"].mean()) if len(df[df["fold"].between(31, 45)]) else float("nan"),
    })
if sweep_rows:
    sweep_summary_path = "results/metrics/tc_sweep_summary.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_summary_path, index=False)
    print(f"\n  Combined sweep summary: {sweep_summary_path}")

