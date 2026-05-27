"""Smoke: resample 5-min to daily and verify NYSE session alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEEK6 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WEEK6))

from engine_daily.data_daily import resample_5min_to_daily, load_daily_universe


def _make_synth_5min_session(date: str, n_bars: int = 78, p0: float = 100.0):
    """Synthetic 5-min bars for one regular session 09:35-16:00 ET."""
    start = pd.Timestamp(f"{date} 09:35", tz="US/Eastern")
    idx = pd.date_range(start, periods=n_bars, freq="5min")
    rng = np.random.default_rng(42)
    log_close = np.log(p0) + np.cumsum(rng.normal(0, 0.001, n_bars))
    vol = rng.integers(1000, 10000, n_bars).astype(float)
    return pd.DataFrame({"log_close": log_close, "volume": vol}, index=idx)


def test_single_session():
    """One 5-min session -> one daily bar with last log_close + sum volume."""
    df = _make_synth_5min_session("2024-01-02", n_bars=78, p0=100.0)
    daily = resample_5min_to_daily(df)
    assert len(daily) == 1, f"expected 1 daily bar, got {len(daily)}"
    assert np.isclose(daily.iloc[0]["log_close"], df["log_close"].iloc[-1]), \
        "daily log_close must equal last 5-min log_close"
    assert daily.iloc[0]["volume"] == df["volume"].sum(), \
        "daily volume must equal sum of 5-min volumes"
    # Index must be tz-naive Timestamp at midnight
    assert daily.index.tz is None, "daily index should be tz-naive"
    assert daily.index[0] == pd.Timestamp("2024-01-02"), \
        f"index should be 2024-01-02, got {daily.index[0]}"
    print(f"[PASS] single_session: 78 5-min bars -> 1 daily bar @ {daily.index[0].date()}")


def test_multi_session():
    """Five sessions across a week -> five daily bars."""
    dfs = []
    for date in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]:
        dfs.append(_make_synth_5min_session(date, n_bars=78, p0=100.0))
    df = pd.concat(dfs)
    daily = resample_5min_to_daily(df)
    assert len(daily) == 5, f"expected 5 daily bars, got {len(daily)}"
    expected_dates = [pd.Timestamp(d) for d in
                      ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]]
    assert list(daily.index) == expected_dates, "dates not in order"
    print(f"[PASS] multi_session: 5 sessions -> 5 daily bars in order")


def test_half_day():
    """Half-day session (early close 13:00) -> still one daily bar with last bar's log_close."""
    df = _make_synth_5min_session("2024-11-29", n_bars=42, p0=100.0)  # 09:35 - 13:00 = 42 bars
    daily = resample_5min_to_daily(df)
    assert len(daily) == 1, "half-day -> 1 daily bar"
    assert np.isclose(daily.iloc[0]["log_close"], df["log_close"].iloc[-1])
    print(f"[PASS] half_day: 42 5-min bars (early close) -> 1 daily bar")


def test_real_data_aapl():
    """Real 5-min data: AAPL 2022-01 should have ~20 trading days."""
    pq = WEEK6.parents[0] / "Week 4" / "data" / "validated" / "5min_phase1" / "AAPL.parquet"
    if not pq.exists():
        print("[SKIP] real_data_aapl: parquet not found")
        return
    import pandas as pd
    df = pd.read_parquet(pq)
    # Slice to Jan 2022
    jan = df[(df.index >= "2022-01-01") & (df.index < "2022-02-01")]
    daily = resample_5min_to_daily(jan)
    n_trading_days_jan_2022 = 20  # 20 trading days in Jan 2022 (NYSE)
    assert len(daily) == n_trading_days_jan_2022, \
        f"AAPL Jan 2022 should have {n_trading_days_jan_2022} daily bars, got {len(daily)}"
    assert (daily["log_close"] > 4.0).all() and (daily["log_close"] < 6.0).all(), \
        "AAPL log-prices outside sanity band"
    print(f"[PASS] real_data_aapl: AAPL Jan 2022 -> {len(daily)} daily bars, "
          f"log_close range [{daily['log_close'].min():.3f}, {daily['log_close'].max():.3f}]")


def test_load_universe():
    """End-to-end: build a small 2-ticker universe and slice to a window."""
    cache = {
        "AAA": pd.concat([
            _make_synth_5min_session("2024-01-02"),
            _make_synth_5min_session("2024-01-03"),
            _make_synth_5min_session("2024-01-04"),
        ]),
        "BBB": pd.concat([
            _make_synth_5min_session("2024-01-02"),
            _make_synth_5min_session("2024-01-04"),  # missing 01-03
        ]),
    }
    daily_uni = load_daily_universe(cache, "2024-01-02", "2024-01-04")
    assert set(daily_uni.keys()) == {"AAA", "BBB"}
    assert len(daily_uni["AAA"]) == 3
    assert len(daily_uni["BBB"]) == 2
    print(f"[PASS] load_universe: 2 tickers, AAA=3d, BBB=2d (missing 01-03 dropped)")


if __name__ == "__main__":
    test_single_session()
    test_multi_session()
    test_half_day()
    test_real_data_aapl()
    test_load_universe()
    print("\ntest_data_resample: ALL PASSED")
