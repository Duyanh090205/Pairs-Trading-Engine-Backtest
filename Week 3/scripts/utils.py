"""
Data Utilities Module - Week 3
Handles loading, assertions, session formatting and bias injection for D1.
"""

import os
import glob
import numpy as np
import pandas as pd

EASTERN       = "America/New_York"
SESSION_START = 9 * 60 + 30    # 09:30 ET in minutes
SESSION_END   = 15 * 60 + 59   # 15:59 ET in minutes

# CMS/DUK OLS params (verified by verify_params.py)
_CMS_DUK_ALPHA = -0.695816
_CMS_DUK_BETA  =  1.048794


# ---------------------------------------------------------------------------
# CP1 — Load and verify clean data
# ---------------------------------------------------------------------------

def load_and_verify_clean_data(raw_dir):
    """
    Load all raw CSVs, convert timestamps, session-filter, sort, run 6 assertions.

    Returns
    -------
    df : pd.DataFrame
        All tickers, all session bars, ts_utc as DatetimeIndex.
    summary : dict
        {total_files, tickers, date_range, total_bars, assertions}
    """
    # 1. Collect and load all files
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # 2. Convert window_start (nanoseconds) to UTC + ET
    df["ts_utc"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    df["ts_et"]  = df["ts_utc"].dt.tz_convert(EASTERN)

    # 3. Sort by (ticker, ts_utc)
    df = df.sort_values(["ticker", "ts_utc"]).reset_index(drop=True)

    # 4. Session filter: 09:30–15:59 ET
    mins = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df = df[(mins >= SESSION_START) & (mins <= SESSION_END)].copy()
    df = df.reset_index(drop=True)

    # 5. Set index to ts_utc
    df = df.set_index("ts_utc")

    # 6. Run 6 assertions
    assertions = {}

    # A1: monotonic timestamps per ticker
    mono = df.groupby("ticker").apply(
        lambda g: g.index.is_monotonic_increasing, include_groups=False
    )
    assertions["A1_monotonic"]  = bool(mono.all())

    # A2: no duplicate (ticker, ts)
    dup = df.reset_index().duplicated(subset=["ticker", "ts_utc"])
    assertions["A2_no_duplicates"] = bool(~dup.any())

    # A3: high >= close >= low
    assertions["A3_ohlc_close"] = bool(
        (df["high"] >= df["close"]).all() and (df["close"] >= df["low"]).all()
    )

    # A4: high >= open >= low
    assertions["A4_ohlc_open"] = bool(
        (df["high"] >= df["open"]).all() and (df["open"] >= df["low"]).all()
    )

    # A5: no NaN in OHLC
    assertions["A5_no_nan"] = bool(
        ~df[["open", "close", "high", "low"]].isna().any().any()
    )

    # A6: sufficient bars per (ticker, day) — flag days < 300
    df["_date"] = df["ts_et"].dt.date
    bars_per_day = df.groupby(["ticker", "_date"]).size()
    assertions["A6_bars_per_day"] = {
        "min": int(bars_per_day.min()),
        "max": int(bars_per_day.max()),
        "median": float(bars_per_day.median()),
        "days_under_300": int((bars_per_day < 300).sum()),
        "pass": bool((bars_per_day < 100).sum() == 0),  # hard fail only if <100 bars
    }
    df = df.drop(columns=["_date"])

    summary = {
        "total_files":  len(files),
        "tickers":      sorted(df["ticker"].unique().tolist()),
        "n_tickers":    df["ticker"].nunique(),
        "date_range":   (str(df["ts_et"].dt.date.min()), str(df["ts_et"].dt.date.max())),
        "total_bars":   len(df),
        "assertions":   assertions,
    }

    # Print summary
    print("\n===== CP1 / CP4 — Data Pipeline Summary =====")
    print(f"  Total files loaded : {summary['total_files']}")
    print(f"  Tickers            : {summary['tickers']}")
    print(f"  Date range         : {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"  Total bars (session-filtered): {summary['total_bars']:,}")
    print("  Assertions:")
    for k, v in assertions.items():
        if k == "A6_bars_per_day":
            status = "[PASS]" if v["pass"] else "[FAIL]"
            print(f"    {k}: {status}  min={v['min']}  max={v['max']}  "
                  f"median={v['median']:.0f}  days<300={v['days_under_300']}")
        else:
            status = "[PASS]" if v else "[FAIL]"
            print(f"    {k}: {status}")
    print("=" * 46)

    return df, summary


# ---------------------------------------------------------------------------
# CP3 — H1: Random future-close substitution
# ---------------------------------------------------------------------------

def inject_h1_future_close(df, k_percent, seed=42):
    """
    Replace close[t] with close[t+1] for k% of rows per ticker.
    Simulates vendor buffer-delay where next bar's close is written to current bar.
    Works on integer positions within each ticker group to avoid duplicate-index issues.
    """
    rng    = np.random.default_rng(seed)
    result = df.reset_index()   # work with integer positions

    n_injected = 0
    # Record (int_pos, expected_close) for validation
    validation_pairs = []

    for ticker in sorted(result["ticker"].unique()):
        mask      = result["ticker"] == ticker
        int_pos   = result.index[mask].tolist()  # integer positions in result
        eligible  = int_pos[:-1]                 # all but last bar
        n_inject  = max(1, int(len(eligible) * k_percent / 100))
        chosen    = rng.choice(len(eligible), size=n_inject, replace=False)

        for c in chosen:
            pos_i    = eligible[c]
            pos_next = int_pos[int_pos.index(pos_i) + 1]
            next_close = result.at[pos_next, "close"]
            result.at[pos_i, "close"] = next_close
            validation_pairs.append((pos_i, next_close))

        n_injected += n_inject

    # Validate: spot-check first 20 injected pairs
    for pos_i, expected_close in validation_pairs[:20]:
        actual = result.at[pos_i, "close"]
        assert abs(actual - expected_close) < 1e-9, \
            f"H1 validation failed at int_pos={pos_i}"

    result = result.set_index("ts_utc")
    print(f"  H1 k={k_percent}%: injected {n_injected} bars ({len(df)} total)")
    return result


# ---------------------------------------------------------------------------
# CP3 — H2: Timestamp backdating
# ---------------------------------------------------------------------------

def inject_h2_timestamp_backdating(df, k_percent, seed=42):
    """
    Subtract 60 seconds from window_start for k% of rows (excluding first bar each day).
    Simulates vendor using bar-open instead of bar-close timestamp.
    """
    rng    = np.random.default_rng(seed)
    result = df.copy().reset_index()   # work with ts_utc as column

    # Build exclusion mask: first bar of each (ticker, date)
    result["_date"] = result["ts_utc"].dt.tz_convert(EASTERN).dt.date
    first_bars = result.groupby(["ticker", "_date"])["ts_utc"].transform("min")
    is_first   = result["ts_utc"] == first_bars

    eligible_mask = ~is_first
    eligible_pos  = result.index[eligible_mask].tolist()

    n_inject = max(1, int(len(eligible_pos) * k_percent / 100))
    chosen   = rng.choice(len(eligible_pos), size=n_inject, replace=False)
    chosen_pos = [eligible_pos[i] for i in chosen]

    sixty_ns = 60_000_000_000  # 60 seconds in nanoseconds
    result.loc[chosen_pos, "window_start"] -= sixty_ns
    result["ts_utc"] = pd.to_datetime(result["window_start"], unit="ns", utc=True)

    result = (result
              .drop(columns=["_date"])
              .sort_values(["ticker", "window_start"])
              .set_index("ts_utc"))

    print(f"  H2 k={k_percent}%: backdated {n_inject} timestamps by 60s ({len(df)} total)")
    return result


# ---------------------------------------------------------------------------
# CP3 — H3: Spread-level injection
# ---------------------------------------------------------------------------

def inject_h3_spread_level(df, k_percent, seed=42):
    """
    Add spread_biased column. For k% of CMS/DUK timestamps, write S(t+1) at row t.
    Simulates pre-computed feature column cached with wrong lag offset.
    """
    rng = np.random.default_rng(seed)

    # Extract and align CMS/DUK
    cms = df[df["ticker"] == "CMS"][["close"]].rename(columns={"close": "CMS"})
    duk = df[df["ticker"] == "DUK"][["close"]].rename(columns={"close": "DUK"})
    aligned = cms.join(duk, how="inner").dropna()

    # Compute clean spread S(t)
    s_clean = (np.log(aligned["CMS"])
               - _CMS_DUK_ALPHA
               - _CMS_DUK_BETA * np.log(aligned["DUK"]))
    s_clean.name = "spread_biased"

    # Initialise spread_biased column in full df (NaN for non-CMS/DUK rows)
    # Use reset_index to avoid duplicate-ts issues with .loc
    result = df.copy().reset_index()
    result["spread_biased"] = np.nan

    # Build a fast lookup: ts_utc -> spread value for CMS rows
    s_clean_dict = s_clean.to_dict()

    # Set honest S(t) only on CMS rows (spread defined once per timestamp)
    cms_mask = result["ticker"] == "CMS"
    for i in result.index[cms_mask]:
        ts = result.at[i, "ts_utc"]
        if ts in s_clean_dict:
            result.at[i, "spread_biased"] = s_clean_dict[ts]

    # Inject S(t+1) for k% of injectable timestamps
    ts_list  = s_clean.index.tolist()
    eligible = ts_list[:-1]
    n_inject = max(1, int(len(eligible) * k_percent / 100))
    chosen   = rng.choice(len(eligible), size=n_inject, replace=False)

    # Build reverse lookup: ts -> row indices in result where ticker=="CMS"
    cms_ts_to_int = {result.at[i, "ts_utc"]: i for i in result.index[cms_mask]}

    inject_map = {}
    for c in chosen:
        ts_i    = eligible[c]
        ts_next = ts_list[c + 1]
        future_val = s_clean.loc[ts_next]
        inject_map[ts_i] = future_val
        if ts_i in cms_ts_to_int:
            result.at[cms_ts_to_int[ts_i], "spread_biased"] = future_val

    # Validate: first 10 injected pairs
    for ts_i, expected in list(inject_map.items())[:10]:
        if ts_i in cms_ts_to_int:
            actual = result.at[cms_ts_to_int[ts_i], "spread_biased"]
            assert abs(actual - expected) < 1e-9, f"H3 validation failed at ts={ts_i}"

    result = result.set_index("ts_utc")
    print(f"  H3 k={k_percent}%: injected spread_biased for {n_inject} timestamps")
    return result


# ---------------------------------------------------------------------------
# CP3 — H4: Normalization leak (full-year mean/std)
# ---------------------------------------------------------------------------

def inject_h4_normalization_leak(df, k_percent, seed=42):
    """
    For k% of tickers, normalize close using full-2022 mean/std (leaks future info).
    Simulates StandardScaler().fit_transform(df['close']) on full dataset.
    """
    rng     = np.random.default_rng(seed)
    tickers = sorted(df["ticker"].unique().tolist())

    n_inject = max(1, int(len(tickers) * k_percent / 100))
    chosen_idx     = rng.choice(len(tickers), size=n_inject, replace=False)
    chosen_tickers = [tickers[i] for i in chosen_idx]

    result = df.copy()
    for ticker in chosen_tickers:
        mask         = result["ticker"] == ticker
        close_vals   = result.loc[mask, "close"]
        global_mean  = close_vals.mean()
        global_std   = close_vals.std(ddof=1)
        result.loc[mask, "close"] = (close_vals - global_mean) / global_std

        # Validate
        normed = result.loc[mask, "close"]
        assert abs(normed.mean()) < 0.01,  f"H4 mean check failed for {ticker}"
        assert abs(normed.std() - 1.0) < 0.01, f"H4 std check failed for {ticker}"

    print(f"  H4 k={k_percent}%: normalized {n_inject}/{len(tickers)} tickers: {chosen_tickers}")
    return result
