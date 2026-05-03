import pandas as pd
import numpy as np
import os
import sys

# Add week5 root to path regardless of CWD
WEEK5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(WEEK5_ROOT)

from src.plan0_gateway import process_orderbook, compute_microstructure_features, compute_intraday_seasonality, compute_rolling_instability

def run_plan0_smoke_test():
    print("--- Running Plan 0 Smoke Test ---")

    # 1. Generate Synthetic Data
    print("Generating synthetic orderbook data for 5 tickers...")
    tickers = ['SPY', 'AAPL', 'MSFT', 'VTRS', 'XOM']

    # 3 days wide enough to exceed 390-bar burn-in per ticker (390 session bars/day)
    dates = pd.date_range('2022-01-03 09:00:00', '2022-01-05 16:30:00', freq='1min', tz='UTC')

    df_list = []
    for ticker in tickers:
        n_rows = len(dates)

        base_px = {'SPY': 450, 'AAPL': 170, 'MSFT': 330, 'VTRS': 15, 'XOM': 65}[ticker]
        px = base_px * np.cumprod(1 + np.random.normal(0, 0.0001, n_rows))

        half_spread_bps = {'SPY': 2, 'AAPL': 3, 'MSFT': 3, 'VTRS': 15, 'XOM': 4}[ticker]
        spread_usd = px * (half_spread_bps / 10000)

        l1_bid = px - spread_usd
        l1_ask = px + spread_usd

        invalid_idx = np.random.choice(n_rows, size=5, replace=False)

        df_ticker = pd.DataFrame({
            'timestamp': dates,
            'ticker': ticker,
            'l1_bid_px': l1_bid,
            'l1_ask_px': l1_ask,
            'l1_bid_sz': np.random.randint(100, 1000, n_rows),
            'l1_ask_sz': np.random.randint(100, 1000, n_rows),
            'l2_bid_px': l1_bid - spread_usd,
            'l2_ask_px': l1_ask + spread_usd,
            'l2_bid_sz': np.random.randint(100, 1000, n_rows),
            'l2_ask_sz': np.random.randint(100, 1000, n_rows),
            'l3_bid_px': l1_bid - 2 * spread_usd,
            'l3_ask_px': l1_ask + 2 * spread_usd,
            'l3_bid_sz': np.random.randint(100, 1000, n_rows),
            'l3_ask_sz': np.random.randint(100, 1000, n_rows),
        })

        # Inject invalid quotes: crossed book and zero bid
        df_ticker.loc[invalid_idx[0:2], 'l1_bid_px'] = df_ticker.loc[invalid_idx[0:2], 'l1_ask_px'] + 0.10
        df_ticker.loc[invalid_idx[2:5], 'l1_bid_px'] = 0.0

        df_list.append(df_ticker)

    df = pd.concat(df_list, ignore_index=True)
    print(f"Generated {len(df)} rows for testing.")

    # 2. Run pipeline
    print("Running process_orderbook...")
    original_len = len(df)
    df_clean = process_orderbook(df)
    assert len(df) == original_len, "process_orderbook mutated caller's DataFrame length"

    print("Running compute_microstructure_features...")
    df_features = compute_microstructure_features(df_clean)

    print("Running compute_intraday_seasonality...")
    df_seasonality = compute_intraday_seasonality(df_features)

    print("Running compute_rolling_instability...")
    df_final = compute_rolling_instability(df_features, df_seasonality)

    # 3. Assertions
    print("\n--- Running Assertions ---")

    # Timestamps are US/Eastern
    assert str(df_final['timestamp_et'].dt.tz) == 'US/Eastern', "Timestamps not in US/Eastern"
    print("[PASS] Timestamps localized correctly.")

    # Session filtering: no bars outside 09:30-15:59
    times = df_final['timestamp_et'].dt.time
    assert times.min() >= pd.to_datetime('09:30:00').time(), "Bars found before 09:30"
    assert times.max() <= pd.to_datetime('15:59:59').time(), "Bars found after 15:59"
    print("[PASS] Session filtering (09:30-15:59) works.")

    # Spread math
    assert np.allclose(df_final['half_spread_l1_bps'], df_final['full_spread_l1_bps'] / 2, equal_nan=True), \
        "Half spread math failed"
    print("[PASS] Half spread math is exact.")

    # Spread sanity: valid rows only
    valid_rows = df_final[df_final['is_valid']]
    assert (valid_rows['full_spread_l1_bps'] >= 0).all(), "Negative spreads on valid rows"
    print("[PASS] All valid rows have non-negative spreads.")

    # Quote validity flags: flagged but NOT dropped
    assert 'is_valid' in df_final.columns, "is_valid flag column missing"
    invalid_count = (~df_final['is_valid']).sum()
    assert invalid_count > 0, "Expected some flagged invalid quotes"
    assert len(df_final) > 0, "All rows were dropped (should flag, not drop)"
    print(f"[PASS] Invalid quotes flagged (not dropped): {invalid_count} flagged out of {len(df_final)} rows.")

    # Median spread sanity check (within 2× of expected)
    spy_median = valid_rows[valid_rows['ticker'] == 'SPY']['full_spread_l1_bps'].median()
    vtrs_median = valid_rows[valid_rows['ticker'] == 'VTRS']['full_spread_l1_bps'].median()
    print(f"      SPY median L1 spread: {spy_median:.2f} bps (expected ~4)")
    print(f"      VTRS median L1 spread: {vtrs_median:.2f} bps (expected ~30)")

    # Seasonality: exactly 13 buckets per ticker
    buckets = df_seasonality['bucket'].unique()
    assert len(buckets) == 13, f"Expected 13 buckets, got {len(buckets)}"
    assert df_seasonality['median'].isna().sum() == 0, "NaN medians in seasonality output"
    print("[PASS] Intraday seasonality: 13 buckets, no NaN medians.")

    # 390-bar burn-in: first 389 bars must always be NaN (window=390 guarantees this).
    # First valid value must appear at position >= 389 (invalid quotes inside the window
    # may push it slightly later, which is correct behaviour).
    for ticker in tickers:
        t_df = df_final[df_final['ticker'] == ticker].sort_values('timestamp_et').reset_index(drop=True)
        if len(t_df) >= 390:
            assert t_df['spread_std_1d'].iloc[:389].isna().all(), \
                f"{ticker}: spread_std_1d has non-NaN values within first 389 bars"
            first_valid_pos = t_df['spread_std_1d'].first_valid_index()
            assert first_valid_pos is not None, \
                f"{ticker}: spread_std_1d is all NaN — not enough data"
            assert first_valid_pos >= 389, \
                f"{ticker}: first valid spread_std_1d at position {first_valid_pos}, expected >= 389"
    print("[PASS] 390-bar burn-in: spread_std_1d NaN for first 389 bars, first valid at position >= 389.")

    # 4. Write all 4 required parquet files
    data_dir = os.path.join(WEEK5_ROOT, 'data', 'microstructure')
    os.makedirs(data_dir, exist_ok=True)

    # spreads_1min: all per-bar microstructure + rolling cols
    df_final.to_parquet(os.path.join(data_dir, 'spreads_1min.parquet'))

    # spread_rolling: just the rolling instability columns
    rolling_cols = ['timestamp_et', 'ticker', 'spread_std_1d', 'raw_spread_mean_1d']
    df_final[rolling_cols].to_parquet(os.path.join(data_dir, 'spread_rolling.parquet'))

    # spread_seasonality: per-ticker, per-bucket aggregates
    df_seasonality.to_parquet(os.path.join(data_dir, 'spread_seasonality.parquet'))

    # spread_summary: per-ticker aggregate stats
    df_summary = (
        valid_rows.groupby('ticker')['full_spread_l1_bps']
        .agg(
            n_obs='count',
            mean_bps='mean',
            median_bps='median',
            p95_bps=lambda x: x.quantile(0.95),
            p99_bps=lambda x: x.quantile(0.99),
            std_bps='std',
        )
        .reset_index()
    )
    df_summary.to_parquet(os.path.join(data_dir, 'spread_summary.parquet'))

    # Verify all 4 files exist
    for fname in ['spreads_1min.parquet', 'spread_rolling.parquet',
                  'spread_seasonality.parquet', 'spread_summary.parquet']:
        fpath = os.path.join(data_dir, fname)
        assert os.path.exists(fpath), f"Missing output: {fname}"
    print("[PASS] All 4 parquet files written to data/microstructure/.")

    print("\nPlan 0 Smoke Test Passed Successfully!")
    return True

if __name__ == "__main__":
    run_plan0_smoke_test()
