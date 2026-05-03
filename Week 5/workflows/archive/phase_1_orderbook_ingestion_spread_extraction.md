## Phase 1 — Orderbook Ingestion & Spread Extraction

**Input:** `orderbook.parquet`.
**Output:** Per-ticker, per-bar spread features + rolling statistics.

### 1.1 Timestamp Alignment

1. Localize to `US/Eastern` (consistent with Week 4 Phase 0)
2. Session filter: **09:30–15:59 ET** (matching Week 4 Phase 2 execution window)
3. Drop pre-market rows (09:00 ET)
4. **Quote Quality Filters:** Flag rows where `l1_ask_px <= l1_bid_px` (crossed/locked markets) or `mid_px <= 0`. Do NOT drop rows, as this breaks the strict 1-minute execution alignment.
   - **Cost-Alignment Assertion:** For every trade event timestamp, the cost model must find the matching ticker-level spread row. If the row is flagged as invalid, assign a conservative fallback: `fallback_cost = max(static_60bps_leg_equivalent, ticker_p95_dynamic_cost_from_formation)`.

### 1.2 Derived Microstructure Features

Compute per ticker, per bar:

```
mid_px             = (l1_bid_px + l1_ask_px) / 2
full_spread_l1     = l1_ask_px - l1_bid_px
full_spread_l1_bps = (full_spread_l1 / mid_px) × 10,000
half_spread_l1_bps = full_spread_l1_bps / 2

full_spread_l2_bps = ((l2_ask_px - l2_bid_px) / mid_px) × 10,000
full_spread_l3_bps = ((l3_ask_px - l3_bid_px) / mid_px) × 10,000

liquidity_l1       = l1_bid_sz × mid_px    # dollar terms, one side
```

### 1.3 Intraday Spread Seasonality Profile

**No-Lookahead Rule:** All spread seasonality medians used in execution must be estimated ONLY on the formation window of the current fold. Full-sample seasonality may be reported descriptively (Phase 4.5) but cannot feed the trading-window cost model.

Compute per ticker, per 30-min intraday bucket (on formation window):

```
# 13 buckets: 09:30-10:00, 10:00-10:30, ..., 15:30-15:59
for bucket in intraday_buckets:
    spread_seasonality[ticker][bucket] = {
        'mean':   mean(full_spread_l1_bps in bucket),
        'median': median(full_spread_l1_bps in bucket),
        'p95':    quantile(full_spread_l1_bps in bucket, 0.95)
    }
```

**Purpose:** (1) Empirically validates Week 4's 30-bar session warmup. (2) Feeds Kill Zone analysis (Phase 4.5).

### 1.4 Rolling Spread Instability (Seasonality-Adjusted)

Spreads follow a deterministic U-shape intraday. To isolate stochastic risk from predictable seasonality, we compute rolling spread standard deviation on the **seasonality-adjusted spread**:

```
rolling_window = 390 bars (1 trading day)

# De-mean using the bucket medians from Phase 1.3:
adj_spread_bps     = full_spread_l1_bps(t) - spread_seasonality[ticker][bucket_at_t].median
spread_std_1d      = rolling_std(adj_spread_bps, window=390)

# We still track raw mean for reporting:
raw_spread_mean_1d = rolling_mean(full_spread_l1_bps, window=390)
```

`spread_std_1d` is the key input to the slippage model's impact component (Phase 2).

### 1.5 Output

```
data/microstructure/
├── spreads_1min.parquet       # mid_px, spread_{l1,l2,l3}_bps, liquidity_l1
├── spread_rolling.parquet     # spread_mean_1d, spread_std_1d, spread_z
├── spread_seasonality.parquet # per-ticker, per-bucket aggregates
└── spread_summary.parquet     # per-ticker aggregate stats (mean, median, p95, p99)
```

---
