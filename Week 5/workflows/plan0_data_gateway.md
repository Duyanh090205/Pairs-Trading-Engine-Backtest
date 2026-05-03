# Plan 0 — Data Gateway (Phase 1)

> [!NOTE]
> **Status (2026-05-03):** ✅ IMPLEMENTED, AUDITED, FIXED. Synthetic smoke test passes. Real-data ingest pending.

## Objective
Ingest `orderbook.parquet` (213M rows), clean anomalies without breaking time alignment, and extract structural spread features (full spread, half spread, rolling instability) to be consumed by the cost model. Standalone data pipeline with zero dependency on Week 4 outputs.

## Input
- `data/orderbook.parquet` (4.1 GB, ~213M rows)
  - Schema: `timestamp, ticker, l{1,2,3}_bid_px, l{1,2,3}_bid_sz, l{1,2,3}_ask_px, l{1,2,3}_ask_sz`

## Output
```
data/microstructure/
├── spreads_1min.parquet       # per-bar microstructure features
├── spread_rolling.parquet     # per-bar rolling stats (spread_std_1d, raw_spread_mean_1d)
├── spread_seasonality.parquet # per-ticker × 13-bucket aggregates (mean, median, p95)
└── spread_summary.parquet     # per-ticker aggregates (n_obs, mean, median, p95, p99, std)
```

## Reuse from Prior Weeks
| Code | Source | Adaptation |
|---|---|---|
| Timestamp alignment (UTC→ET) | Week 4 Phase 0 (`gateway.py`) | Direct reuse |
| Session filter (09:30–15:59) | Week 4 Phase 0 (`gateway.py`) | Direct reuse |

## Files (all implemented)
- `src/plan0_gateway/__init__.py`
- `src/plan0_gateway/ingest.py`
- `src/plan0_gateway/features.py`
- `src/plan0_gateway/seasonality.py`
- `src/plan0_gateway/rolling.py`
- `src/plan0_gateway/smoke_test.py`

## Implementation Notes (post-audit)

### `ingest.py` — `process_orderbook(df)`
- **Copies input** before mutation (`df = df.copy()`).
- Handles tz-naive (assumes UTC, then converts to US/Eastern) and tz-aware inputs.
- Session filter: `09:30:00 ≤ time ≤ 15:59:59`.
- `is_valid = (l1_ask_px > l1_bid_px) AND (l1_bid_px > 0)`. **Flagged, not dropped** (preserves 1-min timeline alignment with Week 4 trade timestamps).

### `features.py` — `compute_microstructure_features(df)`
- Copies input. Computes:
  - `mid_px = (l1_bid_px + l1_ask_px) / 2`
  - `full_spread_l1_bps`, `half_spread_l1_bps` (half = full / 2 exactly)
  - `full_spread_l2_bps`, `full_spread_l3_bps`
  - `liquidity_l1 = l1_bid_sz × mid_px` (one side, per spec)

### `seasonality.py` — `compute_intraday_seasonality(df)`
- Filters to `is_valid=True` rows only.
- Vectorized bucket assignment via seconds-since-midnight + `pd.cut(right=False)`.
- 13 fixed buckets `09:30-10:00 ... 15:30-15:59`.
- Per `(ticker, bucket)`: mean, median, p95.
- **No-lookahead:** caller (Plan 2 walk-forward) must filter `df` to formation window before calling.

### `rolling.py` — `compute_rolling_instability(df, seasonality_df)`
- Vectorized bucket assignment (same logic as seasonality.py).
- Merges seasonality medians on `[ticker, bucket]`.
- `adj_spread_bps = full_spread_l1_bps − seasonality_median` (only for valid rows with a median).
- `spread_std_1d = rolling(window=390, min_periods=390).std()` per ticker. **390-bar burn-in** enforced.
- `raw_spread_mean_1d`: same window, on `is_valid`-masked spreads. Pre-mask via temp column to avoid closure-over-df fragility.

## Smoke Test (synthetic, passes)
`python src/plan0_gateway/smoke_test.py`

Asserts:
1. Input DataFrame length unchanged after `process_orderbook` (no mutation)
2. Timestamps are US/Eastern
3. No bars outside 09:30–15:59
4. `half_spread_l1_bps == full_spread_l1_bps / 2`
5. Valid rows have non-negative spreads
6. Invalid quotes flagged (not dropped)
7. SPY median spread ≈ 4 bps; VTRS median ≈ 30 bps
8. Exactly 13 seasonality buckets per ticker, no NaN medians
9. **390-bar burn-in:** `spread_std_1d` is NaN for first 389 bars; first valid value at position ≥ 389
10. All four parquet files written to `data/microstructure/`

## Real-Data Ingest (separate from smoke test)

```python
# Pseudocode for the real-data run
raw = pd.read_parquet("data/orderbook.parquet")    # consider chunking if memory-bound
clean = process_orderbook(raw)
features = compute_microstructure_features(clean)

# For seasonality: use only formation window of fold 1 (Plan 2 will redo per-fold)
formation_mask = (features["timestamp_et"] >= FOLD1_FORMATION_START) & \
                 (features["timestamp_et"] <= FOLD1_FORMATION_END)
seasonality = compute_intraday_seasonality(features[formation_mask])

rolling = compute_rolling_instability(features, seasonality)

# Write all four parquets to data/microstructure/
```

> [!IMPORTANT]
> The smoke-test seasonality uses the entire synthetic sample as a stand-in for "formation window."
> On real data, seasonality must be recomputed per fold inside Plan 2's walk-forward loop.
> The four parquets written here are baseline artifacts; Plan 2 will regenerate seasonality per fold.

---

## ⛔ HARD STOP — Review Before Plan 2

**Synthetic smoke test:**
- [x] All 4 parquet files written
- [x] Timestamps US/Eastern
- [x] No bars outside session
- [x] Quote quality flags present (flagged, not dropped)
- [x] Spread values non-negative on valid rows
- [x] 390-bar burn-in enforced

**Real-data ingest (pending):**
- [ ] `data/microstructure/*.parquet` produced from `data/orderbook.parquet`
- [ ] File sizes consistent with expected row counts (~213M for spreads_1min)
- [ ] Per-ticker spread distributions match expected ranges (SPY ~4.7, VTRS ~32)

**Audit fixes applied:**
- `df.copy()` added to ingest.py and features.py (no input mutation)
- `min_periods=300 → 390` in rolling.py (burn-in correctness)
- `apply()` → vectorized in seasonality.py and rolling.py (performance on 213M rows)
- `liquidity_l1` corrected to `l1_bid_sz × mid_px` (one side, per spec)
- `_spread_valid` temp column for raw_spread_mean_1d (closure-fragility fix)
- All 4 parquet files written by smoke_test.py
- Absolute paths via `WEEK5_ROOT`
