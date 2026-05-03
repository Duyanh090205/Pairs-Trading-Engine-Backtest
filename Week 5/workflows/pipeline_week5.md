# Pipeline Week 5: The Engine — The "Microstructure" Reality

> [!NOTE]
> **Status as of 2026-05-03**
>
> | Phase | Plan | Status |
> |---|---|---|
> | 1 | Plan 0 — Data Gateway | ✅ Implemented, audited, smoke test passes (synthetic). Real-data ingest pending. |
> | 2 + 2.5 | Plan 1 — Cost Model | ✅ Implemented, audited, smoke test passes. `calculate_static_round_trip_cost()` added during audit. |
> | 3 | Plan 2 — Cost Application | ⏳ Renamed and rescoped: per-trade Python loop, no Numba engine. Blocked on Week 4 outputs. |
> | 4–6 | Plan 3 — Validation & Report | ⏳ Only `red_flags.py` implemented. Other 9 modules blocked on Plan 2 output. |

## 0. Objective

**Theme:** The Friction.

**Deliverable:** A "Net-of-Fees Performance Report."

**Core Question:** The Week 4 strategy showed profitability under a flat 60 bps cost assumption. When we replace that with empirical bid-ask spreads that widen during spread instability, does alpha survive — or was it an illusion eaten by friction?

**Scope Boundary:** This week **measures** friction on the existing strategy. We do NOT modify the strategy itself (no sizing changes, no new entry/exit gates, no regime suppression). Same signals, same positions — different cost model.

**Data:** `orderbook.parquet` — 213M rows, 504–526 tickers, 1-min frequency, 2022-01-03 to 2026-03-19. 3-level limit order book (L1/L2/L3 bid/ask price + size).

---

## 0.1 Data Profile

| Property | Value |
|---|---|
| Total rows | 212,975,144 |
| Frequency | 1-minute (dominant; some 2–5 min gaps) |
| Tickers | 504 (2022-01) → 526 (2026-03) |
| Columns | `timestamp`, `ticker`, `l{1,2,3}_bid_px`, `l{1,2,3}_bid_sz`, `l{1,2,3}_ask_px`, `l{1,2,3}_ask_sz` |
| NaN count | 0 across all columns |
| Date range | 2022-01-03 09:00 → 2026-03-19 23:59 |

**L1 Spread (bps):** Universe median ~10; SPY ~4.7; VTRS ~32; max observed >150 (stress prints).

**Depth:** L2 spread ~20 bps (1.5× L1 size), L3 ~43 bps (2.5× L1 size).

**Known Limitation:** `bid_sz == ask_sz` at all levels (symmetric/synthetic LOB). No order-flow imbalance signals can be derived. Size data is treated as representative liquidity at each price level only.

---

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
├── spreads_1min.parquet       # mid_px, spread_{l1,l2,l3}_bps, half_spread_l1_bps, liquidity_l1, is_valid + rolling cols
├── spread_rolling.parquet     # timestamp_et, ticker, spread_std_1d, raw_spread_mean_1d
├── spread_seasonality.parquet # per-ticker, per-bucket: mean, median, p95
└── spread_summary.parquet     # per-ticker: n_obs, mean_bps, median_bps, p95_bps, p99_bps, std_bps
```

---

## Phase 2 — The Slippage Model

**Goal:** Replace Week 4's static 60 bps with a dynamic, empirically-calibrated friction function.
**Design Principle:** Cost increases when spread instability is high.

### 2.1 Three-Component Cost Model

Total execution cost per leg, per trade:

$$C_{total}(t) = C_{spread}(t) + C_{impact}(t) + C_{borrow}(t)$$

**Component 1 — Spread Cost (empirical, variable):**

```
C_spread(t) = half_spread_l1_bps(t)
```

This varies per ticker, per bar, per market condition. NOT a fixed number.

**Component 2 — Market Impact (spread-instability-scaled):**

```
C_impact(t) = κ × spread_std_1d(t)

where:
  spread_std_1d(t) = rolling 1-day std of seasonality-adjusted L1 spread (from Phase 1.4)
  κ               = impact coefficient assigned per liquidity tier (see 2.2)
```

**Intuition:** When `spread_std_1d` is high → quotes are unstable → fill slips beyond quoted spread. When low → stable liquidity → minimal extra impact.

**Component 3 — Borrow Cost (unchanged from Week 4 §3.2):**

```
borrow_rate_bps_annual = 50    # default; sensitivity {30, 50, 100}
borrow_cost_daily = (borrow_rate_bps_annual / 10,000) / 252 × short_notional_$
```

Borrow cost applies only to overnight short exposure in the primary model. Intraday same-day shorts (e.g. EOS-flattened trades) have zero borrow cost. Conservative sensitivity sweeps may prorate intraday borrow as `holding_minutes / 390`.

### 2.2 Impact Coefficient Assignment (κ)

κ is assigned per liquidity tier as a **fixed default**:

```
# Tier by median full L1 spread on formation window:
Tier 1 (Tight, <8 bps median):   κ = 0.3    # e.g., SPY, AAPL, MSFT
Tier 2 (Medium, 8–20 bps):       κ = 0.5    # e.g., most S&P 500
Tier 3 (Wide, >20 bps):          κ = 0.8    # e.g., VTRS, T, UBER
```

**Validation (report alongside κ, do not use for selection):**

```
# Per ticker, on formation window — sanity-check that tier assignment is reasonable:
empirical_impact_proxy = median(full_spread_l2_bps - full_spread_l1_bps) / spread_std_1d
# Report: histogram of proxy by tier. If proxy distribution overlaps heavily across tiers → tiers are not discriminating.
```

L2−L1 spread difference measures how much price worsens when L1 liquidity is consumed. The proxy validates that tiers capture real depth differences, but κ values themselves are fixed to avoid in-sample fitting.

**Sensitivity grid for κ:** {0.5×, 1.0×, 1.5×} of calibrated values.

### 2.3 Total Round-Trip Cost Formula

For a pair trade entry (buy A, sell B) and exit (sell A, buy B), costs must be resolved strictly to dollars to avoid denominator mismatch:

```
# Entry (Dollars)
cost_entry_A_$ = (C_spread_A(t_entry) + C_impact_A(t_entry)) / 10_000 × notional_A
cost_entry_B_$ = (C_spread_B(t_entry) + C_impact_B(t_entry)) / 10_000 × notional_B

# Exit (Dollars)
cost_exit_A_$  = (C_spread_A(t_exit) + C_impact_A(t_exit)) / 10_000 × notional_A
cost_exit_B_$  = (C_spread_B(t_exit) + C_impact_B(t_exit)) / 10_000 × notional_B

# Total Trade Cost (Dollars)
total_trade_cost_$ = cost_entry_A_$ + cost_entry_B_$ + cost_exit_A_$ + cost_exit_B_$

# Borrow (accrued daily while position open)
total_borrow_$ = Σ borrow_cost_daily(d)  for d in holding_days
```

### 2.4 Three Cost Regimes (Always Report Side-by-Side)

Every metric in the report is computed under all three:

| Regime | Description |
|---|---|
| **Gross** | Zero transaction costs (upper bound on alpha) |
| **Static 60 bps** | Week 4 baseline: strictly defined as **30 bps per leg on entry + 30 bps per leg on exit** applied to actual leg notional. |
| **Dynamic** | This model: empirical spread + instability-scaled impact + borrow |

No cherry-picking. All three always shown together.

---

## Phase 2.5 — Integration Interface Contract

Week 5 is strictly a Layer 2 evaluator. It must not depend on Week 4 internals, but rather consume a strict interface generated by Week 4.

**Required `trade_log` output from Week 4:**
```
Columns: trade_id, fold_id, pair_id, ticker_A, ticker_B, side_A, side_B, 
entry_ts, exit_ts, notional_A_entry, notional_B_entry, 
notional_A_exit, notional_B_exit, gross_pnl_dollars, allocated_capital
```

**Required `rebalance_log` output from Week 4:**
```
Columns: trade_id, fold_id, pair_id, ticker, rebalance_ts, 
delta_shares, price_at_rebalance, notional_rebalanced
```

**Expected `cost_log` output from Week 5:**
```
Columns: trade_id, spread_cost_dollars, impact_cost_dollars, 
borrow_cost_dollars, rebalance_cost_dollars, total_cost_dollars, 
net_pnl_dollars, net_return
```

---

## Phase 3 — Cost Application

> [!IMPORTANT]
> Renamed from "Backtester Upgrade." Week 5 does NOT re-execute the strategy. Plan 2 is a **per-trade post-processing pass** over Week 4's existing `trade_log` and `rebalance_log` outputs. No Numba engine.

**Goal:** Apply the dynamic cost model to Week 4's existing trades; produce `cost_log.parquet` with three regime columns.
**Constraint:** Cost is computed by lookup at `entry_ts`, `exit_ts`, and each `rebalance_ts` — never by re-running the strategy.

### 3.1 Three Cost Computations Per Trade (Strict Dollar Accounting)

**Dynamic regime — per trade, two timestamps × two legs:**
```
At entry_ts:
  entry_cost_A_$ = (C_spread_A(t_entry) + C_impact_A(t_entry)) / 10_000 × notional_A_entry
  entry_cost_B_$ = (C_spread_B(t_entry) + C_impact_B(t_entry)) / 10_000 × notional_B_entry

At exit_ts:
  exit_cost_A_$ = (C_spread_A(t_exit) + C_impact_A(t_exit)) / 10_000 × notional_A_exit
  exit_cost_B_$ = (C_spread_B(t_exit) + C_impact_B(t_exit)) / 10_000 × notional_B_exit

Per holding day (overnight only):
  borrow_cost_$ = (rate / 10_000) / 252 × short_notional_$

Per rebalance event (one-sided, on rebalance_log row):
  rebalance_cost_$ = (C_spread(t) + C_impact(t)) / 10_000 × notional_rebalanced
```

**Static 60 bps regime** — `calculate_static_round_trip_cost()` from Plan 1:
```
total = (notional_A_entry + notional_B_entry + notional_A_exit + notional_B_exit) × 30bps / 10_000
```

**Gross regime** — `cost = 0` literally.

All three are written as separate columns in `cost_log.parquet`, computed in a single pass.

### 3.2 Walk-Forward κ Calibration

κ re-tiered **per fold** on formation-window median spread (`assign_kappa_tier()` from Plan 1). Stored per fold in `data/kappa_per_fold.parquet` for audit.

Spread data is real-time observable — `half_spread_l1_bps(t)` and `spread_std_1d(t)` are known at time t. The `min_periods=390` burn-in in `compute_rolling_instability()` enforces non-availability before a full trading day's data accumulates; during burn-in, `calculate_impact_cost()` falls back to a conservative 15 bps flat.

`calibrate_fold()` raises `ValueError` if `formation_end >= trading_start` (lookahead guard).

---

## Phase 4 — Net-of-Fees Performance Report

### 4.1 Sharpe Ratio Recalculation (Central Deliverable)

For each of the three cost regimes, calculate daily returns using strictly matched denominators:

```
daily_cost_return(d) = daily_cost_dollars(d) / portfolio_equity_previous_day
daily_returns_net(d) = daily_returns_gross(d) - daily_cost_return(d)
Sharpe_net = mean(daily_returns_net) / std(daily_returns_net) × √252
```

**The "Before vs After" Table:**

| Metric | Gross | Static 60 bps | Dynamic Model |
|---|---|---|---|
| Sharpe (annualized) | | | |
| CAGR | | | |
| MaxDD (bar-level) | | | |
| Calmar | | | |
| Win Rate | | | |
| Avg Trade Count / Fold | | | |
| Avg RT Cost (bps) | | | |
| % Folds Profitable | | | |

### 4.2 Cost Decomposition Waterfall

Break down how gross alpha is consumed:

```
Gross Alpha (bps/trade)
  − Spread Cost (half-spread × 2 legs × 2 sides)
  − Market Impact (κ × σ_spread × 2 legs × 2 sides)
  − Borrow Cost (daily accrual on short leg)
  − Rebalance Cost (threshold re-hedge events)
  = Net Alpha (bps/trade)
```

Visualize as waterfall chart. Partition by regime (Bear 2022 / Bull 2023–2026).

### 4.3 Regime-Conditional Cost Analysis

Merge with Week 4 Phase 4.1 regime partitions:

| Regime | Avg L1 Spread (bps) | Avg Dynamic RT Cost (bps) | Sharpe Gross | Sharpe Net | Δ Sharpe |
|---|---|---|---|---|---|
| Late Bear 2022 | | | | | |
| Early Bull 2023 | | | | | |
| Mid Bull 2024 | | | | | |
| Late Bull 2025–Q1 2026 | | | | | |

**Hypothesis:** Bear 2022 spreads wider → dynamic costs higher → net Sharpe degrades more than static assumption. Bull spreads tighter → dynamic model may be more favorable than 60 bps.

### 4.4 Impact Prediction Validation

Demonstrate that spread instability correctly predicts depth-decay execution friction:

```
# Per ticker per day:
future_cost_proxy = full_spread_l2_bps - full_spread_l1_bps

# Cross-sectional regression against our model's impact term:
future_cost_proxy = α + β × spread_std_1d + ε

# Report: β coefficient, R², scatter plot
```

If β is positive, this supports the macro thesis that spread instability translates to actual depth leakage. If β is weak, the dynamic model can still be valid if `spread_std_1d` predicts realized execution friction. Therefore this test is supporting evidence, not a hard pass/fail condition.

### 4.5 Kill Zone & Intraday Spread Seasonality

Two outputs from one computation. Group all bars into 13 × 30-min intraday buckets:

```
for bucket in [09:30-10:00, 10:00-10:30, ..., 15:30-15:59]:
    # Part A — Spread seasonality (U-shape validation)
    for tier in [Tier1_Tight, Tier2_Medium, Tier3_Wide]:
        avg_spread[tier][bucket] = mean(spread_l1_bps in tier & bucket)

    # Part B — Kill zone (net alpha by time-of-day)
    avg_full_spread_bps  = mean(full_spread_l1_bps in bucket)
    avg_gross_alpha_dollars = mean(gross_pnl_dollars per trade entered in bucket)
    avg_net_alpha_dollars   = avg_gross_alpha_dollars - avg_cost_dollars_in_bucket
    kill_zone               = True if avg_net_alpha_dollars < 0
```

**Output A:** Line chart — x = time-of-day, y = avg spread, one line per tier. Validates U-shape (wide at open, tight midday, wide near close). If absent → flag data anomaly.

**Output B:** Heatmap — net alpha by time-of-day × regime. Identifies buckets where friction exceeds alpha.

### 4.6 Negative Control under Dynamic Costs

Run Week 4 §3.5 NC pairs through all three cost regimes:

```
for nc_pair in [CVNA_ISRG, synthetic_RW]:
    for regime in [Gross, Static_60bps, Dynamic]:
        nc_sharpe[nc_pair][regime] = run_backtest(nc_pair, cost_model=regime)

# Pass criteria:
# 1. NC Sharpe under Dynamic ≈ 0 or negative (same as Week 4)
# 2. Dynamic cost does NOT accidentally make NC profitable
# 3. Primary Sharpe (net dynamic) > NC Sharpe (net dynamic) + 2σ bootstrap threshold
```

**Why:** A cost model with bugs can create or destroy apparent alpha. NC through the same model proves it's directionally neutral.

### 4.7 Overfitting Diagnostics on Net Returns

Recompute Week 4 §4.4 DSR and PBO on net-of-dynamic-cost returns:

```
DSR_net = deflated_sharpe(daily_returns_net, n_trials, variance_of_sharpe_estimator)
PBO_net = combinatorial_purged_CV(daily_returns_net, k_folds)
```

Report side-by-side:

| Metric | Gross | Static 60bps | Dynamic |
|---|---|---|---|
| Raw Sharpe | | | |
| DSR (p-value) | | | |
| PBO | | | |

**Key insight:** If DSR_net p-value degrades significantly vs DSR_gross → statistical significance is fragile to friction, even if raw net Sharpe is positive.

### 4.8 OAT Sensitivity (Slippage Parameters)

Extend Week 4 §4.5 grid. All sweeps use the **Dynamic cost model** (Gross/Static are baseline comparisons from §4.1, not parameters to sweep):

| Parameter | Values Swept |
|---|---|
| κ multiplier | {0.5×, **1.0×**, 1.5×} |
| Borrow rate | {30, **50**, 100 bps/year} |
| Spread level used | {**L1**, L2} |

→ 3 OAT dimensions × ~3 values = ~9 additional runs × 45 folds.

---

## Phase 5 — Red Flag Triggers

Inherit all Week 4 §3.7 red flags. Add:

| Trigger | Condition | Action |
|---|---|---|
| Dynamic cost blow-up | Mean RT cost > 150 bps any fold | Audit spread data alignment, check for stale quotes |
| Cost > Gross Alpha | Net Sharpe < 0 AND Gross Sharpe > 1.0 | Flag: alpha real but untradeable at this frequency |
| Spread data gap | > 5% missing spread observations in trading window | Fall back to static 60 bps for affected bars |
| κ instability | κ tier changes > 2× across consecutive folds for same ticker | Flag: liquidity regime shift |
| NC cost model leak | NC Sharpe under dynamic model > 0.5 | Audit cost model for sign errors or look-ahead |
| DSR degradation | DSR_net p-value > 0.10 while DSR_gross p-value < 0.05 | Flag: alpha not statistically significant after costs |
| Math violation | `total_cost_dollars < 0` OR `net_pnl_dollars > gross_pnl_dollars` | Throw Error: Cost model must strictly subtract PnL |

---

## Phase 6 — Report Structure

The "Net-of-Fees Performance Report" deliverable:

| § | Content | Source Phase |
|---|---|---|
| 1 | **Executive Summary:** Does the strategy survive fees? One-line verdict. | — |
| 2 | **Empirical Spread Profile:** Median, p95, p99 spreads by tier and regime. Intraday U-shape. | 1, 4.5A |
| 3 | **Slippage Model Spec:** Three-component model, κ calibration + proxy validation. | 2 |
| 4 | **The Before/After Table:** Sharpe Gross vs Net, all three regimes. | 4.1 |
| 5 | **Cost Waterfall:** Where does gross alpha go? | 4.2 |
| 6 | **Regime-Conditional Costs:** Bear vs Bull spread behavior and net Sharpe. | 4.3 |
| 7 | **Spread-Vol Correlation:** Empirical proof spreads widen with vol. | 4.4 |
| 8 | **Kill Zone + Seasonality:** Time-of-day heatmap of net alpha + U-shape validation. | 4.5 |
| 9 | **Negative Control (Dynamic):** NC results under all three cost regimes. | 4.6 |
| 10 | **Overfitting Diagnostics (Net):** DSR/PBO on net returns. | 4.7 |
| 11 | **Sensitivity Analysis:** OAT for κ, borrow, spread level. | 4.8 |
| 12 | **Verdict:** Honest conclusion + what would need to change if net Sharpe < 0. | — |

---

## Critical Lock-Ins

21. **Dynamic slippage replaces static 60 bps** as primary cost model; static 60 bps retained as comparison anchor
22. **Three-component cost:** half-spread + volatility-scaled impact + borrow
23. **κ is pre-specified by liquidity tier** using formation-window median full L1 spread. L2−L1 widening is used only as validation, not for performance fitting.
24. **Spread data is real-time observable** — no forward-looking spread statistics in execution
25. **Symmetric LOB sizes acknowledged as limitation** — no order-flow imbalance signals derived
26. **All three cost regimes reported side-by-side** — no cherry-picking
27. **Net Sharpe is the decision metric** — gross Sharpe is informational only
28. **Strategy unchanged** — Week 5 measures friction only, does not modify signals/sizing/gates
29. **NC validation under dynamic costs** — proves cost model is directionally neutral
30. **DSR/PBO recomputed on net returns** — statistical significance must survive friction

---

## Extensions (Run If Time Permits)

These are valuable but not required for the core deliverable. Each is self-contained and does not depend on the others.

### EXT-1 — Execution Latency Stress Test with Real Spread Cost

Extends Week 4 §3.6. The latency sweep {t+1, t+2, t+5, t+10} now charges the **actual spread at the delayed execution bar**, not the spread at the signal bar:

```
for lag in [1, 2, 5, 10]:
    t_exec = t_signal + lag
    fill_price    = price(t_exec)                          # already in Week 4
    fill_cost_bps = C_spread(t_exec) + C_impact(t_exec)    # NEW: cost at execution bar

# Output: alpha decay curve with two y-axes:
#   y1 = Sharpe (net of dynamic cost at execution bar)
#   y2 = Avg cost at execution bar vs cost at signal bar
```

**Key question:** Does latency increase cost as well as decrease alpha? If spreads autocorrelate (wide persists), latency compounds both alpha loss and cost increase.

**Pass criterion:** Net Sharpe at t+5 > 0 AND cost at t+5 < 1.5× cost at t+1.

### EXT-2 — Cross-Leg Spread Correlation

For each surviving pair (A, B), compute:

```
corr_spread_AB = corr(spread_l1_bps_A, spread_l1_bps_B)    # per fold, on trading window
```

**Report:**
- Distribution of `corr_spread_AB` across surviving pairs
- Mean correlation per regime partition
- Conditional cost: `avg RT cost when corr > 0.7` vs `avg RT cost when corr < 0.3`

**Implication:** High cross-leg correlation means both legs widen simultaneously → RT cost compounds beyond the sum of individual leg costs. This is a portfolio-level microstructure risk invisible in per-leg analysis.

### EXT-3 — Book-Walk Model (Position Size > L1 Depth)

When the trade's share count exceeds L1 available size, the fill walks into deeper levels:

```
order_shares = position_notional / mid_px

if order_shares <= l1_bid_sz:
    fill_px = l1_ask_px
    effective_spread_bps = spread_l1_bps

elif order_shares <= l1_bid_sz + l2_bid_sz:
    filled_l1 = l1_bid_sz
    filled_l2 = order_shares - l1_bid_sz
    fill_px   = (filled_l1 × l1_ask_px + filled_l2 × l2_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px × 10,000 × 2

else:    # walks to L3
    filled_l1 = l1_bid_sz
    filled_l2 = l2_bid_sz
    filled_l3 = order_shares - l1_bid_sz - l2_bid_sz
    fill_px   = (filled_l1 × l1_ask_px + filled_l2 × l2_ask_px + filled_l3 × l3_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px × 10,000 × 2
```

**Integration:** When book-walk triggers, `C_spread(t)` in §2.1 is replaced by `0.5 × effective_spread_bps`. The impact component still applies on top.

**Diagnostic:** Track `spread_leakage = effective_spread - quoted_spread`. If consistently large → position sizing exceeds available depth.

**Note:** Book-walk results are diagnostic only because LOB sizes are symmetric/synthetic. They should not be used as the primary cost model.

---

## Deferred to Week 6+

These items **change the strategy** (not just measure cost) and are out of scope for Week 5:

- **Liquidity-gated position sizing** — caps shares at fraction of L1 depth
- **Cost-aware rebalance gate** — defers rebalance when spreads are wide
- **HMM regime-detection gateway** — suppresses entries during stressed regime
- **Order-flow toxicity (VPIN/Kyle's Lambda)** — requires asymmetric LOB data
- **Optimal execution (TWAP/VWAP)** — beyond current scope
- **Adaptive κ via online learning** — replace tier-based with Kalman-estimated impact
- **Per-pair Kalman δ optimization** — bucket-level δ as middle ground
- **Cross-tertile pairs analysis** — T1-T2, T1-T3, T2-T3 volume groups
