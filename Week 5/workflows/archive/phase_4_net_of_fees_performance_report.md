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
