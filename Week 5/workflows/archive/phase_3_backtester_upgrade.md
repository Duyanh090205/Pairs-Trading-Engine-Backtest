## Phase 3 — Backtester Upgrade

**Goal:** Hook the dynamic cost model into the Week 4 execution engine via the integration interface.
**Constraint:** NO changes to signal logic, position sizing, or entry/exit rules. Only the cost computation changes.

### 3.1 Four Hook Points (Strict Dollar Accounting)

**Hook 1 — Entry Cost:**
```
# Week 4: entry_cost_$ = 0.0030 × (notional_A + notional_B)
# Week 5: 
entry_cost_A_$ = (C_spread_A(t) + C_impact_A(t)) / 10_000 × notional_A
entry_cost_B_$ = (C_spread_B(t) + C_impact_B(t)) / 10_000 × notional_B
entry_cost_$   = entry_cost_A_$ + entry_cost_B_$
```

**Hook 2 — Exit Cost:**
```
exit_cost_A_$ = (C_spread_A(t) + C_impact_A(t)) / 10_000 × notional_A
exit_cost_B_$ = (C_spread_B(t) + C_impact_B(t)) / 10_000 × notional_B
exit_cost_$   = exit_cost_A_$ + exit_cost_B_$
```

**Hook 3 — Borrow Cost:** Daily accrual in dollars, unchanged from Week 4.

**Hook 4 — Threshold Rebalance Cost:**
Computed per row in the `rebalance_log`:
```
rebalance_cost_$ = (C_spread(t) + C_impact(t)) / 10_000 × notional_rebalanced
```

### 3.2 Numba Compatibility

All slippage inputs are pre-computed as numpy arrays aligned to the 1-min execution timeline. The Numba `@njit` engine receives them as additional input arrays — no Python objects inside the hot loop:

```
@njit(cache=True, fastmath=False)
def execute_with_slippage(
    prices_A, prices_B,                 # existing
    z_scores, positions,                # existing
    half_spread_l1_bps_A, half_spread_l1_bps_B, # NEW: strictly half-spread
    sigma_A, sigma_B,                   # NEW: seasonality-adjusted spread_std_1d
    kappa_A, kappa_B,                   # NEW: impact coefficient per ticker
    borrow_rate_daily,                  # existing
    ...
):
```

### 3.3 Walk-Forward Integration

Slippage model parameters re-calibrated **each fold** on the formation window:
- κ tier assignment: based on formation-window median spread
- Spread rolling stats: burn-in = 390 bars at start of trading window

Spread data is real-time observable — `spread_l1_bps(t)` and `spread_std_1d(t)` are known at time t. No forward-looking information.

---
