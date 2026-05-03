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
