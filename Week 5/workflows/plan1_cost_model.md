# Plan 1 — Cost Model & Interface (Phase 2 + 2.5)

> [!NOTE]
> **Status (2026-05-03):** ✅ IMPLEMENTED, AUDITED, FIXED. Smoke test passes with kappa-tier, sensitivity-grid, and three-regime checks.

## Objective
Build a dynamic, empirically-calibrated friction function that replaces Week 4's static 60 bps assumption. Define the strict integration interface with Week 4. Provide all primitives needed by Plan 2.

## Input
- `data/microstructure/spreads_1min.parquet` (consumed by Plan 2)
- `data/microstructure/spread_rolling.parquet` (consumed by Plan 2)
- Week 4 `trade_log` schema (defined in `interface_contract.py`)

## Output
- `src/plan1_cost_model/` Python module (no data produced — Plan 2 produces `cost_log.parquet`)

## Reuse from Prior Weeks
| Code | Source | Adaptation |
|---|---|---|
| Borrow daily accrual | Week 4 Phase 3 (`pnl.py`) | Standalone function with same `(rate/10000)/252` math |
| Static TC formula | Week 4 Phase 3 (`pnl.py`) | Reproduced as `calculate_static_round_trip_cost()` |

## Files (all implemented)
- `src/plan1_cost_model/__init__.py` — exports all functions
- `src/plan1_cost_model/spread_cost.py` — `calculate_spread_cost()`
- `src/plan1_cost_model/impact_cost.py` — `assign_kappa_tier()`, `calculate_impact_cost()`
- `src/plan1_cost_model/borrow_cost.py` — `calculate_borrow_cost()`
- `src/plan1_cost_model/round_trip.py` — `calculate_round_trip_cost()`, `calculate_static_round_trip_cost()`
- `src/plan1_cost_model/interface_contract.py` — schemas + validators
- `src/plan1_cost_model/smoke_test.py`

## Cost Model

### Three-component dynamic cost (per leg, per trade event)
```
C_total(t) = C_spread(t) + C_impact(t) + C_borrow(t)
```

**`calculate_spread_cost(half_spread_l1_bps, notional_$)`**
```
C_spread_$ = notional_$ × half_spread_l1_bps / 10_000
```

**`calculate_impact_cost(kappa, spread_std_1d, notional_$, multiplier=1.0)`**
```
if spread_std_1d is NaN:    # burn-in or data gap
    impact_bps = 15.0 × multiplier        # conservative fallback
else:
    impact_bps = kappa × spread_std_1d × multiplier
C_impact_$ = notional_$ × impact_bps / 10_000
```

**`assign_kappa_tier(median_full_spread_bps)` — pre-specified, NOT in-sample fitted**
```
< 8 bps   → κ = 0.3   (Tier 1, tight)
8–20 bps  → κ = 0.5   (Tier 2, medium; boundaries inclusive)
> 20 bps  → κ = 0.8   (Tier 3, wide)
```
Re-evaluated per fold using formation-window median.

**`calculate_borrow_cost(short_notional, entry_ts, exit_ts, rate=50)`**
```
if entry_ts.date() == exit_ts.date():
    return 0.0
holding_days = (exit_ts.date() − entry_ts.date()).days   # calendar days
daily = (rate_bps_annual / 10_000) / 252 × short_notional
return daily × holding_days
```

**`calculate_round_trip_cost(entry_A, entry_B, exit_A, exit_B, borrow)`** — pure aggregation.

### Static 60 bps baseline (Week 4 reproduction)
**`calculate_static_round_trip_cost(notional_A_entry, notional_B_entry, notional_A_exit, notional_B_exit, tc_bps_per_leg=30.0)`**
```
entry_cost = (notional_A_entry + notional_B_entry) × tc_bps / 10_000
exit_cost  = (notional_A_exit  + notional_B_exit ) × tc_bps / 10_000
total      = entry_cost + exit_cost
```
Matches Week 4 `pnl.py`: `tc_rate × (notional_A + notional_B)` at entry and again at exit. Default 30 bps/leg = 60 bps per side = 120 bps round-trip notional-weighted.

## Three Cost Regimes (always reported side-by-side)

| Regime | Cost formula | Notes |
|---|---|---|
| **Gross** | `cost = 0` | Upper bound on alpha |
| **Static 60 bps** | `calculate_static_round_trip_cost()` | Week 4 baseline anchor |
| **Dynamic** | `calculate_round_trip_cost(spread + impact, ..., borrow)` per trade | Empirical, regime-dependent |

## Interface Contract (`interface_contract.py`)

**Required from Week 4:**
```python
TRADE_LOG_SCHEMA = {
    'trade_id': 'string', 'fold_id': 'int', 'pair_id': 'string',
    'ticker_A': 'string', 'ticker_B': 'string',
    'side_A': 'int', 'side_B': 'int',
    'entry_ts': 'datetime64[ns, US/Eastern]',
    'exit_ts':  'datetime64[ns, US/Eastern]',
    'notional_A_entry': 'float', 'notional_B_entry': 'float',
    'notional_A_exit':  'float', 'notional_B_exit':  'float',
    'gross_pnl_dollars': 'float', 'allocated_capital': 'float',
}

REBALANCE_LOG_SCHEMA = {
    'trade_id': 'string', 'fold_id': 'int', 'pair_id': 'string',
    'ticker': 'string', 'rebalance_ts': 'datetime64[ns, US/Eastern]',
    'delta_shares': 'float', 'price_at_rebalance': 'float',
    'notional_rebalanced': 'float',
}
```

**Output by Week 5 (Plan 2 produces this):**
```python
COST_LOG_SCHEMA = {
    'trade_id': 'string',
    'spread_cost_dollars':    'float',
    'impact_cost_dollars':    'float',
    'borrow_cost_dollars':    'float',
    'rebalance_cost_dollars': 'float',
    'total_cost_dollars':     'float',
    'net_pnl_dollars':        'float',
    'net_return':             'float',
}
```

Validators check column existence. **Type/timezone validation is NOT enforced** — Plan 2 must verify `entry_ts.dtype == 'datetime64[ns, US/Eastern]'` before running cost lookups, otherwise spread joins will produce silent NaN columns.

## Smoke Test (passes)
`python src/plan1_cost_model/smoke_test.py`

Asserts:
1. **Kappa tier boundaries:** `assign_kappa_tier(7.99)=0.3`, `(8.0)=0.5`, `(20.0)=0.5`, `(20.01)=0.8`
2. **Sensitivity grid monotonicity:** impact cost at 0.5× < 1.0× < 1.5×
3. **Three regimes compute and differ:** Gross=$0, Static>0, Dynamic>0, Static≠Dynamic
4. **Static formula sanity:** 4 legs × 30 bps × $100k = $1200
5. **100-trade cost log:** all `total_cost_$ ≥ 0`, dollar-denominated
6. **Borrow logic:** 0 intraday, >0 overnight
7. **Schema validation:** `validate_cost_log()` passes on conforming DataFrame

---

## ⛔ HARD STOP — Review Before Plan 2

- [x] All cost functions return dollar-denominated values
- [x] κ assigned per formation-window median, not in-sample fitted
- [x] Borrow cost = 0 for intraday; >0 for overnight
- [x] `interface_contract.py` validates all 3 schemas (column existence)
- [x] Synthetic test: 100 trades produce valid `cost_log`
- [x] Three regimes (Gross/Static/Dynamic) compute without error
- [x] `total_cost_dollars ≥ 0` for every test trade
- [x] Sensitivity grid monotonic
- [x] `calculate_static_round_trip_cost()` exists and matches Week 4 `pnl.py`

**Audit fixes applied:**
- Added `calculate_static_round_trip_cost()` (was missing — broke three-regime spec)
- Smoke test exercises `assign_kappa_tier()` (was hardcoded)
- Smoke test exercises sensitivity multiplier (was unused)
- Smoke test computes three regimes (was dynamic-only)
