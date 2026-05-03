# Plan 2 — Cost Application (Phase 3)

> [!IMPORTANT]
> **Renamed from "Backtester Integration".** Week 5 does NOT re-execute the strategy. Plan 2 is a post-hoc Python loop that prices Week 4's existing trades under three cost regimes. No Numba engine, no state machine.

> [!WARNING]
> **Blocked by Week 4:** Requires `trade_log.parquet`, `rebalance_log.parquet`, and per-fold equity parquets from Week 4.

## Objective
Apply the dynamic cost model to Week 4's executed trades and produce `cost_log.parquet` with three regime columns (Gross / Static 60 bps / Dynamic).

## Architectural Decision
Plan 2 is **per-trade post-processing**, not bar-by-bar execution.
- Iterate over `trade_log.parquet` rows.
- Look up spread / σ / κ at `entry_ts` and `exit_ts`.
- Compose dollar costs via Plan 1 functions.
- Aggregate rebalance costs by `trade_id`.
- ~thousands of trades × 4 lookups = milliseconds in Python; Numba is unnecessary.

## Input
- Week 4: `trade_log.parquet`, `rebalance_log.parquet`
- Week 4: `src/phase4_defense/orchestrator.py:FOLD_SCHEDULE` (45 folds)
- Plan 0: `data/microstructure/spreads_1min.parquet`, `spread_rolling.parquet`
- Plan 1: cost functions + schema validators

## Output
- `data/cost_log.parquet` — per-trade costs under three regimes
- `data/kappa_per_fold.parquet` — audit trail of κ assignments per fold per ticker

## Reuse from Prior Weeks
| Code | Source | Adaptation |
|---|---|---|
| `FOLD_SCHEDULE` (45 folds) | Week 4 `phase4_defense/orchestrator.py` | Direct import or copy of constant |
| Borrow daily formula | Already extracted into Plan 1 | — |

## Files to Create
- `src/plan2_backtester/__init__.py`
- `src/plan2_backtester/walk_forward.py` — per-fold κ calibration
- `src/plan2_backtester/hooks.py` — per-trade cost computation
- `src/plan2_backtester/orchestrator.py` — main entry point

> [!NOTE]
> The previous scaffold included `numba_engine.py`. **Delete that file.** It was the wrong abstraction (Week 4 already has a Numba engine; Week 5 doesn't need one).

## Implementation

### Step 1 — `walk_forward.py`

```python
import pandas as pd
from src.plan1_cost_model.impact_cost import assign_kappa_tier

# Pull from Week 4 — copy the 45-fold tuple list
FOLD_SCHEDULE = [
    (1,  pd.Timestamp("2022-01-03"), pd.Timestamp("2022-06-30"), pd.Timestamp("2022-07-01")),
    # ... 45 entries total ending at fold 45 (formation 2025-09-01..2026-02-28, trading 2026-03)
]

def calibrate_fold(fold_id, formation_start, formation_end, trading_start,
                   tickers, spreads_df) -> dict[str, float]:
    """
    Returns {ticker: kappa}. Uses ONLY [formation_start, formation_end] data.
    """
    if formation_end >= trading_start:
        raise ValueError(f"Lookahead: formation_end {formation_end} >= trading_start {trading_start}")

    formation_slice = spreads_df.loc[
        (spreads_df.timestamp_et >= formation_start) &
        (spreads_df.timestamp_et <= formation_end) &
        (spreads_df.is_valid)
    ]
    medians = formation_slice.groupby("ticker")["full_spread_l1_bps"].median()
    return {t: assign_kappa_tier(medians.get(t, 50.0)) for t in tickers}
```

### Step 2 — `hooks.py`

Per-trade, not per-bar.

```python
from src.plan1_cost_model import (
    calculate_spread_cost, calculate_impact_cost, calculate_borrow_cost,
    calculate_round_trip_cost, calculate_static_round_trip_cost,
)

def _lookup(spreads_df, ticker, ts):
    """Returns (half_spread_l1_bps, spread_std_1d) at the bar at-or-before ts."""
    # spreads_df is MultiIndexed [ticker, timestamp_et] for O(1) lookup
    try:
        row = spreads_df.loc[(ticker, ts)]
    except KeyError:
        # Fall back to nearest-prior bar within 1 minute
        row = spreads_df.loc[ticker].loc[:ts].iloc[-1]
    return row["half_spread_l1_bps"], row["spread_std_1d"]

def trade_dynamic_cost(trade_row, spreads_df, kappa_map) -> dict:
    """Returns {spread_$, impact_$, borrow_$, total_$}."""
    hsa_in, sa_in = _lookup(spreads_df, trade_row.ticker_A, trade_row.entry_ts)
    hsb_in, sb_in = _lookup(spreads_df, trade_row.ticker_B, trade_row.entry_ts)
    hsa_out, sa_out = _lookup(spreads_df, trade_row.ticker_A, trade_row.exit_ts)
    hsb_out, sb_out = _lookup(spreads_df, trade_row.ticker_B, trade_row.exit_ts)
    kA = kappa_map[trade_row.ticker_A]
    kB = kappa_map[trade_row.ticker_B]

    e_sp = calculate_spread_cost(hsa_in, trade_row.notional_A_entry) + \
           calculate_spread_cost(hsb_in, trade_row.notional_B_entry)
    e_im = calculate_impact_cost(kA, sa_in, trade_row.notional_A_entry) + \
           calculate_impact_cost(kB, sb_in, trade_row.notional_B_entry)
    x_sp = calculate_spread_cost(hsa_out, trade_row.notional_A_exit) + \
           calculate_spread_cost(hsb_out, trade_row.notional_B_exit)
    x_im = calculate_impact_cost(kA, sa_out, trade_row.notional_A_exit) + \
           calculate_impact_cost(kB, sb_out, trade_row.notional_B_exit)

    # Identify short leg by side; apply borrow only on short notional
    short_notional = (trade_row.notional_A_entry if trade_row.side_A == -1
                      else trade_row.notional_B_entry)
    borrow = calculate_borrow_cost(short_notional, trade_row.entry_ts, trade_row.exit_ts)

    spread_total = e_sp + x_sp
    impact_total = e_im + x_im
    total = calculate_round_trip_cost(e_sp + e_im, 0, x_sp + x_im, 0, borrow)
    # (Above lumps A and B together; equivalent to summing all four legs.)
    return {
        "spread_$": spread_total,
        "impact_$": impact_total,
        "borrow_$": borrow,
        "total_$":  total,
    }

def trade_static_cost(trade_row) -> float:
    return calculate_static_round_trip_cost(
        trade_row.notional_A_entry, trade_row.notional_B_entry,
        trade_row.notional_A_exit,  trade_row.notional_B_exit,
        tc_bps_per_leg=30.0,
    )

def rebalance_dynamic_cost(reb_row, spreads_df, kappa_map) -> float:
    hs, sigma = _lookup(spreads_df, reb_row.ticker, reb_row.rebalance_ts)
    k = kappa_map[reb_row.ticker]
    return calculate_spread_cost(hs, reb_row.notional_rebalanced) + \
           calculate_impact_cost(k, sigma, reb_row.notional_rebalanced)
```

### Step 3 — `orchestrator.py`

```python
def load_spread_lookup() -> pd.DataFrame:
    """Merges spreads_1min and spread_rolling, indexes by [ticker, timestamp_et]."""
    s = pd.read_parquet("data/microstructure/spreads_1min.parquet")
    r = pd.read_parquet("data/microstructure/spread_rolling.parquet")
    df = s.merge(r, on=["timestamp_et", "ticker"], how="left")
    return df.set_index(["ticker", "timestamp_et"]).sort_index()

def run_plan2(week4_dir: Path):
    trade_log     = pd.read_parquet(week4_dir / "trade_log.parquet")
    rebalance_log = pd.read_parquet(week4_dir / "rebalance_log.parquet")
    validate_trade_log(trade_log)
    validate_rebalance_log(rebalance_log)

    spreads_df = load_spread_lookup()

    cost_rows, kappa_audit = [], []
    for fold_id, f_start, f_end, t_start in FOLD_SCHEDULE:
        fold_trades = trade_log[trade_log.fold_id == fold_id]
        if fold_trades.empty:
            continue
        tickers = pd.concat([fold_trades.ticker_A, fold_trades.ticker_B]).unique()
        kappa_map = calibrate_fold(fold_id, f_start, f_end, t_start, tickers, spreads_df.reset_index())
        for t, k in kappa_map.items():
            kappa_audit.append({"fold_id": fold_id, "ticker": t, "kappa": k})

        fold_rebals = rebalance_log[rebalance_log.fold_id == fold_id]
        for trade in fold_trades.itertuples():
            d = trade_dynamic_cost(trade, spreads_df, kappa_map)
            stc = trade_static_cost(trade)
            reb_$ = sum(rebalance_dynamic_cost(r, spreads_df, kappa_map)
                        for r in fold_rebals[fold_rebals.trade_id == trade.trade_id].itertuples())

            total_dyn = d["total_$"] + reb_$
            cost_rows.append({
                "trade_id": trade.trade_id,
                "spread_cost_dollars":    d["spread_$"],
                "impact_cost_dollars":    d["impact_$"],
                "borrow_cost_dollars":    d["borrow_$"],
                "rebalance_cost_dollars": reb_$,
                "total_cost_dollars":     total_dyn,         # primary (dynamic)
                "total_cost_static":      stc,               # comparison
                "total_cost_gross":       0.0,               # comparison
                "net_pnl_dollars":        trade.gross_pnl_dollars - total_dyn,
                "net_pnl_static":         trade.gross_pnl_dollars - stc,
                "net_return":             (trade.gross_pnl_dollars - total_dyn) / trade.allocated_capital,
            })

    cost_log = pd.DataFrame(cost_rows)
    validate_cost_log(cost_log)

    cost_log.to_parquet("data/cost_log.parquet")
    pd.DataFrame(kappa_audit).to_parquet("data/kappa_per_fold.parquet")
```

## Smoke Test (run when Week 4 outputs available)
1. Run on a single fold's trades end-to-end.
2. Verify all three regimes computed; `total_cost_static ≠ total_cost_dollars` for variable-spread trades.
3. Verify `total_cost_dollars ≥ 0` for 100% of trades.
4. Verify `net_pnl_dollars ≤ gross_pnl_dollars` for 100% of trades.
5. Verify rebalance cost lines aggregated by `trade_id` are non-negative.
6. Verify κ values per fold match `data/kappa_per_fold.parquet` audit.
7. Verify schema: `validate_cost_log(cost_log)` passes.
8. Anti-lookahead spot-check: confirm `calibrate_fold` raises on `formation_end >= trading_start`.

---

## ⛔ HARD STOP — Review Before Plan 3

- [ ] `data/cost_log.parquet` exists with three regime columns
- [ ] `data/kappa_per_fold.parquet` exists; ≤ ~5% κ tier changes across consecutive folds (else flag via `red_flags.check_kappa_instability`)
- [ ] `total_cost_dollars ≥ 0` for 100% of trades
- [ ] `net_pnl_dollars ≤ gross_pnl_dollars` for 100% of trades
- [ ] κ re-tiered per fold on formation-window median spread
- [ ] No `formation_end >= trading_start` violations
- [ ] Rebalance cost line items match `rebalance_log` row count (sum check)
- [ ] Three regime columns (`total_cost_dollars`, `total_cost_static`, `total_cost_gross`) all populated

**Removed from prior plan:**
- ❌ Numba `execute_with_slippage` engine
- ❌ "Numba compiles without fallback" hard-stop
- ❌ Static 60 bps as Numba variant (use Plan 1's `calculate_static_round_trip_cost()`)
- ❌ Per-bar hook signatures (replaced with per-trade)
