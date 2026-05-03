# Phase 3 Plan — Backtest & Validation Framework

## Objective
Bar-level PnL engine, negative control bootstrap, latency decay test, and per-fold audit logs. Produces the metrics and evidence that goes into the whitepaper.

## Input
- `data/validated/1min_phase2.parquet` (Phase 0)
- Signals and positions from Phase 2 engine
- Pair metadata from Phase 1 (for sizing)

## Phase 2 Interface Contract (what PnL.py must consume correctly)

Engine `pair_df` column semantics — DO NOT recompute these in Phase 3:

| Column | Contract |
|---|---|
| `position` | Already 1-bar execution-lagged. Use directly for fill detection. |
| `signal` | Pre-lag signal — use ONLY for timestamp verification (`exec_ts > signal_ts`). |
| `beta_post` | Read at **entry bar** (`position[t] != 0 and position[t-1] == 0`) to size shares_B. |
| `rebalance_cost` | Pre-computed dollars. Add directly to cost decomposition — do not recompute. |
| `rebalance_event` | Bool flag. Pair with `rebalance_cost` for audit log section [2]. |

Entry detection: `position[t] != 0 and position[t-1] == 0`
Exit detection: `position[t] == 0 and position[t-1] != 0`

Shares at entry:
```python
shares_a = long_notional / close_a[entry_bar]
shares_b = (short_notional * beta_post[entry_bar]) / close_b[entry_bar]
```
Direction: `position = +1` → long A, short B. `position = -1` → short A, long B.

## Cross-Phase Dependencies for Phase 4

- `select_delta` is separate from `run_fold_execution`. Orchestrator must call them in sequence:
  `select_delta(pairs_df, formation_5min, prev_delta) → optimal_delta` then pass to `run_fold_execution`.
- `prev_delta` must be tracked across folds by the orchestrator for instability detection.
- 6 zero-pair folds (7, 9, 16, 19, 25, 27): orchestrator must skip execution and PnL for these folds — no errors, just no output.

## Reuse from Prior Weeks
| Code | Source | Adaptation needed |
|---|---|---|
| PnL bar-level calculation | `Week 3/scripts/engine.py` (CP5.3) | Adapt: use bar-level equity for MaxDD (not daily MTM) |
| Sharpe / CAGR computation | `Week 3/scripts/engine.py` | Direct reuse |
| MaxDD on equity curve | `Week 3/scripts/engine.py` | Change from daily to bar-level |
| Calmar ratio | `Week 3/scripts/engine.py` | Adapt: use bar-level MaxDD |
| Trade log structure | `Week 3/scripts/engine.py` | Direct reuse |

## New Code Required
| Component | Why not in prior weeks |
|---|---|
| Borrow cost daily accrual | Not in Week 3 TC model |
| Rebalance cost separate tracking | Not in Week 3 (rebalance didn't exist) |
| Block bootstrap NC threshold | Week 3 used fixed |Sharpe| < 0.5 as NC pass criterion |
| Latency sweep wrapper | Not in prior weeks |
| 7-section audit log | Week 3 had simpler logging |

## Files to Create
- `src/phase3_backtest/pnl.py`
- `src/phase3_backtest/metrics_runner.py`
- `src/phase3_backtest/neg_control.py`
- `src/phase3_backtest/latency.py`
- `src/phase3_backtest/audit_log.py`

---

## Module 3a — PnL Engine (`pnl.py`)

### Bar-Level PnL (adapt from `Week 3/scripts/engine.py`)
```python
# Bar-level PnL (1-min resolution)
pnl_a = shares_A * (price_A_t - price_A_prev)
pnl_b = shares_B * (price_B_t - price_B_prev)
bar_pnl = pnl_a + pnl_b

# Entry cost (at entry bar only)
if position_changed_to_nonzero:
    entry_cost = 0.003 * (shares_A * price_A_t + shares_B * price_B_t)   # 30bps

# Exit cost (at exit bar only)
if position_changed_to_zero:
    exit_cost = 0.003 * (shares_A * price_A_t + shares_B * price_B_t)   # 30bps
```

### Borrow Cost (NEW — not in Week 3)
```python
# Accrued daily at close on short leg notional
# Default: 50 bps/year
borrow_cost_daily = (50 / 10_000) / 252 * short_notional_t
# Charged at end of each trading day, even if no rebalance
```

### Cost Decomposition Tracking
Track separately in the output:
1. Entry/exit commission (bps)
2. Borrow cost (cumulative)
3. Rebalance cost (cumulative, from threshold breaches in Phase 2)

### Equity Curve (bar-level compound from 1.0)
```python
daily_returns = bar_pnl.resample('D').sum() / total_capital
equity = (1 + daily_returns).cumprod()
equity.iloc[0] = 1.0
```

### Metrics (adapt from `Week 3/scripts/engine.py`)
```python
# Sharpe: daily returns × √252
sharpe = (daily_returns.mean() / daily_returns.std(ddof=1)) * np.sqrt(252)

# MaxDD: bar-level (NOT daily MTM — critical for intraday strategy)
bar_equity = (1 + bar_pnl / total_capital).cumprod()   # 1-min resolution
rolling_max = bar_equity.cummax()
max_dd = (bar_equity / rolling_max - 1).min()   # most negative

# CAGR
n_trading_days = len(daily_returns)
cagr = (equity.iloc[-1] ** (252 / n_trading_days)) - 1

# Calmar: CAGR / bar-level MaxDD
calmar = cagr / abs(max_dd)
```

---

## Module 3b — Negative Control (`neg_control.py`)

### Two Controls (run through same engine per fold)
1. Empirical: CVNA/ISRG (known non-cointegrated)
2. Synthetic: 2 simulated random walks (i.i.d. Gaussian)

### Block Bootstrap Pass Criterion (NEW — replaces fixed threshold from Week 3)
```python
from scipy.signal import resample

def block_bootstrap_sharpe(nc_daily_returns, n_bootstrap=1000, block_size=1):
    """block_size = 1 trading day"""
    sharpes = []
    n = len(nc_daily_returns)
    n_blocks = n // block_size

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        block_starts = rng.integers(0, n_blocks, size=n_blocks) * block_size
        resampled = np.concatenate([
            nc_daily_returns.iloc[s:s+block_size].values
            for s in block_starts
        ])[:n]
        sharpe = (resampled.mean() / resampled.std(ddof=1)) * np.sqrt(252)
        sharpes.append(sharpe)

    return np.array(sharpes)

nc_bootstrap = block_bootstrap_sharpe(nc_daily_returns)
threshold = nc_bootstrap.mean() + 2 * nc_bootstrap.std()
# Primary strategy aggregate Sharpe must exceed threshold
```

---

## Module 3c — Latency Sweep (`latency.py`)

Wrapper around Phase 2 engine — shifts `position_executed[t]` by additional lag.

```python
def run_with_latency(positions_raw, additional_lag):
    # positions_raw already has t+1 baseline lag from engine
    # Apply additional shift
    positions_lagged = np.roll(positions_raw, additional_lag)
    positions_lagged[:additional_lag] = 0
    return positions_lagged

latency_configs = {
    't+1': 0,    # baseline (already in engine)
    't+2': 1,    # 1 additional bar
    't+5': 4,    # 4 additional bars
    't+10': 9,   # 9 additional bars
    'random': None   # Uniform(1, 5) extra bars per trade
}
```

**Pass criterion:**
- Sharpe at t+5 > 0: signal has genuine cointegration-based edge
- Graceful degradation (no cliff): Sharpe decreases monotonically with latency
- If t+2 kills Sharpe → signal is microstructure noise → FAIL

**Run on default config only.** Do NOT cross with OAT grid.

---

## Module 3d — Audit Log (`audit_log.py`)

Per-fold `.txt` with 7 sections (8 logged items, some combined):

```
=== FOLD {N} AUDIT LOG ===
Generated: {timestamp}

[1] PARAMETER HASH
config_hash = {sha256 of fold config dict}
Formation: {start} to {end}
Trading: {start} to {end}
delta: {optimal_delta}

[2] KALMAN δ + MULTI-CRITERION METRICS
Selected δ: {value}
Grid results:
  δ=1e-7: kurt={}, HL={}, ACF78={}
  ...
Constraints passed: HL ∈ [1,10]d={}, ACF78>0.7={}

[3] TRADE LOG SAMPLE (first 20 trades)
ticker_A | ticker_B | entry_ts | exit_ts | state | gross_bps | net_bps | exit_reason

[4] COMPARATIVE METRICS
Primary Sharpe: {}, MaxDD: {}, CAGR: {}, Calmar: {}
NC Empirical (CVNA/ISRG) Sharpe: {}
NC Bootstrap threshold (mean+2σ): {}
NC PASS/FAIL: {}

[5] TIMESTAMP VERIFICATION PROOF
Total trades: {}
Trades where exec_ts > signal_ts: {} (must be 100%)
LOOKAHEAD STATUS: PASS / FAIL

[6] NEGATIVE CONTROL BOOTSTRAP
Bootstrap Sharpes — mean: {}, std: {}
Primary Sharpe: {} (vs threshold: {})
Discrimination STATUS: PASS / FAIL

[7] RED FLAG STATUS + ENVIRONMENT HASH
Lookahead flag (Sharpe>5): {}
NC discrimination fail: {}
Kalman spread degenerate: {}
δ boundary: {}
δ instability: {}
Universal constraint fail: {}
Python: {}, numpy: {}, pandas: {}, statsmodels: {}, numba: {}
Platform: {}
```

### Red Flag Triggers
| Trigger | Condition | Action |
|---|---|---|
| Lookahead leak | Sharpe > 5.0 | Halt, log ERROR, do not proceed |
| NC discrimination fail | Primary Sharpe ≤ NC bootstrap mean+2σ | Halt, log ERROR |
| Kalman spread degenerate | var(prior) > 2× var(static PCA spread) OR < 0.1× | Log WARNING, flag δ |
| δ boundary | Auto-selected δ at grid edge | Log WARNING |
| δ instability | δ jumps >2 grid steps vs previous fold | Log WARNING |
| Universal constraint fail | No δ passes all 3 criteria | Log ERROR, skip fold |

---

## Module 3e — Metrics Runner (`metrics_runner.py`)

Per fold AND aggregated:
- Sharpe (daily), MaxDD (bar-level), CAGR, Calmar, Win Rate
- Trade count, avg holding time (in 1-min bars), turnover
- Fold-equal-weighted Sharpe distribution (histogram data for whitepaper §1)
- Concatenated equity curve (bar-level, for whitepaper §1)

## Skills to Invoke
- After implementing all 5 modules, run **`/simplify`** — focus on PnL calculation loops and the bootstrap code (performance-sensitive).
- After 1-fold smoke test passes all red flags, run **`/review`** against pipeline spec §3.1–3.8 (walk-forward arch, TC model, PnL mechanics, NC validation, latency, red flags, audit log).

## Smoke Test (before hard stop)
1. Run full fold 1 (Fold 1: formation Jul–Dec 2022, trading Jan 2023) end-to-end (Phase 1 → 2 → 3)
2. Assert Sharpe < 5.0 (red flag: no lookahead)
3. Assert `exec_ts > signal_ts` for 100% of trades
4. Assert NC (CVNA/ISRG) Sharpe < bootstrap threshold (NC discrimination pass)
5. Assert latency sweep: Sharpe(t+5) > 0 and monotonically decreasing
6. Assert cost decomposition: 3 separate line items in output (commission + borrow + rebalance)
7. Assert audit log `.txt` created for Fold 1 in `results/logs/`
8. Bar-level equity curve exported to `results/metrics/fold01_equity.parquet`

---
## ⛔ HARD STOP — Review Before Phase 4

**Before proceeding to Phase 4, verify:**
- [ ] Fold 1 completes without any red flag (Sharpe < 5.0, NC pass, no lookahead)
- [ ] Bar-level MaxDD is computed (not daily MTM)
- [ ] Cost decomposition has 3 separate items: commission, borrow, rebalance
- [ ] Latency sweep: Sharpe(t+5) > 0 for Fold 1
- [ ] Audit log `.txt` generated for Fold 1 with all 7 sections
- [ ] Bootstrap NC threshold computed and primary Sharpe exceeds it
- [ ] `/simplify` run — PnL loop and bootstrap code reviewed
- [ ] `/review` run — confirmed against pipeline spec §3.1–3.8

**Signal to proceed:** User explicitly types "Phase 3 approved, proceed to Phase 4"
