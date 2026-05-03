# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Week 4 of a Quant Finance program. The goal is a **cointegration pairs trading strategy** that defends across Bear (2022) and Bull (2023–2026) regimes. Deliverable: Strategy Whitepaper.

**Definitive pipeline spec:** `docs/pipeline/pipeline_week4.md` — all design decisions, parameters, and critical lock-ins are there. Read it before modifying any phase logic.

**Per-phase implementation plans** (with hard stops and skill invocations): `docs/pipeline/phase{0-4}_plan.md`

## Running the Code

All commands run from the repo root with `src/` on the Python path:

```bash
# Run Phase 0 gateway (full universe, 2022-01-03 to 2026-03-19)
python -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from src.phase0_data_gateway.gateway import run_gateway
run_gateway()
"

# Smoke test a specific ticker subset
python -c "
import sys; sys.path.insert(0, '.')
from src.phase0_data_gateway.gateway import run_gateway
run_gateway(date_start='2022-01-03', date_end='2022-01-31', tickers=['AAPL','MSFT'])
"

# Inspect validated output
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.io import read_5min, read_1min, list_validated_tickers
print(list_validated_tickers('5min'))
print(read_5min('AAPL').head())
"
```

## Architecture

### Data Flow

```
data/minute_ohlc_flatfiles/{TICKER}_{YYYY-MM-DD}.csv   ← 546K files, ns UTC timestamps
        ↓ Phase 0 (gateway.py)
data/validated/
    5min_phase1/{TICKER}.parquet    ← log_close + volume, 09:35-15:55 ET
    1min_phase2/{TICKER}.parquet    ← OHLCV, 09:30-15:59 ET
    meta_flags.parquet              ← bad-data audit trail
    gateway_summary.json
        ↓ Phase 1 (discovery.py) — called per fold by orchestrator
    surviving pairs + [α_PCA, β_PCA, half_life_days, R]
        ↓ Phase 2 (kalman.py + delta_selector.py + engine.py)
    positions, rebalance events
        ↓ Phase 3 (pnl.py + neg_control.py + latency.py + audit_log.py)
    results/metrics/ + results/logs/
        ↓ Phase 4 (orchestrator.py + regime/persistence/volume_strat/overfitting/sensitivity)
    results/figures/ + whitepaper inputs
```

### Module Responsibilities

**`src/utils/`** — shared across all phases, no phase imports:
- `io.py`: `load_ticker_raw()` (CSV→ET DatetimeIndex), `read_5min/1min()`, `discover_tickers()`
- `stats.py`: `compute_ou_halflife()`, `bh_fdr_correct()`, `block_bootstrap_sharpe()`, `rolling_zscore()`
- `metrics.py`: `compute_sharpe/max_dd/cagr/calmar/win_rate()`, `compute_all()`

**`src/phase0_data_gateway/gateway.py`** — standalone; all other phases read its parquet outputs, never raw CSVs directly. Call `run_gateway()` once before any phase.

**`src/phase1_cointegration/discovery.py`** (to be built) — called per fold. Inputs: 5min parquet slice + formation window dates. Outputs: DataFrame of surviving pairs.

**`src/phase2_execution/`** (to be built) — three modules: `kalman.py` (2D state [α,β]), `delta_selector.py` (multi-criterion grid search on formation window), `engine.py` (Numba `@njit` state machine).

**`src/phase3_backtest/`** (to be built) — `pnl.py`, `neg_control.py`, `latency.py`, `audit_log.py`.

**`src/phase4_defense/orchestrator.py`** — defines FOLD_SCHEDULE (45 folds), REGIME_MAP, and result-loading utilities. Run `python run_phase4.py` (analytical, ~5s) or with `--persistence --volume-strat --structural-oat` flags for slow analyses.

### Rolling Walk-Forward Architecture

45 monthly folds: 6-month formation window → 1-month trading window, rolling by 1 month.

**CRITICAL — Fold schedule** (§4.1 regime labels refer to TRADING windows, not formation):
```
Fold 1:  Formation 2022-01-03 → 2022-06-30  |  Trading Jul 2022
Fold 2:  Formation 2022-02-01 → 2022-07-31  |  Trading Aug 2022
...
Fold 7:  Formation 2022-07-01 → 2022-12-31  |  Trading Jan 2023
Fold 45: Formation 2025-09-01 → 2026-02-28  |  Trading Mar 2026
```
When calling `discovery.run(formation_start, formation_end)`, the dates are the FORMATION window, never the trading window.

**Frozen per fold** (set from Phase 1 on formation data, never changed during trading): pair list, `[α_PCA, β_PCA]` Kalman init, δ (auto-selected), R (measurement noise), Z thresholds.

**Updated bar-by-bar** during trading: Kalman state `θ̂(t)`, rolling Z mean/std, `β_ref` on threshold rebalance.

**Fold boundary**: open positions carry forward to natural exit (zero-cross). Kalman re-inits from new `[α_PCA, β_PCA]` each fold.

## Critical Design Decisions (from pipeline spec)

- **PCA hedge ratio**: secondary eigenvector only (`v_2`, not `v_1`). `β_PCA = -v_2[0]/v_2[1]`. Never use OLS for hedge ratio.
- **Kalman spread**: use **prior** state (`θ_{t|t-1}`) for signal generation. Posterior (`θ_{t|t}`) is for sizing only. Using posterior gives kurtosis ~13 (untradeable noise).
- **Kalman Q**: Implemented as `δ · R · I₂` (spec says `δ · I₂`, but R normalisation is necessary — see phase2_plan.md). NOT `δ/(1-δ) · I` from Week 2.
- **Kalman init**: `[α_PCA, β_PCA]` with `P₀ = R · I₂`.
- **δ selection**: multi-criterion per fold — minimize `|kurtosis - 3|` subject to `median_HL ∈ [1,10] days` AND `median_ACF(lag=78) > 0.7`. Grid: `{1e-7, 1e-6, 1e-5, 1e-4, 1e-3}`.
- **Z-score window** (Phase 2): `half_life_days × 390` bars (1-min), capped at 2000.
- **MaxDD**: bar-level (1-min equity curve), NOT daily MTM.
- **No inner join at Phase 0**: pairwise inner join with ≥80% overlap happens in Phase 1 per pair.
- **BH-FDR**: applied over all Johansen p-values within a fold at `q=0.05`.

## Data Format

Raw CSVs: `ticker,volume,open,close,high,low,window_start,transactions`
- `window_start` is **nanosecond UTC integer** — convert with `pd.to_datetime(x, unit='ns', utc=True).dt.tz_convert('US/Eastern')`
- File naming: `{TICKER}_{YYYY-MM-DD}.csv` in `data/minute_ohlc_flatfiles/`

Validated parquet index is tz-aware `US/Eastern` DatetimeIndex. Phase 1 reads `log_close` (already log-transformed) from 5min parquets. Phase 2 reads raw `close` from 1min parquets for PnL.

## Agent Skills Available

- `.agents/skills/brainbytes-dev-everything-claude-trading-kalman-filters/SKILL.md` — read **before** implementing `kalman.py`. Contains Q/R calibration guidance, production quality gates, and diagnostic checklist that maps directly to the multi-criterion δ selection.
- `.agents/skills/eddiecunningham-linsdex-linsdex/SKILL.md` — JAX parallel Kalman (deferred to Week 5+).
- `.agents/skills/lobehub-skills-search-engine/SKILL.md` — search marketplace for additional reference skills.

## Implementation Status

| Module | Status |
|---|---|
| `src/utils/` (io, stats, metrics) | ✅ Complete |
| `src/phase0_data_gateway/gateway.py` | ✅ Complete — 528/528 tickers, 2 issues fixed |
| `src/phase1_cointegration/discovery.py` | ✅ Complete — 45 folds, 20,035 pairs, 0 errors |
| `src/phase2_execution/kalman.py` | ✅ Complete — 2D Kalman, Joseph stabilised P update, Q=δ·R·I |
| `src/phase2_execution/delta_selector.py` | ✅ Complete — multi-criterion, 500-bar burn-in, instability flag |
| `src/phase2_execution/engine.py` | ✅ Complete — Numba state machine, EOS@15:55, concentration cap |
| `src/phase3_backtest/` (5 modules) | ✅ Complete — 22/22 smoke test, 5 bugs fixed, 45-fold run done |
| `src/phase4_defense/` (6 modules) | ✅ Complete — 10/12 whitepaper outputs generated; §3/§4 need --persistence/--volume-strat |

**Phase 2 key deviation**: `Q = δ·R·I₂` not `δ·I₂`. Documented in `docs/pipeline/phase2_plan.md`.
**Phase 2 interface**: Phase 3 PnL reads `position`, `beta_post`, `rebalance_cost` from engine output. See `docs/pipeline/phase3_plan.md` for contract.
**Phase 3 key fixes**: Entry/exit pricing corrected to i−1 bar (execution price). Kalman degenerate check uses kurtosis, not cross-scale R ratio. NC beta uses OLS not std-ratio.
**Phase 4 key fixes**: (1) TC sigma formula: `|arith_annual/Sharpe|` not `|CAGR/Sharpe|` — CAGR is geometric, underestimates σ for catastrophic folds. (2) PBO sampling: `random.sample(all_combos, 10000)` not sequential enumeration — prevents early-fold bias. (3) P_2022 fallback: Fold 7 has 0 pairs; persistence.py falls back to Fold 6. (4) Fold schedule month-anchor fix in orchestrator.py. See `results/pipeline_results.md` §"Phase 4 Spec Inconsistency Audit" for full list.

**Pipeline results**: `results/pipeline_results.md` — all phase results, bugs, deviations, open issues.

**Hard stop between phases**: each phase must be explicitly approved before the next begins. See `docs/pipeline/phase{N}_plan.md` for the approval checklist and required phrase.
