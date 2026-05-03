# Phase 4 Plan — Multi-Regime Defense

## Objective
Orchestrate all 45 rolling walk-forward folds, then run regime analysis, pair persistence, volume stratification, overfitting diagnostics, and OAT sensitivity. Produce the 12 whitepaper sections.

## Input
- `data/validated/5min_phase1.parquet` and `1min_phase2.parquet` (Phase 0)
- Phase 1, 2, 3 modules (import directly)

## No New Reuse from Prior Weeks
All prior-week code is wrapped inside Phase 1–3 modules. Phase 4 is new orchestration logic.

## Phase 2 Implementation Notes for Orchestrator

Carry-forward requirements from Phase 2 audit (see `results/pipeline_results_phase0_phase1.md`):

1. **Call sequence per fold** (orchestrator must enforce):
   ```python
   # Step A: discover pairs
   pairs_df = discovery.run(form_start, form_end)
   # Step B: select delta on formation window (NOT trading window)
   formation_5min = load_5min(form_start, form_end)
   optimal_delta, delta_metrics = select_delta(pairs_df, formation_5min, prev_delta=prev_delta)
   prev_delta = optimal_delta   # carry forward for instability tracking
   # Step C: execute on trading window
   results = run_fold_execution(pairs_df, trading_1min, delta=optimal_delta)
   ```

2. **Delta grid extension for sensitivity**: Current grid `{1e-7, ..., 1e-3}` — δ=1e-7 consistently hits lower boundary across folds. OAT sensitivity (`sensitivity.py`) should include a grid extension test with `{1e-9, 1e-8, 1e-7, 1e-6, 1e-5}`.

3. **Numba warm-up**: First call to `_state_machine` triggers JIT compilation (~2–3s). Run a dummy call at orchestrator startup before fold timing begins:
   ```python
   from src.phase2_execution.engine import _state_machine
   _state_machine(np.zeros(10, dtype=np.float64), 2.0)   # warm up JIT
   ```

4. **Rebalance cost contract**: Phase 2 pre-computes `rebalance_cost` in dollars using `0.003 × |Δβ| × short_notional` (price terms cancel). Phase 3 PnL must consume this directly — do not recompute.

5. **Zero-pair folds (7, 9, 16, 19, 25, 27)**: Orchestrator must skip cleanly — log "no pairs, skip fold N" and record `fold_metrics[n] = None`. Downstream aggregation must handle sparse dict.

6. **Fold 23/40 concentration cap**: 6,008 → ~600 pairs, 5,913 → ~590 pairs after cap. All within N_open_pairs_max=50 budget. No special handling needed beyond what Phase 2 already does.

## Files to Create
- `src/phase4_defense/orchestrator.py`
- `src/phase4_defense/regime.py`
- `src/phase4_defense/persistence.py`
- `src/phase4_defense/volume_strat.py`
- `src/phase4_defense/overfitting.py`
- `src/phase4_defense/sensitivity.py`

---

## Module 4a — Rolling Walk-Forward Orchestrator (`orchestrator.py`)

### Fold Schedule
```
Fold 1:  Formation [2022-07-01 → 2022-12-31]  Trading [2023-01-01 → 2023-01-31]
Fold 2:  Formation [2022-08-01 → 2023-01-31]  Trading [2023-02-01 → 2023-02-28]
...
Fold 45: Formation [2025-09-01 → 2026-02-28]  Trading [2026-03-01 → 2026-03-19]
```

### Embargo Between Formation End and Trading Start
```python
embargo_days = max(1, int(0.5 * (Z_window_bars / 390)))
# Z_window_bars = min(half_life_days × 390, 2000)
# embargo_days computed as days (390 1-min bars = 1 trading day)
```

### Per-Fold Execution Sequence
```python
for fold_n, (form_start, form_end, trade_start, trade_end) in enumerate(fold_schedule):
    # 1. Phase 1: discover pairs on formation window
    pairs_df = discovery.run(form_start, form_end)
    if pairs_df is None or len(pairs_df) == 0:
        log(f"Fold {fold_n}: no surviving pairs, skip")
        continue

    # 2. Phase 2a: select optimal delta on formation window
    optimal_delta, delta_metrics = delta_selector.select_delta(pairs_df, formation_data, R_map)
    if optimal_delta is None:
        log(f"Fold {fold_n}: universal constraint fail, skip")
        continue

    # 3. Phase 2b: run execution engine on trading window
    positions, rebalance_events = engine.run(
        pairs_df, trade_start, trade_end, optimal_delta
    )

    # 4. Phase 3: PnL + metrics + audit log
    fold_metrics = pnl.compute(positions, rebalance_events, trade_start, trade_end)
    neg_control.run(trade_start, trade_end)   # CVNA/ISRG + synthetic
    audit_log.write(fold_n, fold_metrics, optimal_delta, delta_metrics)

    # 5. Store fold outputs
    fold_results[fold_n] = fold_metrics
```

### Cross-Fold Boundary Handling
```python
# Open positions from previous fold carried forward:
# engine tracks open_positions dict → position only closed on natural exit
# Pairs dropped from new universe: no new entries, but existing position runs to close
# New pairs in new fold: entries enabled from fold start
# Kalman state: re-init from [alpha_PCA, beta_PCA] of NEW fold's Phase 1 output
```

### Aggregation Output
```python
# 1. Concatenated bar-level equity curve across all 45 folds
equity_concat = pd.concat([fold_results[n]['bar_equity'] for n in fold_results])
equity_concat.to_parquet('results/metrics/equity_concat.parquet')

# 2. Fold-equal-weighted Sharpe distribution
sharpe_dist = pd.Series([fold_results[n]['sharpe'] for n in fold_results])
sharpe_dist.to_csv('results/metrics/sharpe_distribution.csv')
```

---

## Module 4b — Regime Partition (`regime.py`)

### Fold-to-Regime Mapping
| Regime | Folds | N folds |
|---|---|---|
| Late Bear 2022 | 1–6 (Trading: Jul–Dec 2022*) | 6 |
| Early Bull 2023 | 7–18 (Trading: Jan–Dec 2023) | 12 |
| Mid Bull 2024 | 19–30 (Trading: Jan–Dec 2024) | 12 |
| Late Bull 2025–Q1 2026 | 31–45 (Trading: Jan 2025–Mar 2026) | 15 |

*Note: Fold 1 formation starts Jul 2022 → trading window = Jan 2023 (not Jul 2022). Regime labels reflect formation market conditions. Document this distinction in whitepaper.

### Report Per Regime
```python
for regime, fold_range in regime_map.items():
    regime_sharpes = [fold_results[n]['sharpe'] for n in fold_range]
    print(f"{regime}: mean={np.mean(regime_sharpes):.2f}, "
          f"median={np.median(regime_sharpes):.2f}, "
          f"IQR={np.percentile(regime_sharpes, [25, 75])}, "
          f"pct_positive={np.mean(np.array(regime_sharpes) > 0):.1%}")
```

**Caveat to include in whitepaper:** Bear sample N=6 is underpowered for strong inference.

---

## Module 4c — Pair Persistence (`persistence.py`)

```python
# Step 1: Identify P_2022 from Fold 1 output
P_2022 = fold_results[1]['surviving_pairs']   # pairs from Jul–Dec 2022 formation

# Step 2: For each fold 7+ (trading starts Jan 2023+)
persistence_pct = {}
for fold_n in range(7, 46):
    form_start, form_end, _, _ = fold_schedule[fold_n]
    formation_data = load_formation_data(form_start, form_end)

    still_passing = 0
    for (ticker_A, ticker_B) in P_2022:
        # Re-test Johansen on this pair using this fold's formation window
        if run_johansen(ticker_A, ticker_B, formation_data):
            still_passing += 1

    persistence_pct[fold_n] = still_passing / len(P_2022)

# Output: line chart data
pd.Series(persistence_pct).to_csv('results/metrics/pair_persistence.csv')
```

Output: line chart "% of P_2022 still passing Johansen at fold t" → whitepaper §3.

---

## Module 4d — Volume Stratification (`volume_strat.py`)

Hussein-inspired. Document in whitepaper as OUT-OF-DOMAIN extension, not replication.

```python
for fold_n, (form_start, form_end, _, _) in enumerate(fold_schedule):
    # Compute share-volume ADV per ticker in survivor universe
    formation_data = load_1min_data(form_start, form_end)
    adv_share = {}
    for ticker in survivor_tickers:
        daily_vol = formation_data[ticker]['volume'].resample('D').sum()
        adv_share[ticker] = daily_vol.mean()

    # Tertile split (T1=low, T2=mid, T3=high)
    adv_series = pd.Series(adv_share)
    t1_thresh = adv_series.quantile(1/3)
    t2_thresh = adv_series.quantile(2/3)

    tertile_map = {}
    for ticker, adv in adv_share.items():
        if adv <= t1_thresh:    tertile_map[ticker] = 'T1'
        elif adv <= t2_thresh:  tertile_map[ticker] = 'T2'
        else:                   tertile_map[ticker] = 'T3'

    # Tag pairs by tertile bucket (within-tertile only)
    for pair in fold_results[fold_n]['surviving_pairs']:
        ta = tertile_map.get(pair.ticker_A)
        tb = tertile_map.get(pair.ticker_B)
        if ta == tb:   # same tertile
            pair.tertile_bucket = ta   # T1, T2, or T3

    # Stratify Sharpe by bucket
    bucket_sharpes[fold_n] = {
        'T1': [p.sharpe for p in pairs if p.tertile_bucket == 'T1'],
        'T2': [p.sharpe for p in pairs if p.tertile_bucket == 'T2'],
        'T3': [p.sharpe for p in pairs if p.tertile_bucket == 'T3'],
    }
```

Compare T3-T3 vs T1-T1 across folds → whitepaper §4. Cross-tertile deferred to Week 5+.

---

## Module 4e — Overfitting Diagnostics (`overfitting.py`)

```python
# Deflated Sharpe Ratio (Bailey & López de Prado)
# DSR adjusts for the number of trials and selection bias
from mlfinlab.backtest_statistics import deflated_sharpe_ratio

dsr = deflated_sharpe_ratio(
    sharpe_ratio=aggregate_sharpe,
    sharpe_ratio_stars=[fold_results[n]['sharpe'] for n in fold_results],
    number_of_returns=total_trading_bars
)

# Probability of Backtest Overfitting (PBO)
# k-fold combinatorial path testing
from mlfinlab.backtest_statistics import backtest_overfitting

pbo, sr_distribution = backtest_overfitting(
    returns_per_fold={n: fold_results[n]['daily_returns'] for n in fold_results},
    n_jobs=-1
)
```

Output to whitepaper §5: Raw Sharpe + DSR + PBO.

---

## Module 4f — OAT Sensitivity Grid (`sensitivity.py`)

Pre-registered. One-At-a-Time (OAT) around default config. For robustness reporting only — do NOT tune.

### Default Config Anchor
```python
DEFAULT_CONFIG = {
    'formation_months': 6,
    'trading_months':   1,
    'Z_entry':          2.0,
    'delta':            'auto',   # multi-criterion selection
    'tc_bps':           60,
    'borrow_bps_yr':    50,
    'stop_loss':        None,
    'rebalance_X_pct':  0.10,
    'max_holding':      'EOS',
    'N_open_pairs_max': 50,
}
```

### OAT Sweeps (vary 1, hold others at default)
```python
OAT_GRID = {
    'formation_months': [3, 6, 9],
    'trading_months':   ['2w', '1m', '6w'],
    'Z_entry':          [1.75, 2.0, 2.25],
    'tc_bps':           [30, 45, 60, 75],
    'borrow_bps_yr':    [30, 50, 100],
    'stop_loss':        [None, -0.025, -0.05],
    'rebalance_X_pct':  [0.05, 0.10, 0.20, float('inf')],
    'max_holding':      ['EOS', '1d', '3d'],
    'N_open_pairs_max': [20, 50, 100],
}
# ~27 additional runs × 45 folds = ~1,215 fold-runs
```

### 3 Targeted Combo Runs
```python
COMBO_RUNS = [
    # 1. Tight signal
    {'Z_entry': 1.75, 'formation_months': 3, 'stop_loss': -0.025},
    # 2. Conservative signal
    {'Z_entry': 2.25, 'formation_months': 9, 'stop_loss': None},
    # 3. High cost stress
    {'tc_bps': 75, 'borrow_bps_yr': 100, 'rebalance_X_pct': 0.05},
]
```

### Decision Rule
Final config = **median net Sharpe of default run across 45 folds**. OAT is for robustness reporting only.

### Stop-Loss Robustness (embedded in OAT)
```python
# Per-pair cumulative PnL smoothed to avoid whipsaw on 1-min bars
rolling_pnl_5bar = pair_pnl_pct.rolling(window=5).mean()   # default K=5
if state != 0 and rolling_pnl_5bar < SL_threshold:
    state = 0
    exit_reason = 'SL'

# Sensitivity: SL ∈ {None, -2.5%, -5%} × K ∈ {3, 5, 10}
```

Report: exit reason breakdown (% zero-cross / % SL / % max-holding / % EOS) + avg return per exit → whitepaper §9.

---

## Whitepaper Section Outputs

| § | Content | Output file |
|---|---|---|
| 1 | 45-fold Sharpe distribution | `results/metrics/sharpe_distribution.csv` + `results/figures/sharpe_hist.png` |
| 2 | Regime Sharpe breakdown | `results/metrics/regime_sharpes.csv` + `results/figures/regime_bar.png` |
| 3 | Pair persistence decay curve | `results/metrics/pair_persistence.csv` + `results/figures/persistence_line.png` |
| 4 | Volume sensitivity (T1/T2/T3 Sharpe) | `results/metrics/volume_strat.csv` |
| 5 | Raw Sharpe + DSR + PBO | `results/metrics/overfitting_diagnostics.csv` |
| 6 | OAT grid results + 3 combos | `results/metrics/oat_sensitivity.csv` |
| 7 | NC bootstrap distribution | `results/metrics/nc_bootstrap.csv` + `results/figures/nc_bootstrap_hist.png` |
| 8 | Alpha decay curve (latency) | `results/metrics/latency_decay.csv` + `results/figures/latency_curve.png` |
| 9 | Stop-loss + exit reason breakdown | `results/metrics/exit_reasons.csv` |
| 10 | Cost decomposition gross→net | `results/metrics/cost_decomp.csv` |
| 11 | δ trajectory across folds | `results/metrics/delta_trajectory.csv` + `results/figures/delta_traj.png` |
| 12 | Universe count + binding filter per fold | `results/metrics/universe_counts.csv` |

## Skills to Invoke
- After `orchestrator.py` runs 3 folds successfully, run **`/simplify`** — the fold loop is the outermost hot path. Check for any per-fold data being held in memory unnecessarily (memory leak risk across 45 folds).
- After all 45 folds complete, run **`/review`** — verify regime boundaries, pair persistence logic, and OAT grid results are correctly structured for whitepaper reporting.
- Run **`/review`** again on the sensitivity.py output to confirm OAT results are NOT being used for tuning (pre-registered robustness check only).

## Smoke Test (before hard stop)
1. Run `orchestrator.py` for 3 folds (Folds 1–3) — assert all fold outputs present
2. Assert concatenated equity curve has correct bar count (3 months × 7,800 bars ≈ 23,400 bars)
3. Assert fold-equal-weighted Sharpe distribution has 3 values
4. Assert regime partition fold assignments are correct (Fold 1–6 = Late Bear 2022)
5. Run `persistence.py` on P_2022 at Fold 7 — assert output is a percentage in [0, 100]
6. Run `volume_strat.py` on Fold 1 — assert pairs tagged T1/T2/T3, no cross-tertile pairs
7. Run 1 OAT variation (Z_entry=1.75 vs default Z_entry=2.0) for 1 fold — confirm different trade count
8. Audit all 12 whitepaper output files exist after full 45-fold run

---
## ⛔ HARD STOP — Final Review Before Whitepaper

**Before writing the whitepaper, verify:**
- [ ] All 45 folds complete without unhandled errors
- [ ] Regime boundaries match pipeline spec exactly (Fold 1–6 = Bear, 7–18 = Early Bull, etc.)
- [ ] Pair persistence curve exported and shows decay over time (expected — cointegration is not permanent)
- [ ] Volume stratification: T3-T3 vs T1-T1 comparison present
- [ ] DSR and PBO computed and in output
- [ ] OAT grid confirms decision rule: final config = median of DEFAULT run (not best OAT)
- [ ] All 12 whitepaper output files exist in `results/metrics/` and `results/figures/`
- [ ] `/simplify` run — fold loop memory safety confirmed
- [ ] `/review` run — OAT results confirmed as robustness (not tuning)

**Signal to proceed:** User explicitly types "Phase 4 approved, begin whitepaper"
