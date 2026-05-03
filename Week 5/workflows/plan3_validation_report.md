# Plan 3 — Validation & Report (Phase 4, 5, 6)

> [!WARNING]
> **Blocked by Plan 2:** Requires `data/cost_log.parquet` and Week 4 fold equity parquets.
> `red_flags.py` is the only module currently implemented.

## Objective
Produce the "Net-of-Fees Performance Report." Construct analytics modules, fire red flags, assemble the 12-section deliverable.

## Input
- `data/cost_log.parquet` (from Plan 2)
- `data/kappa_per_fold.parquet` (from Plan 2)
- `data/microstructure/spread_summary.parquet` (from Plan 0)
- Week 4 fold equity parquets: `Week 4/results/metrics/{baseline,final}/equities/fold{NN}_equity.parquet`
- Week 4 `phase4_defense/orchestrator.py:REGIME_MAP`
- Week 4 `phase4_defense/overfitting.py` (DSR / PBO formulas)

## Output
- `reports/net_of_fees_report.md` + supporting figures
- All 8 analytics module outputs (DataFrames or parquet files in `reports/`)

## Reuse from Prior Weeks
| Code | Source | Adaptation |
|---|---|---|
| DSR / PBO formulas | Week 4 `phase4_defense/overfitting.py` | Re-run on net returns |
| `REGIME_MAP` | Week 4 `phase4_defense/orchestrator.py` | Direct reuse |
| Sharpe / MaxDD / CAGR / Calmar | Week 4 `utils/metrics.py` | Direct reuse |

## Files
- `src/plan3_validation/__init__.py` ✓
- `src/plan3_validation/red_flags.py` ✅ **IMPLEMENTED** (7 trigger functions)
- `src/plan3_validation/sharpe_net.py` — to implement
- `src/plan3_validation/cost_waterfall.py` — to implement
- `src/plan3_validation/regime_costs.py` — to implement
- `src/plan3_validation/impact_validation.py` — to implement (DIAGNOSTIC ONLY)
- `src/plan3_validation/kill_zone.py` — to implement
- `src/plan3_validation/negative_control.py` — to implement (path decision required)
- `src/plan3_validation/overfitting_net.py` — to implement
- `src/plan3_validation/sensitivity_oat.py` — to implement
- `src/plan3_validation/report_builder.py` — to implement

## Implementation

### 4.1 — `sharpe_net.py`
```python
def generate_before_after_table(cost_log, fold_equity_dir) -> pd.DataFrame:
    """
    Reconstructs daily net returns under each regime by subtracting
    daily cost / prev_day_equity from daily gross return.
    Source: Week 4 fold equity parquets (bar-level → resampled to daily).

    Output columns: [Gross, Static60bps, Dynamic]
    Rows: Sharpe×√252, CAGR, MaxDD (bar), Calmar, Win Rate,
          Avg Trades/Fold, Avg RT Cost (bps), % Folds Profitable
    """
```

### 4.2 — `cost_waterfall.py`
```python
def generate_cost_waterfall(cost_log) -> pd.DataFrame:
    """
    Per-trade bps decomposition:
      Gross alpha
      − Spread cost
      − Impact cost
      − Borrow cost
      − Rebalance cost
      = Net alpha
    Partitioned by Bear 2022 (folds 1–6) vs Bull 2023+ (folds 7–45).
    """
```

### 4.3 — `regime_costs.py`
```python
REGIME_MAP = {
    "Late Bear 2022":         range(1, 7),
    "Early Bull 2023":        range(7, 19),
    "Mid Bull 2024":          range(19, 31),
    "Late Bull 2025-Q12026":  range(31, 46),
}

def analyze_regime_costs(cost_log, spreads_df) -> pd.DataFrame:
    """
    Per regime: avg L1 spread, avg dynamic RT cost (bps),
                Sharpe gross, Sharpe net, Δ Sharpe.
    """
```

### 4.4 — `impact_validation.py` (DIAGNOSTIC ONLY)
```python
def validate_impact_prediction(spreads_df) -> dict:
    """
    Cross-sectional OLS: (full_spread_l2_bps − full_spread_l1_bps) ~ spread_std_1d.
    Reports β, R², saves scatter to reports/impact_validation.png.

    NOT a pass/fail gate. Supports the macro thesis that
    spread instability predicts depth decay.
    """
```

### 4.5 — `kill_zone.py`
```python
def analyze_kill_zone(cost_log, spreads_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    13 × 30-min intraday buckets.
    Part A: avg spread by bucket × tier → U-shape line chart.
    Part B: avg net alpha by bucket × regime → kill_zone heatmap.
    """
```

### 4.6 — `negative_control.py`
> [!IMPORTANT]
> **Path decision before implementation:**
> Week 4 stores NC results as Sharpe values in `fold_metrics.csv` columns `nc_threshold` / `nc_pass`, **not** as a separate trade log.
>
> **Option A (default):** Re-run NC pairs (CVNA/ISRG empirical, synthetic random walk) through Plan 2 hooks using NC trade timestamps from Week 4 audit logs.
> **Option B:** Skip NC under dynamic; document the gap explicitly in §9 of the report.
>
> Decide before implementing.

```python
def run_dynamic_negative_control(nc_source: str, cost_log) -> pd.DataFrame:
    """
    For each NC pair × {Gross, Static, Dynamic}: compute Sharpe.
    Pass criteria:
      - NC Sharpe under Dynamic ≈ 0 (within bootstrap 2σ of zero)
      - check_nc_leak() returns False
    """
```

### 4.7 — `overfitting_net.py`
```python
def compute_overfitting_diagnostics(cost_log) -> pd.DataFrame:
    """
    Recompute DSR and PBO on net daily returns.
    Reuses formulas from Week 4 phase4_defense/overfitting.py.

    Output: 3 rows × 3 columns
    Rows: Raw Sharpe / DSR p-value / PBO
    Cols: Gross / Static / Dynamic
    """
```

### 4.8 — `sensitivity_oat.py`
```python
def run_oat_sensitivity(cost_log, trade_log, spreads_df) -> pd.DataFrame:
    """
    9 OAT runs: kappa_mult {0.5, 1.0, 1.5} × borrow_rate {30, 50, 100} × spread_level {L1, L2}.

    For kappa_mult: re-aggregate impact_cost_dollars × multiplier (no trade re-loop).
    For borrow_rate: re-aggregate borrow_cost_dollars × ratio (no trade re-loop).
    For spread_level=L2: requires per-trade re-lookup of l2 spreads → trade re-loop needed.

    Output: 9 rows × [params, net_sharpe_dynamic, delta_vs_baseline].
    """
```

### 4.9 — `report_builder.py`

Assembles 12-section Markdown report + figures into `reports/`:

| § | Content | Source |
|---|---|---|
| 1 | Executive Summary (one-line verdict) | `red_flags.check_cost_exceeds_alpha()` |
| 2 | Empirical Spread Profile | `spread_summary.parquet` + 4.5A |
| 3 | Slippage Model Spec | `kappa_per_fold.parquet` + 4.4 |
| 4 | Before/After Table | 4.1 |
| 5 | Cost Waterfall | 4.2 |
| 6 | Regime-Conditional Costs | 4.3 |
| 7 | Spread-Vol Correlation | 4.4 (diagnostic) |
| 8 | Kill Zone + Seasonality | 4.5 |
| 9 | Negative Control | 4.6 |
| 10 | Overfitting Diagnostics (Net) | 4.7 |
| 11 | Sensitivity Analysis | 4.8 |
| 12 | Verdict | `red_flags` summary |

## Smoke Test (after Plan 2 complete)
1. Run all 8 analytics modules on Plan 2 output; verify non-empty DataFrames.
2. Cost waterfall components sum to gross alpha within 1bp rounding.
3. Regime boundaries match `REGIME_MAP` exactly.
4. NC Sharpe under dynamic ≈ 0 (or Option B documented).
5. All 7 red flag triggers fire correctly on synthetic bad input.
6. Report renders: 12 sections, no broken figure links.

---

## ⛔ HARD STOP — Final Review Before Whitepaper

- [ ] All 8 analytics modules produce output
- [ ] Before/After table fully populated, no empty cells
- [ ] Cost waterfall sums correctly (gross − components = net)
- [ ] Regime boundaries match Week 4 `REGIME_MAP`
- [ ] NC under dynamic costs: Sharpe ≈ 0 OR Option B documented
- [ ] DSR/PBO computed on net returns (not gross)
- [ ] OAT sensitivity completed (9 runs)
- [ ] All 7 red flag triggers tested
- [ ] No cherry-picking — all three regimes shown side-by-side
- [ ] §4.4 framed as diagnostic, not pass/fail

**Removed from prior plan:**
- ❌ §4.4 framed as a validation gate — it's diagnostic-only per workflow
- ❌ Vague "Load NC pair trades" — replaced with explicit Option A/B decision
- ❌ Generic "regime partition matches Week 4" — replaced with explicit `REGIME_MAP` import
