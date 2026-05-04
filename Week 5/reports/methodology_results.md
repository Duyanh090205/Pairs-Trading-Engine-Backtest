# WEEK 5 QUANT RESEARCH MEMO
## Net-of-Fees Validation — Pairs Trading Strategy
**Date:** 2026-05-03 | **Status:** Real microstructure (data/orderbook.parquet ingested); production run completed

---

## EXECUTIVE SUMMARY

A cointegration-based pairs trading strategy was developed across 45 walk-forward monthly folds spanning January 2022 through March 2026, covering the S&P 500 universe (~500 tickers). Under the Week 4 static 60 bps round-trip cost assumption, the strategy produced an annualised Sharpe ratio previously reported as 1.978 on net-of-cost daily returns. Week 5 replaces that assumption with an empirical three-component dynamic cost model: (1) L1 bid-ask half-spread, (2) spread-instability-scaled market impact, and (3) overnight borrow accrual.

**Main finding (real microstructure):** Dynamic net Sharpe = **0.443**. The strategy survives friction under the dynamic model. The dynamic model finds ~50% less total cost than the static 60 bps assumption (45.3 bps vs 90.87 bps), because the static 30 bps/leg flat rate was calibrated conservatively relative to the actual traded pairs' real spreads.

**Sharpe computation correction (Week 5):** Earlier Sharpe figures (including Week 4's reported 1.978) were computed on exit-date-only daily returns (~90 observations), then scaled by √252. This inflated Sharpe by approximately √(432/90) ≈ 2.2×. Week 5's validated pipeline uses Week 4's bar-level fold equity curves (1-minute mark-to-market, 432 business days) as the gross return base, applying cost as an exit-day debit. This is the authoritative series. All Sharpe, CAGR, MaxDD, and Calmar figures in this memo reflect the corrected methodology.

**Key deviation from synthetic predictions:** Pre-run projections expected real costs to rise to 70–120 bps. The opposite occurred: real L1 spreads for these S&P 500 pairs average 10–12 bps (tighter than the 19–33 bps synthetic proxy), so spread cost fell. Impact cost rose substantially (4.4 bps synthetic → 10.8 bps real) as real spread volatility exceeds synthetic. Net effect: lower total cost.

**DSR finding:** DSR p-value = 0.000 across all three cost regimes. With 50 strategy variants tested over the 5-week research process, E[max SR] = 2.05 (Bailey & López de Prado 2014). The observed gross Sharpe of 0.503 falls well below this threshold — the strategy does not pass the absolute statistical significance screen. Cost stability flags are clean and PBO = 12.3%, but the Sharpe level is not yet distinguishable from selection bias at this trial count. See Section 6 for full discussion.

No critical red flags fired. The `kappa_instability` flag, which triggered on synthetic data (6 tickers drifting), is False on real data (1 ticker drifted) — real spread levels are more stable across folds than the hash-based proxy suggested.

---

## 1. DATA

### 1.1 Strategy Execution Data (Week 4 — Validated)

| Property | Value |
|---|---|
| Source | S&P 500 constituent 1-minute OHLCV |
| Date range | 2022-01-03 to 2026-03-19 |
| Session filter | 09:30–15:59 US/Eastern (390 bars/day) |
| Walk-forward structure | 45 folds: 6-month formation + 1-month trading |
| Total executed trades | **90 trades** across **22 of 45 folds** |
| Total rebalance events | **44 rebalances** across 11 folds |
| Unique tickers traded | **108** (ticker_A ∪ ticker_B) |
| Timestamp format | ISO strings with mixed UTC offsets (-04:00/-05:00), parsed via `pd.to_datetime(utc=True).dt.tz_convert("US/Eastern")` |
| Equity curves | 432 business-day bar-level (1-min) fold equity parquets — used as gross return base |

**Fold schedule (verified):**
- Fold 1: Formation 2022-01-03 → 2022-06-30, Trading 2022-07
- Fold 45: Formation 2025-09-01 → 2026-02-28, Trading 2026-03
- 23 of 45 folds produced zero trades (strategy filtered pairs below Z-threshold entry = 3.0 with Kalman half-life ≤ 6 days)

### 1.2 Microstructure Data (Real Orderbook — Production Run)

| Property | Value |
|---|---|
| Real file | `data/orderbook.parquet`, 4.1 GB, ingested via Plan 0 |
| Schema | `timestamp, ticker, l{1,2,3}_{bid,ask}_{px,sz}` |
| NaN count in real file | 0 |
| Known LOB limitation | `bid_sz == ask_sz` at all levels (symmetric/synthetic LOB — no order-flow imbalance derivable) |
| Plan 0 outputs | `data/microstructure/spreads_1min.parquet`, `spread_rolling.parquet`, `spread_seasonality.parquet`, `spread_summary.parquet` |
| Avg L1 spread (real) | 10–12 bps full spread across all four regime windows |
| Intraday U-shape | Confirmed: 09:30–10:00 open ~14–39 bps (Tier1–3); 15:30–15:59 close ~2–11 bps |

The real data exhibits the expected intraday U-shape absent from the synthetic proxy. Impact cost rose substantially because real `spread_std_1d` (390-bar rolling σ of de-seasonalised L1 spread) is meaningfully larger than the near-constant synthetic variance.

### 1.3 Regime Partition (from Week 4 REGIME_MAP)

| Regime | Folds | Calendar | Macro Context |
|---|---|---|---|
| Late Bear 2022 | 1–6 | Jul–Dec 2022 | VIX 30+, active Fed hiking |
| Early Bull 2023 | 7–18 | Jan–Dec 2023 | Market recovery, disinflation |
| Mid Bull 2024 | 19–30 | Jan–Dec 2024 | Low-vol sustained bull |
| Late Bull 2025–Q1 2026 | 31–45 | Jan 2025–Mar 2026 | Late-cycle, AI-driven narrow breadth |

---

## 2. METHODOLOGY

### 2.1 Scope Boundary

Week 5 **measures** friction on existing Week 4 trades. It does **not**:
- Modify entry/exit signals
- Change position sizing or Kalman δ parameter
- Add regime filters
- Re-optimise pair selection criteria

The explicit design constraint is: **same trades, same positions, three cost regimes, compare side-by-side.**

### 2.2 Three Cost Regimes

| Regime | Formula | Purpose |
|---|---|---|
| **Gross** | $0 cost | Theoretical alpha ceiling |
| **Static 60 bps** | 30 bps/leg × 4 events | Week 4 baseline anchor |
| **Dynamic** | Empirical spread + κ×σ + borrow | Realistic friction |

All metrics reported side-by-side. No cherry-picking.

### 2.3 Three-Component Dynamic Cost

**Spread component** — crosses the literal quoted market at the exact entry/exit bar:

```
C_spread($) = half_spread_l1_bps(t) / 10,000 × notional($)
half_spread_l1_bps = max( (l1_ask - l1_bid) / (2 × mid_px) × 10,000, 0 )
```

The `max(..., 0)` floor prevents crossed-quote artifacts from producing negative (revenue-generating) spread cost.

**Market impact component** — proportional to daily spread volatility (proxy for instantaneous LOB resilience):

```
C_impact($) = κ × spread_std_1d(t) / 10,000 × notional($)
```

`spread_std_1d(t)` is the **390-bar rolling std of the de-seasonalised L1 spread** (subtract intraday bucket median, then 390-bar rolling σ with min_periods=390). First 390 bars use a conservative fallback of 15 bps for `spread_std_1d`.

**κ tier assignment** — calibrated on formation window only (no lookahead):

```
Formation-window median full L1 spread:
  < 8 bps  → κ = 0.3  (Tier 1 Tight: SPY, AAPL, MSFT)
  8–20 bps → κ = 0.5  (Tier 2 Medium: most S&P 500 mid-caps)
  > 20 bps → κ = 0.8  (Tier 3 Wide: VTRS, T, illiquid names)
```

**Real kappa distribution** (131 fold-ticker pairs):

| κ | Count | % |
|---|---|---|
| 0.3 (Tier 1 Tight) | 39 | 29.8% |
| 0.5 (Tier 2 Medium) | 85 | 64.9% |
| 0.8 (Tier 3 Wide) | 7 | 5.3% |

This contrasts sharply with the synthetic prediction of ~95% Tier 2. Real data shows meaningful Tier 1 representation (tightly-spread pairs) and a small Tier 3 tail (wide/illiquid names).

**Borrow component** — overnight accrual on short leg only:

```
C_borrow($) = (50 bps/yr / 10,000) / 365 × short_notional × holding_calendar_days
            = 0 for same-day trades (5 of 90 confirmed intraday)
```

`holding_calendar_days` = `(exit_date − entry_date).days` counts calendar days including weekends. Borrow accrues on every calendar day (Actual/365 convention), so the divisor is 365 not 252.

**Full round-trip cost:**

```
total_cost = C_spread_A(entry) + C_impact_A(entry)  [leg A entry]
           + C_spread_B(entry) + C_impact_B(entry)  [leg B entry]
           + C_spread_A(exit)  + C_impact_A(exit)   [leg A exit]
           + C_spread_B(exit)  + C_impact_B(exit)   [leg B exit]
           + C_borrow(short_leg)
           + Σ [C_spread + C_impact] per rebalance event
```

**Static baseline (exact Week 4 reproduction):**

```
total_cost_static = 30bps/10,000 × (notional_A_entry + notional_B_entry)   [entry]
                  + 30bps/10,000 × (notional_A_exit  + notional_B_exit)    [exit]
```

### 2.4 Lookahead Prevention

κ is calibrated on the formation window only. An explicit guard raises `ValueError` if `formation_end ≥ trading_start` is ever triggered. This was tested and confirmed in the smoke test suite.

### 2.5 Sharpe Computation Methodology

All Sharpe ratios use **Week 4's bar-level fold equity curves** as the gross return base:

1. Load each fold's 1-minute equity parquet, resample to end-of-business-day, compute daily pct_change.
2. Concatenate across all folds (non-overlapping windows) → 432-day gross daily return series.
3. Apply cost as an exit-day debit: `net_daily[exit_date] = gross_daily[exit_date] − cost_$/AUM`.
4. `Sharpe = mean(net_daily) / std(net_daily, ddof=1) × √252`.

This avoids the exit-date-only approach (which uses only ~90 observations and inflates Sharpe by ≈2×).

---

## 3. BEFORE/AFTER TABLE — CORE DELIVERABLE

Based on 90 executed trades, 22 active folds, real microstructure:

| Metric | Gross | Static 60bps | Dynamic |
|---|---|---|---|
| **Sharpe (annualised)** | **0.5031** | 0.3653 | **0.4428** |
| CAGR | 0.81% | 0.57% | 0.69% |
| MaxDD (daily equity approx.) | -2.03% | -2.32% | -2.13% |
| Calmar | 0.402 | 0.244 | 0.324 |
| Win Rate | 73.3% | 56.7% | **62.2%** |
| Avg Trades / Active Fold | 4.09 | 4.09 | 4.09 |
| Avg RT Cost (bps of capital) | 0 | 90.87 | **45.3** |
| % Folds Profitable | 68.2% | 54.5% | **68.2%** |

**Reading the table:**

The dynamic model produces **50.0% less cost** than the static (45.3 bps vs 90.87 bps) on the actual executed trades. Win rate improves from 56.7% (static) to 62.2% (dynamic), and folds profitable recovers fully to 68.2% (matching gross) — meaning no fold-level performance is destroyed by the dynamic cost model.

The Win Rate drop from Gross (73.3%) to Dynamic (62.2%) represents 10 trades that had positive gross alpha insufficient to cover realistic friction.

**Note on "Static 60 bps" naming:** The label refers to the 30 bps/leg design from Week 4. Applied at both entry and exit against actual varied notionals, the effective rate relative to `allocated_capital` is 90.87 bps — higher than the nominal 60 bps label. The benchmark is the same formula used in Week 4, so the relative comparison is valid.

---

## 4. COST WATERFALL

Per-trade averages, in bps of allocated capital. Cost columns are negative (they subtract from gross alpha):

| Partition | N | Gross | Spread | Impact | Borrow | Rebalance | Net |
|---|---|---|---|---|---|---|---|
| **Overall** | **90** | **+260.58** | **-32.72** | **-10.79** | **-0.34** | **-1.44** | **+215.30** |
| Bear 2022 | 13 | +144.25 | -19.49 | -7.67 | -0.36 | -0.48 | +116.25 |
| Bull 2023+ | 77 | +280.23 | -34.95 | -11.31 | -0.34 | -1.60 | +232.02 |

**Cost decomposition (real data):**
- Spread: **72.0%** of total dynamic friction (down from 88.1% in synthetic)
- Impact: **23.7%** (up from 8.5% — real spread volatility is substantially higher)
- Rebalance: **3.2%**
- Borrow: **0.7%** (negligible — consistent with mean ~1.7-day holding period; accrues at 50bps/yr on calendar days)

**Shift from synthetic:** The dramatic increase in impact share (8.5% → 23.7%) reflects that real `spread_std_1d` is far larger than the near-constant variance of the hash-based synthetic proxy. Real markets have genuine intraday spread fluctuations that the synthetic could not capture. Spread's share fell because real L1 spreads (10–12 bps) are significantly tighter than the synthetic proxy (19–33 bps range).

**Bear 2022 vs Bull 2023+:** Bear regime has lower gross alpha (144 bps vs 280 bps) AND lower costs (spread 19.49 bps vs 34.95 bps). The gross alpha reduction is larger than the cost reduction, giving lower net alpha (116 bps vs 232 bps). Real Bear 2022 spreads of ~19.5 bps are consistent with VIX-elevated bid-ask widening, though the sample of 13 trades limits precision.

---

## 5. REGIME-CONDITIONAL ANALYSIS

| Regime | N | Avg L1 Spread | Avg RT Cost | Sharpe Gross | Sharpe Dynamic | Δ Sharpe |
|---|---|---|---|---|---|---|
| Late Bear 2022 | 13 | 11.62 bps | 28.00 bps | 2.670 | 2.263 | -0.406 |
| **Early Bull 2023** | **18** | **11.78 bps** | **23.12 bps** | **0.234** | **-0.033** | **-0.267** |
| Mid Bull 2024 | 9 | 10.42 bps | 24.28 bps | 2.347 | 2.081 | -0.265 |
| Late Bull 2025–Q1 2026 | 50 | 10.95 bps | 61.54 bps | 1.227 | 1.131 | -0.096 |

**Methodology note:** Regime Sharpes are computed over each sub-period's business days (zero-fill within each regime's exit-date span) and are not directly comparable to the global equity-curve-based Sharpe of 0.4428. They measure the annualised signal-to-noise within each sub-period only.

**Critical observation (confirmed on real data):** Early Bull 2023 flips from gross-positive (0.234) to dynamic-negative (-0.033) under friction. The strategy's gross alpha in this regime is thin enough that even 23 bps of real RT cost destroys it.

**Late Bull 2025–Q1 2026 is the most robust regime** (50 trades, Δ Sharpe only -0.096). Largest sample and most recent data — the strategy's edge is strongest and most persistent in the recent period. Note: this regime has higher RT cost (61.54 bps) despite similar spreads (~11 bps), suggesting higher spread volatility (and therefore higher κ×σ impact) in this period.

**Avg L1 spreads are uniformly low (10–12 bps)** across all regimes. This is the primary reason real costs undershot synthetic predictions — the actual traded S&P 500 pairs in this universe are tight-spread names, not the broader hash-based distribution.

---

## 6. OVERFITTING DIAGNOSTICS ON NET RETURNS

| Metric | Gross | Static | Dynamic |
|---|---|---|---|
| Raw Sharpe | 0.5031 | 0.3653 | 0.4428 |
| **DSR p-value** | **0.0000** | **0.0000** | **0.0000** |
| **PBO** | **0.1231** | **0.1231** | **0.1231** |

### DSR (Deflated Sharpe Ratio) — Bailey & López de Prado (2014)

DSR adjusts the observed Sharpe for number of independent trials (strategy variants), finite sample size, and non-normality of returns:

```
DSR = Φ( (SR_obs - E[max_SR]) / σ_SR )
E[max_SR] = Φ⁻¹(1 - 1/n_trials)
```

**DSR = 0.000 (all regimes):** With `n_trials = 50` (conservative estimate of independent strategy configurations evaluated over the 5-week research process — the correct input per Bailey & López de Prado, not fold count), E[max SR] = **2.054**. The observed gross Sharpe of 0.503 is well below this threshold, so the DSR collapses to ~0 across all cost regimes.

**Interpretation:** The strategy does not pass the absolute DSR overfitting screen. A Sharpe of 0.50 is not statistically distinguishable from the maximum you would expect to find by chance across 50 strategy configurations. This does **not** mean the strategy is worthless — cost stability flags are clean, PBO is low (12.3%), and the dynamic model survives friction. But the Sharpe level needs to be approximately 2.05 to clear the DSR bar under 50 trials.

**Recommended action:** Extend the live-trading sample to accumulate more observations (which raises `n_obs` and tightens `σ_SR`), or reduce the number of distinct strategy configurations tested in future research iterations before re-running DSR.

**Note on n_trials:** An earlier version of this pipeline incorrectly used fold count (22) as `n_trials`, which underestimates E[max SR] and inflates DSR. The correct input is the number of *strategy variants* (parameter combinations, model configurations) evaluated across the research process — conservatively 50 for a 5-week programme. Using fold count (22) is a semantic error; folds are IS/OOS splits, not independent strategy trials.

### PBO (Probability of Backtest Overfitting) — López de Prado & Bailey (2014)

Combinatorial cross-validation across fold partitions. PBO = fraction of paths where IS-selected-best underperforms OOS median.

**PBO = 0.1231** (all three regimes): Only 12.3% of combinatorial paths show IS selection failing OOS. Strong evidence against fold-selection overfitting. PBO uses fold count (22) as its input — this is correct; it is a fold-combinatorial test, not a strategy-variant test.

**Note on PBO fix (confirmed):** Earlier implementation produced PBO ≈ 0.40 due to a set-vs-tuple iteration bug in the combinatorial loop. After fixing to consistent tuple iteration, PBO dropped to 0.12. The fix is verified in the current codebase (`overfitting_net.py`).

---

## 7. KILL ZONE ANALYSIS

Strategy entry timing across 13 intraday 30-min buckets:

| Bucket | N Trades | Gross bps | Cost bps | Net bps | Kill Zone? |
|---|---|---|---|---|---|
| 09:30–10:00 | 1 | +217.0 | 22.1 | +194.9 | No |
| **10:00–10:30** | **61** | **+228.0** | **50.4** | **+177.5** | **No** |
| 10:30–11:00 | 11 | +83.2 | 21.3 | +61.9 | No |
| 11:00–11:30 | 5 | +17.8 | 19.8 | **-2.0** | **Yes** |
| 11:30–12:00 | 3 | -26.2 | 13.1 | **-39.3** | **Yes** |
| 12:00–12:30 | 1 | +122.7 | 21.1 | +101.6 | No |
| 13:00–13:30 | 4 | +56.7 | 21.8 | +34.9 | No |
| 14:00–14:30 | 1 | +7923 | 455.9 | +7467 | No (outlier) |
| 15:00–15:30 | 1 | +138.2 | 23.9 | +114.3 | No |
| **15:30–15:59** | **2** | **-2.9** | **8.3** | **-11.2** | **Yes** |

**67.8% of trades (61/90) enter in the 10:00–10:30 window** — this is where the strategy's primary edge is realised (+177 bps net).

**Kill zones (3 buckets, 10 trades = 11.1% of volume):**
- 11:00–11:30: Gross barely positive (18 bps) while costs run 20 bps. Likely lagged post-morning signals firing after initial mean-reversion has partially occurred.
- 11:30–12:00: Gross alpha actually negative (-26 bps) before costs — pair continued to diverge into midday. Costs amplify net loss (-39 bps).
- 15:30–15:59: End-of-day spread widening as market makers reduce exposure before close.

**14:00–14:30 outlier:** Single trade with 7923 bps gross / 7467 bps net. Not flagged as a kill zone (net alpha strongly positive) but warrants scrutiny in a production setting.

---

## 8. SENSITIVITY ANALYSIS (OAT)

9 parameter perturbations around baseline (κ-mult=1.0, borrow=50 bps/yr, L1 spread):

| κ multiplier | Borrow rate | Net Sharpe | Δ vs Baseline |
|---|---|---|---|
| 0.5× | 30 bps/yr | 0.4506 | +0.0078 |
| 0.5× | 50 bps/yr | 0.4504 | +0.0076 |
| 0.5× | 100 bps/yr | 0.4498 | +0.0070 |
| 1.0× | 30 bps/yr | 0.4431 | +0.0002 |
| **1.0× (baseline)** | **50 bps/yr** | **0.4428** | **0.0000** |
| 1.0× | 100 bps/yr | 0.4422 | -0.0006 |
| 1.5× | 30 bps/yr | 0.4354 | -0.0074 |
| 1.5× | 50 bps/yr | 0.4352 | -0.0076 |
| 1.5× | 100 bps/yr | 0.4346 | -0.0082 |
| L2 spread | — | n/a | Deferred |

**Total sensitivity range: 0.435 to 0.451 (spread of 0.016 Sharpe points)**

The strategy is essentially insensitive to the impact and borrow parameters — the entire range of perturbations shifts the Sharpe by at most ±0.008. This follows from the cost decomposition: impact is ~23.7% and borrow is ~0.7% of total friction. The dominant risk factor not tested here is spread level — the L2 sensitivity sweep remains deferred (no L2 column derivable from the symmetric-size LOB).

---

## 9. RED FLAG EVALUATION

| Red Flag | Trigger Condition | Actual Value | Status |
|---|---|---|---|
| `dynamic_cost_blowup` | Mean RT cost > 150 bps | 45.3 bps | **False** |
| `cost_exceeds_alpha` | Net Sharpe < 0 AND Gross > 1.0 | Net = 0.443 | **False** |
| `kappa_instability` | > 2 tickers change tier across folds | 1 ticker drifted | **False** |
| `dsr_degradation` | Failure prob (net) > 10% while failure prob (gross) < 5% | Both failure probs = 100% (DSR=0 for both) | **False** (gross also fails — flag tests differential, not absolute) |
| `math_violation` | Per-row: total_cost < 0 OR net > gross | 0 violations | **False** |

**On `dsr_degradation`:** The flag is False because it only fires when costs *degrade* a previously significant Sharpe (gross DSR > 0.95). Since gross DSR is also 0 (Sharpe of 0.503 << E[max SR] of 2.054), the flag doesn't apply. The absolute DSR failure is more material and is called out explicitly in Section 6 and Section 12.

**On `kappa_instability`:** Only 1 of 108 traded tickers changed κ tier across folds on real data. This is a significant improvement from the synthetic result (6 tickers), confirming that real S&P 500 spread levels are structurally stable — tickers don't oscillate between liquidity tiers fold-to-fold.

**On `dsr_degradation` naming note:** The `check_dsr_degradation` function accepts failure probabilities (1−DSR), not raw DSR values. The call site correctly passes `1−dsr_dynamic` and `1−dsr_gross`. The function's parameter names could mislead a reader into passing raw DSR values — a naming inconsistency with no effect on results but worth fixing.

---

## 10. STRUCTURAL VALIDITY CHECKS (HOLD REGARDLESS OF DATA SOURCE)

The following properties are asserted by the smoke test suite and hold for all 90 trades on real microstructure:

- **No lookahead bias** — κ calibrated on formation window only; `ValueError` guard confirmed
- **No negative costs** — `total_cost_dollars ≥ 0` for all 90 trades
- **Cost strictly subtracts PnL** — `net_pnl_dollars ≤ gross_pnl_dollars` for all 90 trades
- **Component additivity** — `spread + impact + borrow + rebalance == total_cost_dollars` exactly
- **Intraday borrow = 0** — 5 same-day trades carry zero borrow
- **κ values restricted to {0.3, 0.5, 0.8}** — all 131 (fold,ticker) entries validated
- **Three regimes always reported together** — architecture prevents single-regime reporting
- **Schema validation passes** — trade_log, rebalance_log, cost_log schemas formally verified
- **Rebalance attribution** — 44 events across 19 parent trades, all attributed
- **PBO indexing correct** — consistent tuple iteration; PBO 0.40 (wrong) → 0.12 (fixed)
- **DSR/PBO computed on net returns** — confirmed by inspection of `overfitting_net.py`
- **Fold schedule exact match to Week 4** — fold 1 trading 2022-07, fold 45 trading 2026-03
- **Sharpe based on equity-curve returns** — 432-day bar-level series, not exit-date-only

---

## 11. LIMITATIONS AND OPEN QUESTIONS

### 11.1 Symmetric LOB (Primary Remaining Limitation)

The real orderbook has `bid_sz == ask_sz` at all levels — no order-flow imbalance is derivable. Consequences:

1. **No adverse selection model** — impact is estimated purely from spread volatility, not from LOB depth or queue position. For $10k–$20k per-leg notional, book-walking impact is likely negligible, so this is a second-order limitation.
2. **L2 sensitivity deferred** — no meaningful L2 spread column derivable from symmetric-size data. The most material OAT test (walking to L2) remains unavailable.
3. **No intraday liquidity depth** — the U-shape in spreads is captured, but queue resilience at open/close is not.

### 11.2 Negative Control Gap

Week 4 stores NC results as aggregate Sharpe values only — not per-trade logs. Dynamic-cost NC application is architecturally blocked. Static-cost NC pass rate: 8/22 folds (36%). Flagged as Week 6 follow-up.

### 11.3 Impact Model Simplicity

`C_impact = κ × spread_std_1d × notional` is linearised and execution-size-insensitive. It does not model book-walking, permanent vs temporary impact, or adverse selection. For $10k–$20k per-leg notional, book-walking impact is likely negligible at current sizing.

### 11.4 CAGR / MaxDD Approximation

CAGR and MaxDD are computed from daily compounded equity (`(1 + daily_return).cumprod()`), which approximates the true bar-level drawdown. MaxDD is not a true intra-trade bar-level measure. Both are documented approximations consistent across all three cost regimes.

### 11.5 Small Sample Warning

90 trades, ~4 per active fold. Per-fold Sharpe estimates are highly noisy. The 14:00–14:30 outlier (7923 bps gross) has outsized influence on averages in that bucket.

### 11.6 Early Bull 2023 Structural Weakness Persists

18 trades producing Sharpe -0.033 under dynamic costs. While less negative than prior estimates, the regime is still friction-negative. This confirms the strategy has a structural edge gap during post-bear recovery phases — possibly because mean-reversion opportunities are fewer and thinner when correlations rise during recovery rallies.

### 11.7 DSR Absolute Screen Failure

DSR = 0 across all regimes. The strategy needs either a substantially higher Sharpe (>2.05) or a smaller trial count to pass Bailey & López de Prado's deflation screen. This is the primary statistical limitation and the most important open question for Week 6.

---

## 12. QUESTIONS FOR EXTERNAL VALIDATION

1. **κ tier calibration:** Are κ values (0.3 / 0.5 / 0.8 for <8 / 8–20 / >20 bps full spread) empirically reasonable as market impact coefficients for market-order equity pairs at $10k–$20k notional? Real distribution (30% Tier1, 65% Tier2, 5% Tier3) — does this align with practitioner experience for S&P 500 pairs?

2. **DSR application:** `n_trials` is correctly set to 50 (strategy variant count), not fold count. E[max SR] = 2.054 under 50 trials. The observed Sharpe of 0.503 fails this screen. Is 50 a reasonable conservative estimate of variants, or should it be higher (more conservative) or lower (tighter research discipline)?

3. **PBO interpretation:** PBO = 0.12 with 22 folds. Is the combinatorial test sufficiently powered at this sample size?

4. **Impact cost proportion:** At 23.7% of total friction under real data, impact exceeds the typical intuition for small-notional market orders. Is this plausible given the σ-based model (κ × spread_std_1d), or does it suggest the impact coefficient is too aggressive?

5. **Early Bull 2023 flip:** 18 trades producing Sharpe -0.033 under dynamic costs. Structural weakness in recovery-phase markets, or still a model artifact given the symmetric LOB limitation?

6. **14:00–14:30 outlier:** Single trade with 7923 bps gross PnL. What screening / sanity checks are appropriate for such outliers before accepting them at face value?

---

## 13. CODE AUDIT FINDINGS

The following issues were identified and resolved during code review (2026-05-03):

| Finding | File | Severity | Resolution |
|---|---|---|---|
| **C2: Sharpe inflation via exit-date-only returns** | `sharpe_net.py`, `overfitting_net.py`, `sensitivity_oat.py`, `regime_costs.py` | **Critical** | Fixed: use 432-day equity-curve base; apply cost as exit-day debit. Sharpe changed from ~2.5 to 0.44. |
| **B2: Borrow /252 vs /365 unit mismatch** | `borrow_cost.py` | **Moderate** | Fixed: changed to `/365`. `holding_days` counts calendar days (`.days`), so borrow must accrue at calendar-day rate. |
| **B4: No floor on spread — crossed quotes** | `features.py`, `spread_cost.py` | **Moderate** | Fixed: `.clip(lower=0.0)` on all spread columns in Plan 0; `max(..., 0)` guard in spread_cost.py. |
| **C3: n_trials = fold count (22) in DSR** | `overfitting_net.py` | **Moderate** | Fixed: changed to `n_strategy_variants=50`. DSR changed from ~0.93 to 0.000. |
| **CAGR active-days vs full-span mismatch** | `sharpe_net.py` | **Moderate** | Fixed: `_cagr` now uses `(last_date − first_date).days` (1333 cal days) with 365-day year. Previously used `len(daily)=432` active trading days with 252-day year, overstating CAGR by ~2×. CAGR: 1.74%/1.21%/1.48% → 0.81%/0.57%/0.69%. |
| `check_dsr_degradation` parameter naming | `red_flags.py` | Documentation | Parameters named ambiguously (expects 1−DSR, not DSR). Call site correct; rename deferred to Week 6. |
| "Static 60 bps" label vs actual charge | `round_trip.py`, memo | Documentation | Code charges 30 bps × (A+B) at entry AND exit. Effective bps = 90.87 relative to `allocated_capital`. Label refers to per-direction convention from Week 4. |
| L2 sensitivity row returns NaN | `sensitivity_oat.py` | Expected | No L2 spread available from symmetric LOB. Row preserved in output to signal deferred analysis. |

---

## 14. NEXT STEPS

```
Completed (Week 5):
  ✓ Plan 0: data/orderbook.parquet ingested → data/microstructure/*.parquet
  ✓ Plan 2: cost_log.parquet produced (90 trades, dynamic + static + gross)
  ✓ Plan 3: validation suite + net_of_fees_report.md generated
  ✓ All smoke tests green on real microstructure
  ✓ kappa_per_fold.parquet: 131 (fold,ticker) pairs audited
  ✓ C2: Sharpe inflation bug fixed (equity-curve base)
  ✓ B2: Borrow /365 fix (calendar-day convention)
  ✓ B4: Spread floor clip(lower=0) fix (crossed quotes)
  ✓ C3: DSR n_trials corrected to 50 strategy variants

Deferred to Week 6:
  Priority 1: Dynamic-cost negative control
    → Requires per-trade NC logs from Week 4 (currently only aggregate NC Sharpe stored)
    → Re-run Week 4 NC pairs through Plan 2 hooks with synthetic NC timestamps

  Priority 2: L2 sensitivity sweep
    → Requires non-symmetric LOB (bid_sz ≠ ask_sz) or a separate L2 spread column
    → Flag whether current data source can provide this

  Priority 3: Fix check_dsr_degradation parameter naming
    → Rename dsr_net_p → failure_prob_net, dsr_gross_p → failure_prob_gross

  Priority 4: Outlier review
    → 14:00–14:30 trade with 7923 bps gross PnL requires manual verification

  Priority 5: DSR re-evaluation path
    → Extend live-trading sample OR reduce n_strategy_variants to pass DSR screen
    → Re-run with updated n_obs once more fold data is available
```

---

*Pipeline: Plans 0–3 fully implemented and run on real microstructure. All 4 smoke test suites green. 90-trade cost log validated on real data. Dynamic net Sharpe = 0.443 (equity-curve corrected). PBO indexing bug confirmed fixed. DSR uses n_trials=50 (strategy variants). All cost results based on real orderbook.*
