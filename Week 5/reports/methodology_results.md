# WEEK 5 QUANT RESEARCH MEMO
## Net-of-Fees Validation — Pairs Trading Strategy
**Date:** 2026-05-03 | **Status:** Real microstructure (data/orderbook.parquet ingested); production run completed

---

## EXECUTIVE SUMMARY

A cointegration-based pairs trading strategy was developed across 45 walk-forward monthly folds spanning January 2022 through March 2026, covering the S&P 500 universe (~500 tickers). Under the Week 4 static 60 bps round-trip cost assumption, the strategy produced an annualised Sharpe ratio of **1.978** on net-of-cost daily returns. Week 5 replaces that assumption with an empirical three-component dynamic cost model: (1) L1 bid-ask half-spread, (2) spread-instability-scaled market impact, and (3) overnight borrow accrual.

**Main finding (real microstructure):** Dynamic net Sharpe = **2.531**. The strategy survives friction under the dynamic model. The dynamic model finds ~50% less total cost than the static 60 bps assumption (45.4 bps vs 90.87 bps), because the static 30 bps/leg flat rate was calibrated conservatively relative to the actual traded pairs' real spreads.

**Key deviation from synthetic predictions:** Pre-run projections expected real costs to rise to 70–120 bps. The opposite occurred: real L1 spreads for these S&P 500 pairs average 10–12 bps (tighter than the 19–33 bps synthetic proxy), so spread cost fell. Impact cost rose substantially (4.4 bps synthetic → 10.8 bps real) as real spread volatility exceeds synthetic. Net effect: lower total cost and a higher dynamic Sharpe than synthetic.

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
half_spread_l1_bps = (l1_ask - l1_bid) / (2 × mid_px) × 10,000
```

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
C_borrow($) = (50 bps/yr / 10,000) / 252 × short_notional × holding_calendar_days
            = 0 for same-day trades (5 of 90 confirmed intraday)
```

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

---

## 3. BEFORE/AFTER TABLE — CORE DELIVERABLE

Based on 90 executed trades, 22 active folds, real microstructure:

| Metric | Gross | Static 60bps | Dynamic |
|---|---|---|---|
| **Sharpe (annualised)** | **2.8218** | 1.9777 | **2.5306** |
| CAGR | 9.92% | 6.34% | 8.12% |
| MaxDD (daily equity approx.) | -0.22% | -0.39% | -0.23% |
| Calmar | 45.18 | 16.37 | 35.15 |
| Win Rate | 73.3% | 56.7% | **62.2%** |
| Avg Trades / Active Fold | 4.09 | 4.09 | 4.09 |
| Avg RT Cost (bps of capital) | 0 | 90.87 | **45.4** |
| % Folds Profitable | 68.2% | 54.5% | **68.2%** |

**Reading the table:**

The dynamic model produces **50.0% less cost** than the static (45.4 bps vs 90.87 bps) on the actual executed trades. Win rate improves from 56.7% (static) to 62.2% (dynamic), and folds profitable recovers fully to 68.2% (matching gross) — meaning no fold-level performance is destroyed by the dynamic cost model.

The Win Rate drop from Gross (73.3%) to Dynamic (62.2%) represents 10 trades that had positive gross alpha insufficient to cover realistic friction.

**Note on "Static 60 bps" naming:** The label refers to the 30 bps/leg design from Week 4. Applied at both entry and exit against actual varied notionals, the effective rate relative to `allocated_capital` is 90.87 bps — higher than the nominal 60 bps label. The benchmark is the same formula used in Week 4, so the relative comparison is valid.

---

## 4. COST WATERFALL

Per-trade averages, in bps of allocated capital. Cost columns are negative (they subtract from gross alpha):

| Partition | N | Gross | Spread | Impact | Borrow | Rebalance | Net |
|---|---|---|---|---|---|---|---|
| **Overall** | **90** | **+260.58** | **-32.72** | **-10.79** | **-0.50** | **-1.44** | **+215.15** |
| Bear 2022 | 13 | +144.25 | -19.49 | -7.67 | -0.52 | -0.48 | +116.09 |
| Bull 2023+ | 77 | +280.23 | -34.95 | -11.31 | -0.49 | -1.60 | +231.87 |

**Cost decomposition (real data):**
- Spread: **72.0%** of total dynamic friction (down from 88.1% in synthetic)
- Impact: **23.7%** (up from 8.5% — real spread volatility is substantially higher)
- Rebalance: **3.2%**
- Borrow: **1.1%** (negligible — consistent with mean ~1.7-day holding period)

**Shift from synthetic:** The dramatic increase in impact share (8.5% → 23.7%) reflects that real `spread_std_1d` is far larger than the near-constant variance of the hash-based synthetic proxy. Real markets have genuine intraday spread fluctuations that the synthetic could not capture. Spread's share fell because real L1 spreads (10–12 bps) are significantly tighter than the synthetic proxy (19–33 bps range).

**Bear 2022 vs Bull 2023+:** Bear regime has lower gross alpha (144 bps vs 280 bps) AND lower costs (spread 19.49 bps vs 34.95 bps). The gross alpha reduction is larger than the cost reduction, giving lower net alpha (116 bps vs 231 bps). Real Bear 2022 spreads of ~19.5 bps are consistent with VIX-elevated bid-ask widening, though the sample of 13 trades limits precision.

---

## 5. REGIME-CONDITIONAL ANALYSIS

| Regime | N | Avg L1 Spread | Avg RT Cost | Sharpe Gross | Sharpe Dynamic | Δ Sharpe |
|---|---|---|---|---|---|---|
| Late Bear 2022 | 13 | 11.62 bps | 28.16 bps | 9.542 | 7.683 | -1.859 |
| **Early Bull 2023** | **18** | **11.78 bps** | **23.23 bps** | **0.827** | **-0.120** | **-0.947** |
| Mid Bull 2024 | 9 | 10.42 bps | 24.41 bps | 8.566 | 7.348 | -1.219 |
| Late Bull 2025–Q1 2026 | 50 | 10.95 bps | 61.72 bps | 3.263 | 2.998 | -0.265 |

**Critical observation (confirmed on real data):** Early Bull 2023 flips from gross-positive (0.827) to dynamic-negative (-0.120) under friction. The flip persists but is substantially smaller than under synthetic data (-0.378 synthetic vs -0.120 real). The strategy's gross alpha in this regime is thin enough that even 23 bps of real RT cost destroys it.

**Note on Late Bear 2022 Sharpe = 9.542:** Small-sample artifact from 13 trades in 6 folds. Not statistically reliable as a standalone estimate.

**Late Bull 2025–Q1 2026 is the most robust regime** (50 trades, Δ Sharpe only -0.265). Largest sample and most recent data — the strategy's edge is strongest and most persistent in the recent period. Note: this regime has higher RT cost (61.72 bps) despite similar spreads (~11 bps), suggesting higher spread volatility (and therefore higher κ×σ impact) in this period.

**Avg L1 spreads are uniformly low (10–12 bps)** across all regimes. This is the primary reason real costs undershot synthetic predictions — the actual traded S&P 500 pairs in this universe are tight-spread names, not the broader hash-based distribution.

---

## 6. OVERFITTING DIAGNOSTICS ON NET RETURNS

| Metric | Gross | Static | Dynamic |
|---|---|---|---|
| Raw Sharpe | 2.8218 | 1.9777 | 2.5306 |
| **DSR p-value** | **0.9306** | **0.7981** | **0.9120** |
| **PBO** | **0.1231** | **0.1231** | **0.1231** |

### DSR (Deflated Sharpe Ratio) — Bailey & López de Prado (2014)

DSR adjusts the observed Sharpe for number of independent trials (folds), finite sample size, and non-normality of returns:

```
DSR = Φ( (SR_obs - E[max_SR]) / σ_SR )
E[max_SR] = Φ⁻¹(1 - 1/n_trials)
```

**DSR = 0.912 (Dynamic):** After adjusting for 22 trials and observed return distribution non-normality, ~91.2% posterior probability that the true Sharpe under dynamic costs is positive. Conventional thresholds: 0.5 (meaningful), 0.8 (strong). **Passes.** Improved from 0.895 on synthetic data, consistent with higher dynamic Sharpe on real data.

### PBO (Probability of Backtest Overfitting) — López de Prado & Bailey (2014)

Combinatorial cross-validation across fold partitions. PBO = fraction of paths where IS-selected-best underperforms OOS median.

**PBO = 0.1231** (all three regimes): Only 12.3% of combinatorial paths show IS selection failing OOS. Strong evidence against fold-selection overfitting.

**Note on PBO fix (confirmed):** Earlier implementation produced PBO ≈ 0.40 due to a set-vs-tuple iteration bug in the combinatorial loop. After fixing to consistent tuple iteration, PBO dropped to 0.12. The fix is verified in the current codebase (`overfitting_net.py`).

---

## 7. KILL ZONE ANALYSIS

Strategy entry timing across 13 intraday 30-min buckets:

| Bucket | N Trades | Gross bps | Cost bps | Net bps | Kill Zone? |
|---|---|---|---|---|---|
| 09:30–10:00 | 1 | +217.0 | 22.1 | +194.9 | No |
| **10:00–10:30** | **61** | **+228.0** | **50.6** | **+177.4** | **No** |
| 10:30–11:00 | 11 | +83.2 | 21.5 | +61.8 | No |
| 11:00–11:30 | 5 | +17.8 | 19.9 | **-2.1** | **Yes** |
| 11:30–12:00 | 3 | -26.2 | 13.2 | **-39.4** | **Yes** |
| 12:00–12:30 | 1 | +122.7 | 21.4 | +101.3 | No |
| 13:00–13:30 | 4 | +56.7 | 21.9 | +34.8 | No |
| 14:00–14:30 | 1 | +7923 | 455.9 | +7467 | No (outlier) |
| 15:00–15:30 | 1 | +138.2 | 24.0 | +114.1 | No |
| **15:30–15:59** | **2** | **-2.9** | **8.4** | **-11.3** | **Yes** |

**67.8% of trades (61/90) enter in the 10:00–10:30 window** — this is where the strategy's primary edge is realised (+177 bps net).

**Kill zones (3 buckets, 10 trades = 11.1% of volume):**
- 11:00–11:30: Gross barely positive (18 bps) while costs run 20 bps. Likely lagged post-morning signals firing after initial mean-reversion has partially occurred.
- 11:30–12:00: Gross alpha actually negative (-26 bps) before costs — pair continued to diverge into midday. Costs amplify net loss (-39 bps).
- 15:30–15:59: End-of-day spread widening as market makers reduce exposure before close.

**14:00–14:30 outlier:** Single trade with 7923 bps gross / 7467 bps net. This is a data artifact or a very large statistical arb compression event. Not flagged as a kill zone (net alpha strongly positive) but warrants scrutiny in a production setting.

---

## 8. SENSITIVITY ANALYSIS (OAT)

9 parameter perturbations around baseline (κ-mult=1.0, borrow=50 bps/yr, L1 spread):

| κ multiplier | Borrow rate | Net Sharpe | Δ vs Baseline |
|---|---|---|---|
| 0.5× | 30 bps/yr | 2.5736 | +0.0430 |
| 0.5× | 50 bps/yr | 2.5711 | +0.0405 |
| 0.5× | 100 bps/yr | 2.5649 | +0.0343 |
| 1.0× | 30 bps/yr | 2.5331 | +0.0025 |
| **1.0× (baseline)** | **50 bps/yr** | **2.5306** | **0.0000** |
| 1.0× | 100 bps/yr | 2.5244 | -0.0063 |
| 1.5× | 30 bps/yr | 2.4919 | -0.0387 |
| 1.5× | 50 bps/yr | 2.4894 | -0.0413 |
| 1.5× | 100 bps/yr | 2.4830 | -0.0476 |
| L2 spread | — | n/a | Deferred |

**Total sensitivity range: 2.483 to 2.574 (spread of 0.091 Sharpe points)**

The strategy is essentially insensitive to the impact and borrow parameters. This follows from the cost decomposition: impact is ~23.7% and borrow is ~1.1% of total friction. The dominant risk factor not tested here is spread level — the L2 sensitivity sweep remains deferred (no L2 column derivable from the symmetric-size LOB).

---

## 9. RED FLAG EVALUATION

| Red Flag | Trigger Condition | Actual Value | Status |
|---|---|---|---|
| `dynamic_cost_blowup` | Mean RT cost > 150 bps | 45.4 bps | **False** |
| `cost_exceeds_alpha` | Net Sharpe < 0 AND Gross > 1.0 | Net = 2.531 | **False** |
| `kappa_instability` | > 2 tickers change tier across folds | 1 ticker drifted | **False** |
| `dsr_degradation` | Failure prob (net) > 10% while failure prob (gross) < 5% | 8.8% vs 6.9% | **False** |
| `math_violation` | Per-row: total_cost < 0 OR net > gross | 0 violations | **False** |

**On `kappa_instability`:** Only 1 of 108 traded tickers changed κ tier across folds on real data. This is a significant improvement from the synthetic result (6 tickers), confirming that real S&P 500 spread levels are structurally stable — tickers don't oscillate between liquidity tiers fold-to-fold.

**On `dsr_degradation` implementation note:** The `check_dsr_degradation` function accepts failure probabilities (1−DSR), not raw DSR values. The call site correctly passes `1−dsr_dynamic = 0.088` and `1−dsr_gross = 0.069`. The function's parameter names (`dsr_net_p`, `dsr_gross_p`) could mislead a reader into passing raw DSR values — a naming inconsistency with no effect on results but worth fixing.

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

`n_trading_days` in the CAGR calculation is the count of distinct exit dates (not the full backtest calendar span), slightly overstating annualised returns. MaxDD is a daily-equity approximation (PnL aggregated by exit date), not true intra-trade bar-level drawdowns. Both are documented approximations.

### 11.5 Small Sample Warning

90 trades, ~4 per active fold. Per-fold Sharpe estimates are highly noisy. Bear 2022 gross Sharpe of 9.54 from 13 trades should not be taken at face value. The 14:00–14:30 outlier (7923 bps gross) has outsized influence on averages in that bucket.

### 11.6 Early Bull 2023 Structural Weakness Persists

18 trades producing Sharpe -0.12 under dynamic costs. While less negative than synthetic (-0.378), the regime is still friction-negative. This confirms the strategy has a structural edge gap during post-bear recovery phases — possibly because mean-reversion opportunities are fewer and thinner when correlations rise during recovery rallies.

---

## 12. QUESTIONS FOR EXTERNAL VALIDATION

1. **κ tier calibration:** Are κ values (0.3 / 0.5 / 0.8 for <8 / 8–20 / >20 bps full spread) empirically reasonable as market impact coefficients for market-order equity pairs at $10k–$20k notional? Real distribution (30% Tier1, 65% Tier2, 5% Tier3) — does this align with practitioner experience for S&P 500 pairs?

2. **DSR application:** Is `n_trials` correctly set to the count of folds with non-NaN Sharpe values (22)? Or should it count total strategy variants tested across all of Week 4?

3. **PBO interpretation:** PBO = 0.12 with 22 folds. Is the combinatorial test sufficiently powered at this sample size?

4. **Impact cost proportion:** At 23.7% of total friction under real data, impact exceeds the typical intuition for small-notional market orders. Is this plausible given the σ-based model (κ × spread_std_1d), or does it suggest the impact coefficient is too aggressive?

5. **Early Bull 2023 flip:** 18 trades producing Sharpe -0.12 under dynamic costs. Structural weakness in recovery-phase markets, or still a model artifact given the symmetric LOB limitation?

6. **14:00–14:30 outlier:** Single trade with 7923 bps gross PnL. What screening / sanity checks are appropriate for such outliers before accepting them at face value?

---

## 13. CODE AUDIT FINDINGS

The following issues were identified during code review (2026-05-03). No logic bugs were found.

| Finding | File | Severity | Notes |
|---|---|---|---|
| `check_dsr_degradation` parameter naming | `red_flags.py` | Documentation | Parameters named `dsr_net_p / dsr_gross_p` but function expects `1−DSR` (failure probability). Call site is correct; naming could cause misuse. |
| "Static 60 bps" label vs actual charge | `round_trip.py`, memo | Documentation | Code charges 30 bps × (A+B) at entry AND exit = 4 half-spread crossings. Effective bps relative to `allocated_capital` is 90.87, not 60. Label refers to the per-direction cost convention from Week 4. No effect on comparisons. |
| L2 sensitivity row returns NaN | `sensitivity_oat.py` | Expected | No L2 spread available from symmetric LOB. Row preserved in output to signal deferred analysis. |
| CAGR uses exit-date count not calendar days | `sharpe_net.py` | Known approx. | `n_trading_days = len(daily)` = distinct exit dates, not full backtest span. Documented as approximation. |

All structural checks pass: no negative costs, no net > gross violations, no lookahead, PBO indexing correct.

---

## 14. NEXT STEPS

```
Completed (Week 5):
  ✓ Plan 0: data/orderbook.parquet ingested → data/microstructure/*.parquet
  ✓ Plan 2: cost_log.parquet produced (90 trades, dynamic + static + gross)
  ✓ Plan 3: validation suite + net_of_fees_report.md generated
  ✓ All smoke tests green on real microstructure
  ✓ kappa_per_fold.parquet: 131 (fold,ticker) pairs audited

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
```

---

*Pipeline: Plans 0–3 fully implemented and run on real microstructure. All 4 smoke test suites green. 90-trade cost log validated on real data. Dynamic net Sharpe = 2.531 (real) vs 2.413 (synthetic). PBO indexing bug confirmed fixed. All cost results based on real orderbook.*
