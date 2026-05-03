# Signal Logic Document — The Rules of Engagement

## Week 2: Z-Score Signal Engine for Pairs Trading

**Author:** [Your Name]  
**Date:** April 2026  
**Primary Pair:** CMS / DUK (Utilities) | **Secondary Pair:** DOW / LYB (Materials)  

---

## Execution Specification

This section is the complete, self-contained rule set. A developer who reads only this page can implement the engine with zero ambiguity.

### Parameters (all derived from formation period Jan–Jun 2022)

| Parameter | CMS / DUK | DOW / LYB | Source |
|-----------|----------:|----------:|--------|
| α (intercept) | −0.6956 | −0.8770 | OLS on formation log-prices |
| β (hedge ratio) | 1.0487 | 1.0828 | OLS on formation log-prices |
| Half-life (1-min bars) | 679.7 | 777.8 | AR(1) on formation spread |
| Rolling window W | 680 | 778 | `clip(round(HL), 10, 2000)` |
| Entry threshold | ±2.0 | ±2.0 | Fixed; same for all pairs |
| Exit threshold | 0.0 | 0.0 | Fixed; same for all pairs |
| Session warmup | 30 bars | 30 bars | Applied every session open |
| Burn-in | 340 bars | 389 bars | `W // 2`; signal_valid = False |
| Execution lag | 1 bar | 1 bar | `position_executed[t] = position[t−1]` |

### Spread

```
S(t) = log(A_t) − α − β · log(B_t)
```

### Rolling Z-Score

```
rolling_mean(t) = mean(S[t−W+1 : t])
rolling_std(t)  = std(S[t−W+1 : t], ddof=1)
Z(t)            = (S(t) − rolling_mean(t)) / max(rolling_std(t), 1e-10)
```

### State Machine — Exact Rules

```
DEFINITIONS:
  "zero-crossing" = the Z-score changes sign:
      position == +1 (long)  and Z(t) >= 0.0  →  exit
      position == -1 (short) and Z(t) <= 0.0  →  exit
  "NaN bar" = Z(t) is NaN (burn-in or session warmup)

ENTRY (only when position == 0, flat):
  If Z(t) >  +2.0  →  position = -1  (short spread)
  If Z(t) <  -2.0  →  position = +1  (long spread)

EXIT (only when position ≠ 0):
  If Z(t) crosses zero (see definition above)  →  position = 0  (flat)

HOLD (all other cases):
  If Z(t) is NaN       →  hold current position unchanged
  If position ≠ 0 and no zero-crossing  →  hold current position unchanged

CONSTRAINTS:
  - No re-entry until the current position exits via zero-crossing.
  - A single bar CANNOT trigger both an exit and a new entry.
  - An exit requires an actual sign change, not merely touching zero.
  - Entry threshold (±2.0) is fixed and identical for all pairs.
```

### Session Warmup — Exact Application

The warmup applies to **every trading session**, not just the first day:

1. At each calendar date boundary (new session detected by date change in the index), the first 30 bars (30 minutes from 09:30) have their Z-score set to NaN.
2. During warmup bars, the state machine applies the HOLD rule — position is unchanged.
3. If a position was open at the prior session's 15:59 close, it remains open through the warmup. The engine does not force-close at session boundaries.

### Trade Count Convention

**All trade counts reported in this document are after warmup application** unless explicitly labeled "(no warmup)".

| Pair | Trades (trading period, with warmup) | Trades/day |
|------|-------------------------------------:|-----------:|
| CMS / DUK | 90 | 0.71 |
| DOW / LYB | 72 | 0.57 |

### Output Columns

The signal CSV contains these columns for each 1-minute bar in the trading period:

| Column | Type | Description |
|--------|------|-------------|
| `spread` | float64 | `log(A) − α − β·log(B)` |
| `rolling_mean` | float64 | Rolling mean over W bars |
| `rolling_std` | float64 | Rolling std (ddof=1) over W bars |
| `zscore` | float64 | `(spread − rolling_mean) / rolling_std`; NaN during burn-in and warmup |
| `position` | int8 | +1 / −1 / 0 from state machine |
| `signal_valid` | bool | False during burn-in; True otherwise |
| `position_executed` | int8 | `position.shift(1)` — the column for Week 3 PnL |

---

## Engineered for Speed

The engine processes ~97,000 bars per pair. Speed matters because the 11-pair diagnostic sweep must complete in under 60 seconds total.

| Component | Implementation | Throughput |
|-----------|---------------|-----------|
| Z-score | `pandas.DataFrame.rolling()` — vectorized C under the hood | ~97k bars in <50 ms |
| State machine | Numba `@njit(cache=True)` — compiled to native x86 at first import, cached to disk | ~97k bars in <5 ms |
| OLS hedge ratio | `numpy.linalg.lstsq` — LAPACK dgesv, no scipy dependency | ~48k bars in <10 ms |
| Data loading | Glob + `pd.concat` + inner join; one-pass dedup | ~500 CSVs per pair in <2 s |
| Kalman filter | Python loop with 2×2 numpy matrix ops (see Appendix A for why Numba is not applied here) | ~97k bars in <1 s |
| Full 11-pair sweep | Serial loop (no multiprocessing needed at this scale) | **~35 seconds total** |

Design choices:
- **`fastmath=False` in Numba.** `fastmath=True` enables LLVM `nnan` flag, causing `np.isnan()` to return False for actual NaN values. This silently produces false entries on burn-in and warmup bars. Correctness is non-negotiable.
- **`cache=True` in Numba.** The JIT compilation cost (~200 ms) is paid once and persisted to a `__pycache__` directory. Subsequent runs load the cached machine code in <1 ms.
- **No multiprocessing.** At 35 seconds for 11 pairs, parallelization adds complexity (pickling, shared state) for negligible gain. If the universe scales to 100+ pairs, a `ProcessPoolExecutor` with pair-level granularity is the natural upgrade.

---

## 1. Pair Selection

### 1.1 How these pairs were chosen

Pairs were selected from a Week 1 Engle-Granger cointegration scan across 32,000+ candidate pairs. The scan ran an ADF test on OLS residuals — the standard formal stationarity test for cointegration-based strategies (Vidyamurthy, 2004; Chan, 2013).

| Pair | Category | EG t-statistic | Raw p-value | Interpretation |
|------|----------|:--------------:|:-----------:|----------------|
| CMS / DUK | Primary | −5.268 | 0.000051 | Strongly rejects no-cointegration |
| DOW / LYB | Secondary | −4.861 | 0.000294 | Strongly rejects no-cointegration |
| GOOG / GOOGL | Benchmark | −5.283 | 0.000048 | Structural pair (same company) |
| CVNA / ISRG | Neg. control | +0.778 | 0.994 | No cointegration evidence |
| INTC / JPM | Neg. control | +0.132 | 0.989 | No cointegration evidence |

**On statistical significance.** Benjamini-Hochberg corrected p-values across the full 32,000-pair scan do not reject at conventional thresholds. This is expected — BH controls the False Discovery Rate across the entire universe simultaneously. The individual EG t-statistics (−5.27 and −4.86) are the appropriate evidence for pairs selected on economic grounds. The correct framing is: **candidate cointegrated pairs with strong individual-level test evidence**, not "confirmed cointegrated pairs."

### 1.2 Economic rationale

- **CMS / DUK:** U.S. regulated utilities. Similar revenue models (rate-regulated electricity distribution), correlated regulatory exposure, structurally similar sensitivity to interest rates and natural gas prices.
- **DOW / LYB:** Large-cap U.S. specialty chemicals. Correlated feedstock costs (ethylene, propylene), similar cyclical exposure to industrial demand.

### 1.3 Negative controls and benchmark

The negative controls (CVNA/ISRG, INTC/JPM) are cross-sector, economically unrelated stocks with near-zero EG test statistics. If the engine produces profitable signals on these, it is fitting noise. The benchmark (GOOG/GOOGL) is a structural pair — dual share classes of the same company — that calibrates what "easy" looks like.

---

## 2. Data

### 2.1 Session filter

Raw CSVs contain pre-market and after-hours bars. These are filtered by converting timestamps to `America/New_York` and retaining only bars between 09:30 and 15:59 ET inclusive. DST transitions are handled by the tz-aware filter: EDT maps to UTC 13:30–19:59, EST maps to UTC 14:30–20:59.

**Why this matters.** A naive UTC-fixed filter clips the first hour of every summer session — losing 60 bars/day for 7 months and biasing the half-life estimate.

### 2.2 Alignment

Both legs are inner-joined on a shared DatetimeIndex. Bars where either leg has a missing close are dropped. This guarantees zero NaN values post-merge.

| Pair | Total bars (full year) | Formation | Trading |
|------|----------------------:|----------:|--------:|
| CMS / DUK | 96,770 | 47,889 | 48,881 |
| DOW / LYB | 96,770 | 47,889 | 48,881 |

---

## 3. Spread Construction

### 3.1 The log-price spread

```
S(t) = log(A_t) − α − β · log(B_t)
```

Log prices make the relationship scale-invariant and the regression residuals approximately homoskedastic. β is the price elasticity of leg A with respect to leg B.

### 3.2 OLS hedge ratio — formation period only

`log(A) = α + β·log(B) + ε` fit via OLS on formation data exclusively. No trading-period data enters this regression.

| Parameter | CMS / DUK | DOW / LYB |
|-----------|----------:|----------:|
| α | −0.6956 | −0.8770 |
| β | 1.0487 | 1.0828 |

A positive β near 1.0 is expected — same-sector stocks with correlated fundamentals co-move approximately 1:1 in log-price space.

---

## 4. Spread Characterization

### 4.1 Half-life of mean reversion

The spread follows an Ornstein-Uhlenbeck process: `dS = −κ·S·dt + σ·dW`. The half-life is `HL = ln(2)/κ`, estimated via AR(1) regression on the formation spread. A finite, positive HL is the **necessary condition** for a Z-score strategy to have any statistical edge.

| Pair | Half-life (bars) | ≈ Trading days |
|------|----------------:|:--------------:|
| CMS / DUK | 679.7 | 1.74 |
| DOW / LYB | 777.8 | 1.99 |

Both pairs revert on a ~2-day timescale. Chan (2013) notes strong pairs typically have half-lives of 5–40 days at daily frequency; 680–778 minutes at 1-minute frequency is the equivalent regime.

### 4.2 Hurst exponent (supporting diagnostic only)

| Pair | H (formation) | Interpretation |
|------|:-------------:|----------------|
| CMS / DUK | 0.452 | Mean-reverting |
| DOW / LYB | 0.483 | Borderline |

Hurst is a supporting diagnostic, not a primary test — it produces no p-value and classical R/S estimation has a known upward bias (~0.62 for true H = 0.5, per Hamed 2007). DOW/LYB's H = 0.483 is statistically indistinguishable from 0.5 given R/S standard errors. The primary evidence for mean-reversion is the EG test (−4.86, p = 0.0003) and the finite half-life (778 bars).

### 4.3 Rolling window derivation

```
W = clip(round(HL), min = 10, max = 2000)
```

Setting W = HL means the rolling statistics capture one full mean-reversion half-cycle. 680 bars = 11.3 hours ≈ 1.74 trading days — the natural timescale over which the OU process reverts 50% toward equilibrium.

---

## 5. Entry Threshold: Why 2.0 and Not 1.5

### 5.1 Window-truncation bias

Even with W = HL, the empirical Z-score std is **1.34** for CMS/DUK (not the theoretical 1.0). The inflation exists because successive spread observations within one half-life window are autocorrelated — mechanically irreducible with a finite rolling window. The GOOG/GOOGL benchmark (W ≈ HL = 20 bars, short autocorrelation) has Z std of only 1.19, confirming the mechanism.

**Consequence:** A nominal Z = 2.0 corresponds to 2.0 / 1.34 ≈ **1.49 true σ**. A nominal Z = 1.5 corresponds to 1.5 / 1.34 ≈ **1.12 true σ**.

### 5.2 Signal quality comparison: Z = 2.0 vs Z = 1.5

The formation-period Z-score was tested at multiple thresholds. All metrics below are computed on formation data only (no lookahead). Round-trip cost is estimated at 60 bps (4 one-way legs × 15 bps each).

**CMS / DUK (formation, W = 680):**

| Metric | Z = 1.5 | Z = 2.0 | Δ | Edge to |
|--------|--------:|--------:|:-:|:-------:|
| True σ equivalent | 1.12σ | 1.49σ | +33% deeper into tail | **2.0** |
| Trade count | 124 | 88 | −29% fewer trades | **2.0** |
| Avg hold (bars) | 273 | 316 | +16% longer holds | **2.0** |
| Avg hold (hours) | 4.6 | 5.3 | Consistent with 1.74-day HL | **2.0** |
| Total cost (bps) | 7,440 | 5,280 | −29% lower cost | **2.0** |
| Cost per trade (bps) | 60 | 60 | Identical | Neutral |
| Entries in tail (% of bars) | ~26% | ~13% | 2× more selective | **2.0** |

**DOW / LYB (formation, W = 778):**

| Metric | Z = 1.5 | Z = 2.0 | Δ | Edge to |
|--------|--------:|--------:|:-:|:-------:|
| True σ equivalent | 1.06σ | 1.41σ | +33% deeper | **2.0** |
| Trade count | 92 | 74 | −20% fewer | **2.0** |
| Avg hold (bars) | 367 | 385 | +5% longer | **2.0** |
| Total cost (bps) | 5,520 | 4,440 | −20% lower | **2.0** |

### 5.3 The argument against 1.5

At Z = 1.5, the engine enters at **1.12 true standard deviations** — barely outside one sigma. Under a normal distribution, values beyond 1.12σ occur ~26% of the time. One in four bars qualifies for entry. At this level, the majority of entries are capturing routine fluctuation, not genuine divergence from equilibrium. The result:

- **41% more trades** (CMS/DUK: 124 vs 88) with no mechanism to distinguish signal from noise.
- **Shorter average holds** (273 vs 316 bars) — trades enter too early, before the spread has actually diverged, and exit sooner because the deviation was small to begin with.
- **41% higher aggregate cost** (7,440 vs 5,280 bps) — every additional noise-driven entry pays the full 60 bps round-trip ticket without contributing edge.
- **Hold time inconsistent with the reversion timescale.** The half-life is 680 bars ≈ 1.74 days. An average hold of 273 bars (4.6 hours) is only 40% of the half-life — the position exits before the OU process has completed even half its reversion cycle. At Z = 2.0, the 316-bar hold (5.3 hours, 47% of HL) is better aligned but still conservative.

**The core problem:** Z = 1.5 turns the engine from a selective divergence detector into a noise-responsive oscillator. More entries ≠ more edge.

### 5.4 The argument for 2.0

At Z = 2.0 (1.49 true σ), the engine selects the outer ~13% of the empirical distribution. The Gatev et al. (2006) benchmark used the same 2-standard-deviation entry rule. At this level:

- Trade frequency of 0.70/day (CMS/DUK formation) is disciplined for a pair with a 1.74-day reversion timescale.
- Average hold of 316 bars is consistent with the reversion half-cycle — trades enter at genuine divergences and hold through reversion.
- Total cost (5,280 bps over 6 months) is 29% lower than at Z = 1.5.

### 5.5 The adaptive threshold (recommended upgrade)

The adaptive threshold is the 95th percentile of |Z| on the formation period — the empirical equivalent of a "true 2σ event" given the window-truncation artifact:

| Pair | Adaptive Z | True σ equivalent | Trades | Cost reduction vs 2.0 |
|------|----------:|:-----------------:|-------:|:---------------------:|
| CMS/DUK | 2.57 | 1.92σ | 62 | −30% |
| DOW/LYB | 2.59 | 1.82σ | 47 | −37% |

**The case for 2.0 over adaptive is operational.** The adaptive threshold shifts between formation periods and is sensitive to outlier bars. The fixed threshold is predictable, comparable across all pairs, and reproducible. For production, the adaptive threshold is the recommended upgrade.

---

## 6. Results — Trading Period (Out-of-Sample)

All parameters were estimated on the formation period (Jan–Jun 2022). The results below are fully out-of-sample. All trade counts include the 30-bar session warmup.

### 6.1 CMS / DUK (Primary)

| Metric | Value |
|--------|------:|
| OLS α / β | −0.696 / 1.049 |
| Half-life | 679.7 bars (1.74 days) |
| Window | 680 bars |
| Z-score std (trading) | 1.342 |
| Excess kurtosis | −0.146 |
| Coverage gap ±2σ | +8.34 pp |
| Regime shift (cross-val drift) | 0.059 (no shift) |
| Trades (Z = 2.0, with warmup) | **90** |
| Trades/day | 0.71 |
| Rolling windows H < 0.5 | 100% |

**90 trades at 0.71/day** — no regime shift detected (drift 0.059, well below the 0.5 alarm threshold). Mean-reversion persisted across every rolling window in the trading period.

### 6.2 DOW / LYB (Secondary)

| Metric | Value |
|--------|------:|
| OLS α / β | −0.877 / 1.083 |
| Half-life | 777.8 bars (1.99 days) |
| Window | 778 bars |
| Z-score std (trading) | 1.420 |
| Coverage gap ±2σ | +7.89 pp |
| Trades (Z = 2.0, with warmup) | **72** |
| Trades/day | 0.57 |
| Rolling windows H < 0.5 | 100% |

### 6.3 Validation checks

| Check | CMS/DUK | DOW/LYB |
|-------|:-------:|:-------:|
| EG t-stat rejects null? | Yes (−5.27) | Yes (−4.86) |
| Half-life finite? | Yes (680 bars) | Yes (778 bars) |
| H < 0.5 on formation? | Yes (0.452) | Borderline (0.483) |
| 100% MR rolling windows? | Yes | Yes |
| Regime shift detected? | No (0.059) | No |
| Excess kurtosis < 3? | Yes (−0.15) | Yes |

**Benchmark (GOOG/GOOGL):** Z std = 1.19, coverage gap +2.9%, 1,922 trades. This confirms calibration — when W ≈ HL tightly, the gap nearly vanishes.

**Negative controls (CVNA/ISRG, INTC/JPM):** Positive EG t-stats, 66–79% MR windows (unstable), 27–28 trades. The engine does not find signal where none exists.

---

## 7. Flags and Known Limitations

### FLAG 1 — Window-truncation bias: reduced, not eliminated

Z std improved from 1.39 to 1.34 for CMS/DUK with the corrected window. Four alternative pairs with HL > 2,000 still hit the cap. Residual inflation is mechanically irreducible — addressed via the threshold defense in Section 5.

### FLAG 2 — Session warmup suppresses 19% of entries

21 entries removed for CMS/DUK. Whether they were profitable or loss-making is a Week 3 question. The warmup prioritizes signal quality over frequency.

### FLAG 3 — Kalman beta drift is material (see Appendix A)

26% drift on CMS/DUK means static OLS position sizing is structurally misspecified. Recommended response: use Kalman β for position sizing, OLS Z-score for timing. Not yet implemented — Week 3 upgrade.

### FLAG 4 — EG evidence is individual, not portfolio-level

BH-corrected p-values do not reject across the full 32,000-pair universe. Same-sector pair correlation means position sizing must account for cross-pair exposure.

### FLAG 5 — No transaction cost model

All trade counts are gross. At 0.71 trades/day, round-trip cost of 60 bps produces 42.6 bps/day of break-even requirement. Week 3 will determine if average trade PnL exceeds this.

### FLAG 6 — Single formation/trading split

One out-of-sample period. Walk-forward analysis with rolling formation windows is the natural upgrade. Current results are illustrative of the methodology, not evidence of robustness across regimes.

---

## 8. Conclusion

### What this engine does

The static OLS Z-score engine correctly identifies and quantifies mean-reversion in CMS/DUK and DOW/LYB:

- Strong EG cointegration evidence (t < −4.8, p < 0.0003)
- Finite half-lives consistent with intraday reversion (1.7–2.0 trading days)
- 100% of rolling trading windows with H < 0.5
- No excess kurtosis (not genuinely fat-tailed)
- No regime shift (cross-validated drift < 0.5)
- Disciplined frequency: 0.57–0.71 trades/day

### Why 2.0 and not 1.5

Z = 2.0 enters at 1.49 true σ — deep enough to select genuine divergences, producing 29% fewer trades at 29% lower cost than Z = 1.5. Z = 1.5 enters at 1.12 true σ — noise-level, generating 41% more entries with shorter holds that exit before the OU process completes half its reversion cycle. The adaptive threshold (2.57–2.59) is the statistically rigorous upgrade and is recommended for production.

### What comes next (Week 3)

- Leg-level dollar PnL from lagged positions (not spread pct_change)
- Transaction cost model (4 legs × 15 bps per round-trip)
- Kalman β for position sizing with monthly re-estimation
- Walk-forward evaluation across multiple formation/trading windows

---

*All parameters estimated exclusively on the formation period (Jan–Jun 2022). Trading period results are fully out-of-sample.*

*Signal CSVs: `outputs/signals/signals_CMS_DUK.csv` (48,881 rows), `outputs/signals/signals_DOW_LYB.csv` (49,075 rows)*

---
---

## Appendix A — Kalman Filter: Dynamic Hedge Ratio Analysis

### Motivation

The OLS β is frozen at its formation-period value (1.049) for the entire trading period. If the true economic relationship shifts, the frozen β introduces phantom signal: the spread absorbs structural drift as temporary mean-reversion, generating entries that will not revert. For CMS/DUK in 2022, U.S. utilities experienced sector rotation during the Ukraine energy shock — a legitimate structural concern.

### State-space model

```
State:       θ_t = [α_t, β_t]
Transition:  θ_t = θ_{t−1} + w_t,    w ~ N(0, Q)
Observation: log(A_t) = [1, log(B_t)] · θ_t + v_t,    v ~ N(0, R)
```

- δ = 1e-5 → slow adaptation (β half-life ≈ 180 trading days). Conservative: ensures detected drift is structural, not noise.
- The filter runs on the full year (Jan–Dec). Formation data warms it up; no re-initialization at the formation/trading boundary.

### Finding: 26% beta drift

| Metric | Value |
|--------|------:|
| Static OLS β (Jan–Jun) | 1.0487 |
| Kalman β at formation end (Jun) | 0.7629 |
| Kalman β at year end (Dec) | 0.7819 |
| Total drift vs OLS | −0.267 (−25.5%) |

The Kalman filter converged to β ≈ 0.78, 26% below the OLS estimate. Throughout the trading period, the static strategy added ~0.27 × log(DUK_t) as phantom spread to every bar.

### Why the Kalman spread is not directly tradeable

The Kalman posterior spread is approximately the scaled innovation — white noise by construction. The posterior fit removes the autocorrelation that pairs trading relies on. When standardized with the same 680-bar window, it produces 1,814 trades with excess kurtosis 13.5. Not viable as a trading signal.

### Practical use

1. **Entry/exit timing** → use static OLS Z-score (retains mean-reversion timescale)
2. **Position sizing** → use Kalman β (correctly sized exposure)
3. **Rebalance** → monthly or quarterly (structural drift without noise-fitting)

> **The LTCM parallel.** 26% drift mirrors the LTCM failure mode: when an apparently mean-reverting relationship undergoes structural change, a spread entry is a bet against a trend, not a temporary divergence. Without a stop-loss or dynamic β update, the position can diverge indefinitely.

### Why Numba is not applied to the Kalman filter

The Kalman recursion is a Python loop over ~97k bars, mixing 2×2 numpy matrix operations with Python control flow. At ~0.5–1 second per pair, the compile overhead of `@njit` would exceed the runtime for a single pair. For the 11-pair sweep, total Kalman cost is ~10 seconds — acceptable without JIT.

*See: `outputs/figures/kalman_beta_CMS_DUK.png`*

---

## Appendix B — QQ-Plot and Tail Behavior

The QQ-Plot maps trading-period Z-scores against a standard normal distribution.

**Finding:** Extreme tails diverge from the diagonal. Empirical probability of exceeding ±2.0 is ~13.9% for CMS/DUK vs 4.55% theoretical — a +8.34 pp coverage gap.

**This is not fat tails.** Excess kurtosis is −0.15 (marginally platykurtic — *thinner* tails than Gaussian). The gap is the mechanical signature of window-truncation bias. GOOG/GOOGL (W ≈ HL) has a gap of only +2.9%, confirming the mechanism.

> The engine does not assume Gaussian Z-scores. The Z-score is a standardized deviation measure; the entry threshold is calibrated empirically. The QQ-plot is a distributional health check, not a modeling assumption.

*See: `outputs/figures/all_pairs/CMS_DUK/qq_plot.png`*

---

## Appendix C — Rolling Hurst Regime Monitor

The Rolling Hurst chart tracks the stability of H < 0.5 over the trading period using 5,000-bar rolling windows.

- **CMS/DUK:** 100% of windows returned H < 0.5. Mean-reversion regime persisted without interruption.
- **DOW/LYB:** 100% of windows returned H < 0.5.

Extended periods where H > 0.5 would indicate the spread has transitioned to a trending regime, weakening the Z-score engine's core assumption. Short fluctuations crossing 0.5 are expected noise from the variance-ratio estimator.

*See: `outputs/figures/all_pairs/CMS_DUK/rolling_hurst.png`*

---

## Appendix D — OLS Direction Asymmetry

| Pair | β (A on B) | β (B on A, reciprocal) | Gap |
|------|:----------:|:----------------------:|:---:|
| CMS/DUK | 1.049 | 1.108 | 5.7% |
| DOW/LYB | 1.083 | 1.178 | 8.8% |

The chosen direction (A on B) matches the Week 1 EG scan convention. TLS via PCA would eliminate the asymmetry. The 5.7% gap for CMS/DUK has minimal practical impact. The 8.8% gap for DOW/LYB contributes to its slightly higher residual Z-score std (1.42 vs 1.34).

---

## Appendix E — All-Pairs Diagnostic Sweep (11 pairs)

| Pair | Category | EG t-stat | p-value | β | HL | W | Z std | Gap ±2σ | Trades | %MR |
|------|----------|:---------:|:-------:|-----:|----:|:---:|------:|--------:|-------:|:---:|
| CMS/DUK | Primary | −5.27 | 0.00005 | 1.05 | 680 | 680 | 1.34 | +8.3% | 111 | 100% |
| DOW/LYB | Secondary | −4.86 | 0.0003 | 1.08 | 778 | 778 | 1.42 | +7.9% | 89 | 100% |
| A/AFL | Alternative | −5.29 | 0.00005 | 0.84 | 1,777 | 1,777 | 1.42 | +9.5% | 30 | 81% |
| AVGO/GLD | Alternative | −4.35 | 0.0022 | 0.57 | 2,840 | 2,000* | 1.36 | +8.9% | 26 | 32% |
| DDOG/FOXA | Alternative | −5.73 | 0.000006 | 2.00 | 648 | 648 | 1.46 | +9.5% | 82 | 69% |
| HD/MS | Alternative | −4.56 | 0.001 | 0.93 | 1,247 | 1,247 | 1.43 | +9.7% | 40 | 70% |
| LOW/MS | Alternative | −4.18 | 0.004 | 0.89 | 2,095 | 2,000* | 1.42 | +9.3% | 27 | 73% |
| DHI/LVS | Alternative | −4.87 | 0.0003 | 0.76 | 2,236 | 2,000* | 1.41 | +8.9% | 30 | 42% |
| GOOG/GOOGL | Benchmark | −5.28 | 0.00005 | 0.99 | 20 | 20 | 1.19 | +2.9% | 1,922 | 100% |
| CVNA/ISRG | Neg. ctrl | +0.78 | 0.994 | 4.42 | 1,770 | 1,770 | 1.45 | +9.0% | 28 | 66% |
| INTC/JPM | Neg. ctrl | +0.13 | 0.989 | 0.78 | 4,469 | 2,000* | 1.32 | +11.7% | 27 | 79% |

*\* 2,000-bar cap applied (HL exceeds cap). Trade counts in this table are without warmup for cross-pair comparability.*

**Weak alternatives to exclude:** AVGO/GLD (32% MR windows) and DHI/LVS (42% MR windows) show insufficient mean-reversion persistence. 26–30 trades each — too few to assess edge.
