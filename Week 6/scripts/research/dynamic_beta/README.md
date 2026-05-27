# Dynamic-β Smoke Test (Pre-registered Design)

**Status**: pre-registered design, committed BEFORE running. Decision rules below are binding.

## Question

Does dynamic hedge-ratio estimation beat V4's static β at daily frequency, on US equities, post-cost, on this specific V4 setup?

Literature gap: no rigorous head-to-head study exists for this exact setup (see [../../../documents/](../../../documents/) for the research survey). The smoke test is go/no-go: if the cheapest dynamic-β variant cannot beat static at all, the literature gap is empirical, not just published.

## Architectural constraints (do NOT modify in this test)

- **Same fold schedule**: monthly walk-forward, 12-month formation, 1-month trading
- **Same discovery output**: V4 `discovery_daily.run` (Johansen, BH-FDR, |β|≤5 cap, HL∈[5,30])
- **Same downstream**: same Z-window (60d), entry/exit (|Z|≥2 / |Z|≥4), vol-target sizing, alpha refit (60-day), dynamic cost engine, borrow accrual
- **Pair identity is locked to A0**: all three arms operate on the *same pair list* per fold (V4 discovery output). Only β source differs. This is the central defensibility lever — otherwise different arms screen different pairs and comparison is apples-to-oranges.

## Arms

| Arm | β estimator | When β updates | δ / window |
|---|---|---|---|
| **A0** | V4 baseline: Johansen PCA hedge ratio on 12-month formation residuals | At fold boundary only (monthly), constant within trading window | 252 daily bars |
| **A1** | OLS on last 60 days of formation residuals | At fold boundary only (monthly), constant within trading window | 60 daily bars |
| **B1** | A0 β as initial state; Kalman update per bar within trading window | Daily during trading window | HL = 10 days → δ = (ln 2/10)² ≈ 4.8e-3 with Q = δ·R |

Notes:
- A1 isolates the "estimation horizon" hypothesis (short window catches regime change faster).
- B1 isolates the "within-window adaptation" hypothesis (Kalman per-bar).
- All three use the same `R_measurement_noise` from V4 discovery (variance of formation OLS residuals).
- B1 posterior β is *unclamped* in smoke. If smoke says B1 wins, full test re-runs with `posterior β ∈ [0.3·β_form, 3·β_form]` clamp and innovation gating; smoke without clamps tells us the *upper bound* of dynamic-β value.

## Folds (6 selected, quantitatively diverse on V4 outcomes)

Selection from `results/v4/final_dynamic_cost/fold_metrics.csv`:

| Fold | Month | V4 Sharpe | V4 return | Why selected |
|---|---|---|---|---|
| 4 | 2023-04 | +4.55 | +0.85% | Calm winning month — sanity baseline |
| 5 | 2023-05 | -1.09 | -0.64% | 35 pairs |β|>5 in pre-cap audit → β-noise stress test |
| 18 | 2024-06 | -5.93 | -4.08% | Quiet-month loss — failure mode (regime mismatch?) |
| 22 | 2024-10 | -3.85 | -1.68% | 112 trades → high turnover stress |
| 28 | 2025-04 | +4.54 | +2.95% | Late-period winning month — sanity that dynamic doesn't break good periods |
| 32 | 2025-08 | -10.54 | -16.2% | Extreme vol/loss → if dynamic β helps anywhere, it should be here |

Coverage: 2 winners, 4 losers (biased toward losers because *that's where dynamic β has the most room to lift*). Years 2023, 2024, 2025 all represented.

**Bias acknowledgment**: fold selection used V4 outcomes, so selection is informed by results. This is a smoke test, not the final word — selection bias means a positive smoke triggers full 39-fold test, not declaration of victory.

## Pre-registered decision rules

A1 or B1 is declared a **smoke-win** vs A0 iff **both** primary criteria pass:

**P1. Median per-fold Sharpe lift ≥ +0.20** (paired comparison: per-fold Sharpe_arm − Sharpe_A0)

**P2. Arm wins in ≥ 4 of 6 folds** (per-fold sign — Sharpe_arm > Sharpe_A0)

Secondary diagnostic (informative, not gating):

**S1.** 95% block-bootstrap CI on aggregate Sharpe-difference excludes 0 (1000 resamples, folds as blocks)

**S2.** Sign test p-value ≤ 0.20 (one-sided, H1: arm > A0). With n=6 the test is underpowered, this is informative only.

If neither A1 nor B1 wins → **conclusion: static β is sufficient at V4's setup. Do not invest in dynamic β. Update memory.**

If a win is declared → **proceed to full 39-fold test with guardrails (clamp + innovation gate) before integration**.

## Sanity invariants (any violation = test invalid, NOT result)

- **I1**. n_pairs(A0) == n_pairs(A1) == n_pairs(B1) per fold (same discovery output)
- **I2**. β_A1 and β_B1 finite for all pairs (no NaN propagation)
- **I3**. B1 posterior β stays within [-20, 20] for all pairs / all bars (if violated, log + flag; outside cap range but still finite is allowed in smoke since arm has no clamp)
- **I4**. No look-ahead: B1 spread at bar t uses β posterior from bar t-1 (or t prior in same-bar convention — matches V2 kalman.py which uses prior spread)
- **I5**. All three arms produce identical entries when β is identical (sanity check on at least 1 pair where formation OLS β ≈ Johansen β)

## Output

- `results/<timestamp>/per_pair.parquet` — every (fold, pair, arm) result row
- `results/<timestamp>/per_fold.parquet` — aggregated per-arm-per-fold Sharpe, return, n_trades, β-stats
- `results/<timestamp>/report.md` — pre-registered decisions vs actual outcomes
- `results/<timestamp>/run.log` — full execution log with seed, parameters, timing

## Reproducibility

- Single-threaded BLAS (same as V4 pipeline)
- Discovery uses V4's existing `discovery_daily.run` — deterministic
- Kalman uses fixed init from formation-fit (no random init)
- All parameters logged to `run.log` with git SHA

## Out of scope (deferred to full test if smoke positive)

- Posterior β clamping
- Innovation outlier gating (Wang 2022 robust KF)
- δ calibration via EM/MLE or grid-search
- TLS (Total Least Squares) β estimation instead of OLS for A1
- β half-life sweep (this smoke tests one HL=10d for B1, one window=60d for A1)
- Carry-forward interaction (V4's E2 carry-forward is left at default OFF for clarity)
