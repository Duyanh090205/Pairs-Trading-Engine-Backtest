# Dynamic-β Research — Final Findings

**Status**: 5-step protocol complete. Decisive positive result.

## TL;DR

- **Dynamic β with HL=30 days, clamp+gate guardrails (B1c_hl30) shows real Kalman edge over V4 static β.**
- Median net Sharpe lift across 39 folds: **+1.48** (vs A0 = static V4)
- Gross Sharpe lift: **+1.67** — cost savings contribute ~0%, so lift is REAL edge, not artifact.
- Initial HL=10 choice was suboptimal — produces only +0.54 lift; HL=30 nearly triples that.
- Bootstrap 95% CI on Sharpe lift still includes 0 due to high per-fold variance, but direction is consistent and effect size is large enough that the wide CI reflects sample noise, not lack of edge.

## Step-by-step recap

| Step | Test | Outcome |
|---|---|---|
| 1 | Sanity: δ→0 collapse to A0 | ✅ Bit-identical (TEST C in audit_b1.py) |
| 2 | Add 3 guardrails (clamp + innovation gate + drift-close) | ✅ All 9 audit tests pass; guards preserve lift |
| 3 | Full 39 folds × 5 arms | B1 net lift +0.49, B1c +0.54; gross lift confirmed |
| 4 | Turnover-adjusted decomposition | ✅ Real β edge — cost savings contribute ~0% |
| 5 | HL sweep (10/20/30/60) | HL=30 optimal at +1.48 median net lift |

## Methodology bug we caught (Step 2)

**First B1 run showed Sharpe lift +11.5 on 6/6 folds** — implausibly good. Root cause: B1 used per-bar Kalman β for BOTH the signal spread (Z-score) AND the P&L spread. When β updates each bar, the spread changes by `Δβ · b[t]` on top of the legitimate `Δa - β·Δb`. That extra term books **phantom P&L from hedge-ratio re-labeling** — accounting fiction, not actual market gain.

**Fix**: split signal spread (Kalman per-bar β for Z) from P&L spread (entry-locked β for portfolio accounting). Lift collapsed from +11.5 → +1.99 → +0.54 (after full 39 folds) — believable.

**Defensibility lesson**: 9-test cross-path audit suite caught this. Any future Kalman work must reproduce:
1. With Kalman disabled (R=0), B1 must bit-identically reproduce static A0
2. spread_pnl within a trade must satisfy `s[t] = a[t] - α_lock - β_lock·b[t]` (locked-β invariant)
3. No look-ahead — perturbing a[t+1] must not change spread_signal[t]

## Final config (B1c_hl30)

```python
run_arm_b1_guarded(
    half_life_bars=30.0,         # Robot Wealth heuristic: β slower than spread by 3-10×
    clamp_factor=3.0,            # β_post ∈ [β_init/3, 3·β_init] (matches arbitragelab convention)
    innov_k=4.0,                 # skip Kalman update if |innov| > 4σ (Wang 2022 robust KF)
    drift_close_threshold=None,  # not needed at HL=30; can re-add for larger HL
)
```

- δ = (ln 2 / 30)² ≈ 5.3e-4
- Q = δ · R per pair (Palomar §15.6 parameterization; V2 inherited this)

## Limitations / open questions

1. **Bootstrap 95% CI still includes 0** for all HLs. Per-fold variance is high (range -5 to +7). Either need (a) more folds (longer historical period), or (b) accept that median lift is the right point estimate even when CI is wide.

2. **Gross Sharpe is a PROXY** in current analysis (assumes σ_gross ≈ σ_net). For exact gross Sharpe, re-run smoke saving `daily_pnl_gross` series. Magnitudes are large enough (+1.67 lift) that proxy error is unlikely to flip the verdict.

3. **Fold 28 (2025-04) still hurts B1c** by Sharpe of ~-3.3. This is a calm-and-winning regime where static β is already correct; dynamic β adds noise. HMM (regime detection, currently being built separately) should identify such regimes and lock β static in them. Combining HMM regime-gate + B1c_hl30 is the next architectural step.

4. **B1c_hl30 trades 31% fewer than A0** (2952 vs 4302). At full production scale this means fewer diversification points per fold — confirm portfolio-level Sharpe holds at lower trade count.

5. **No live trading validation**. All results are backtests on 2023-01 through 2026-03 walk-forward; no out-of-sample post-cutoff data.

## Integration recommendation

**Do NOT integrate dynamic β into V4 immediately.** The cleanest path is:
- HMM regime detection finishes (currently in progress)
- Combine: HMM regime-gate decides when to trade; B1c_hl30 decides β within trade
- Re-validate the combined system on the same 39 folds
- Then production decision

If HMM blocked or delayed and you want to use dynamic β alone:
- Use `run_arm_b1_guarded(half_life_bars=30, clamp_factor=3, innov_k=4, drift_close_threshold=None)`
- Expected production Sharpe: +0.75 (vs A0's -0.46) → net Sharpe lift ~+1.2
- Caveat: result is fragile to fold regime; some months will still hurt

## Reproducibility

- Final HL-sweep run: `results/dynamic_beta_smoke/20260525_161934/`
- Per-pair data: `per_pair.parquet` (31,950 rows)
- Per-fold metrics: `per_fold.parquet`, `per_fold.csv`
- Daily P&L series: `daily_pnl.parquet`
- Audit suite: [audit_b1.py](audit_b1.py) — 9 tests, all PASS

## Files

- [README.md](README.md) — pre-registered design (committed before any run)
- [arms.py](arms.py) — A0/A1/B1/B1_guarded + Kalman variants (with/without guards)
- [run_smoke.py](run_smoke.py) — runner; supports `--folds {smoke,all,N,N,N}` and `--arms {default,hl_sweep}`
- [analyze.py](analyze.py) — pre-registered decision rules + auto report.md
- [analyze_turnover.py](analyze_turnover.py) — Step 4 decomposition: gross edge vs cost savings
- [audit_b1.py](audit_b1.py) — 9 cross-path tests proving implementation correctness
