# Net-of-Fees Performance Report — Week 5

## 1. Executive Summary

**Verdict:** Strategy survives friction. Dynamic net Sharpe = 2.531.

- Trades evaluated: 90
- Mean dynamic RT cost: $90.88 (~45.4 bps of allocated capital)
- Sharpe (Gross): 2.822
- Sharpe (Static60bps): 1.978
- Sharpe (Dynamic): 2.531

## 2. Empirical Spread Profile

Per-tier intraday U-shape (from kill-zone Part A):

| bucket      |   Tier1_Tight |   Tier2_Medium |   Tier3_Wide |
|:------------|--------------:|---------------:|-------------:|
| 09:30-10:00 |         13.46 |          20.73 |        38.73 |
| 10:00-10:30 |          8.47 |          13.57 |        26.17 |
| 10:30-11:00 |          7.33 |          11.98 |        23.39 |
| 11:00-11:30 |          7.08 |          11.73 |        23.02 |
| 11:30-12:00 |          8.02 |          14.47 |        29.77 |
| 12:00-12:30 |          7.81 |          13.93 |        28.48 |
| 12:30-13:00 |          7.48 |          13.26 |        27.1  |
| 13:00-13:30 |          7.19 |          12.65 |        25.79 |
| 13:30-14:00 |          6.57 |          11.44 |        23.56 |
| 14:00-14:30 |          5.8  |          10.15 |        21.56 |
| 14:30-15:00 |          5.21 |           9.22 |        19.95 |
| 15:00-15:30 |          4.14 |           7.39 |        17.16 |
| 15:30-15:59 |          2.43 |           4.45 |        10.88 |

## 3. Slippage Model Spec

Three-component dynamic cost: `C_total = C_spread + C_impact + C_borrow`.

- Kappa tier distribution across 131 (fold,ticker) pairs:

|   kappa |   count |
|--------:|--------:|
|     0.3 |      39 |
|     0.5 |      85 |
|     0.8 |       7 |

Impact-prediction OLS (diagnostic, no pass/fail):

```
  alpha: 3.0692333692346736
  beta: 1.8859959065863998
  r2: 0.33417121851688847
  p_value: 0.0
  n: 1990514
  note: Diagnostic only. No pass/fail gate per workflow.
```

## 4. Before/After Table

|                    |   Gross |   Static60bps |   Dynamic |
|:-------------------|--------:|--------------:|----------:|
| Sharpe (annual)    |  2.8218 |        1.9777 |    2.5306 |
| CAGR               |  0.0992 |        0.0634 |    0.0812 |
| MaxDD (bar)        | -0.0022 |       -0.0039 |   -0.0023 |
| Calmar             | 45.1788 |       16.3671 |   35.1498 |
| Win Rate           |  0.7333 |        0.5667 |    0.6222 |
| Avg Trades / Fold  |  4.0909 |        4.0909 |    4.0909 |
| Avg RT Cost (bps)  |  0      |       90.8712 |   45.4396 |
| % Folds Profitable |  0.6818 |        0.5455 |    0.6818 |

## 5. Cost Waterfall

All values in bps of allocated capital, per trade. Cost columns are negative.

| regime     |   n_trades |   gross_bps |   spread_bps |   impact_bps |   borrow_bps |   rebalance_bps |   net_bps |
|:-----------|-----------:|------------:|-------------:|-------------:|-------------:|----------------:|----------:|
| Overall    |         90 |      260.58 |       -32.72 |       -10.79 |        -0.5  |           -1.44 |    215.15 |
| Bear 2022  |         13 |      144.25 |       -19.49 |        -7.67 |        -0.52 |           -0.48 |    116.09 |
| Bull 2023+ |         77 |      280.23 |       -34.95 |       -11.31 |        -0.49 |           -1.6  |    231.87 |

## 6. Regime-Conditional Costs

| regime                |   n_trades |   avg_l1_spread_bps |   avg_dyn_rt_cost_bps |   sharpe_gross |   sharpe_dynamic |   delta_sharpe |
|:----------------------|-----------:|--------------------:|----------------------:|---------------:|-----------------:|---------------:|
| Late Bear 2022        |         13 |               11.62 |                 28.16 |          9.542 |            7.683 |         -1.859 |
| Early Bull 2023       |         18 |               11.78 |                 23.23 |          0.827 |           -0.12  |         -0.947 |
| Mid Bull 2024         |          9 |               10.42 |                 24.41 |          8.566 |            7.348 |         -1.219 |
| Late Bull 2025-Q12026 |         50 |               10.95 |                 61.72 |          3.263 |            2.998 |         -0.265 |

## 7. Spread-Vol Correlation (Diagnostic)

```
  alpha: 3.0692333692346736
  beta: 1.8859959065863998
  r2: 0.33417121851688847
  p_value: 0.0
  n: 1990514
  note: Diagnostic only. No pass/fail gate per workflow.
```

## 8. Kill Zone + Seasonality

**Part A — intraday U-shape by tier (avg L1 spread, bps):**

| bucket      |   Tier1_Tight |   Tier2_Medium |   Tier3_Wide |
|:------------|--------------:|---------------:|-------------:|
| 09:30-10:00 |         13.46 |          20.73 |        38.73 |
| 10:00-10:30 |          8.47 |          13.57 |        26.17 |
| 10:30-11:00 |          7.33 |          11.98 |        23.39 |
| 11:00-11:30 |          7.08 |          11.73 |        23.02 |
| 11:30-12:00 |          8.02 |          14.47 |        29.77 |
| 12:00-12:30 |          7.81 |          13.93 |        28.48 |
| 12:30-13:00 |          7.48 |          13.26 |        27.1  |
| 13:00-13:30 |          7.19 |          12.65 |        25.79 |
| 13:30-14:00 |          6.57 |          11.44 |        23.56 |
| 14:00-14:30 |          5.8  |          10.15 |        21.56 |
| 14:30-15:00 |          5.21 |           9.22 |        19.95 |
| 15:00-15:30 |          4.14 |           7.39 |        17.16 |
| 15:30-15:59 |          2.43 |           4.45 |        10.88 |

**Part B — net alpha heatmap (kill_zone=True if avg net_bps < 0):**

| bucket      |   n_trades |   gross_bps |   cost_bps |   net_bps | kill_zone   |
|:------------|-----------:|------------:|-----------:|----------:|:------------|
| 09:30-10:00 |          1 |      217    |      22.14 |    194.85 | False       |
| 10:00-10:30 |         61 |      227.95 |      50.6  |    177.35 | False       |
| 10:30-11:00 |         11 |       83.23 |      21.45 |     61.79 | False       |
| 11:00-11:30 |          5 |       17.8  |      19.92 |     -2.12 | True        |
| 11:30-12:00 |          3 |      -26.24 |      13.2  |    -39.43 | True        |
| 12:00-12:30 |          1 |      122.71 |      21.39 |    101.31 | False       |
| 12:30-13:00 |        nan |      nan    |     nan    |    nan    | False       |
| 13:00-13:30 |          4 |       56.7  |      21.92 |     34.77 | False       |
| 13:30-14:00 |        nan |      nan    |     nan    |    nan    | False       |
| 14:00-14:30 |          1 |     7923.03 |     455.89 |   7467.14 | False       |
| 14:30-15:00 |        nan |      nan    |     nan    |    nan    | False       |
| 15:00-15:30 |          1 |      138.16 |      24.03 |    114.12 | False       |
| 15:30-15:59 |          2 |       -2.94 |       8.4  |    -11.34 | True        |

## 9. Negative Control (Dynamic)

|    |   n_traded_folds |   nc_pass_count_static |   nc_pass_rate_static | nc_dynamic_status   | note                                                                                                                                                                                                                                                                                       |
|---:|-----------------:|-----------------------:|----------------------:|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |               22 |                      8 |              0.363636 | DEFERRED            | Dynamic-cost NC requires re-running Week 4 NC pairs through Plan 2 hooks with synthetic NC trade timestamps. Week 4 stores only aggregate NC Sharpe, not per-trade logs. Plan 3 reports Week 4 static-cost NC results above and flags this as a Week 6 follow-up. (Option B per workflow.) |

## 10. Overfitting Diagnostics (Net)

|             |   Gross |   Static |   Dynamic |
|:------------|--------:|---------:|----------:|
| Raw Sharpe  |  2.8218 |   1.9777 |    2.5306 |
| DSR p-value |  0.9306 |   0.7981 |    0.912  |
| PBO         |  0.1231 |   0.1231 |    0.1231 |

## 11. Sensitivity Analysis

|    |   kappa_mult |   borrow_bps | spread_level   |   net_sharpe |   delta_vs_baseline |
|---:|-------------:|-------------:|:---------------|-------------:|--------------------:|
|  0 |          0.5 |           30 | L1             |       2.5736 |              0.0429 |
|  1 |          0.5 |           50 | L1             |       2.5711 |              0.0405 |
|  2 |          0.5 |          100 | L1             |       2.5649 |              0.0343 |
|  3 |          1   |           30 | L1             |       2.5331 |              0.0025 |
|  4 |          1   |           50 | L1             |       2.5306 |              0      |
|  5 |          1   |          100 | L1             |       2.5244 |             -0.0063 |
|  6 |          1.5 |           30 | L1             |       2.4919 |             -0.0387 |
|  7 |          1.5 |           50 | L1             |       2.4894 |             -0.0413 |
|  8 |          1.5 |          100 | L1             |       2.483  |             -0.0476 |
|  9 |        nan   |          nan | L2             |     nan      |            nan      |

## 12. Verdict

**Strategy survives friction. Dynamic net Sharpe = 2.531.**

Red flag triggers:

- `dynamic_cost_blowup`: False
- `cost_exceeds_alpha`:  False
- `kappa_instability`:   False (drift_count=1)
- `dsr_degradation`:     False
- `math_violation`:      False

All three regimes (Gross / Static / Dynamic) shown side-by-side. No cherry-picking.