## Phase 5 — Red Flag Triggers

Inherit all Week 4 §3.7 red flags. Add:

| Trigger | Condition | Action |
|---|---|---|
| Dynamic cost blow-up | Mean RT cost > 150 bps any fold | Audit spread data alignment, check for stale quotes |
| Cost > Gross Alpha | Net Sharpe < 0 AND Gross Sharpe > 1.0 | Flag: alpha real but untradeable at this frequency |
| Spread data gap | > 5% missing spread observations in trading window | Fall back to static 60 bps for affected bars |
| κ instability | κ tier changes > 2× across consecutive folds for same ticker | Flag: liquidity regime shift |
| NC cost model leak | NC Sharpe under dynamic model > 0.5 | Audit cost model for sign errors or look-ahead |
| DSR degradation | DSR_net p-value > 0.10 while DSR_gross p-value < 0.05 | Flag: alpha not statistically significant after costs |
| Math violation | `total_cost_dollars < 0` OR `net_pnl_dollars > gross_pnl_dollars` | Throw Error: Cost model must strictly subtract PnL |

---
