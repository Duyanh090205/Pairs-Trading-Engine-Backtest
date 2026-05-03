## Extensions (Run If Time Permits)

These are valuable but not required for the core deliverable. Each is self-contained and does not depend on the others.

### EXT-1 — Execution Latency Stress Test with Real Spread Cost

Extends Week 4 §3.6. The latency sweep {t+1, t+2, t+5, t+10} now charges the **actual spread at the delayed execution bar**, not the spread at the signal bar:

```
for lag in [1, 2, 5, 10]:
    t_exec = t_signal + lag
    fill_price    = price(t_exec)                          # already in Week 4
    fill_cost_bps = C_spread(t_exec) + C_impact(t_exec)    # NEW: cost at execution bar

# Output: alpha decay curve with two y-axes:
#   y1 = Sharpe (net of dynamic cost at execution bar)
#   y2 = Avg cost at execution bar vs cost at signal bar
```

**Key question:** Does latency increase cost as well as decrease alpha? If spreads autocorrelate (wide persists), latency compounds both alpha loss and cost increase.

**Pass criterion:** Net Sharpe at t+5 > 0 AND cost at t+5 < 1.5× cost at t+1.

### EXT-2 — Cross-Leg Spread Correlation

For each surviving pair (A, B), compute:

```
corr_spread_AB = corr(spread_l1_bps_A, spread_l1_bps_B)    # per fold, on trading window
```

**Report:**
- Distribution of `corr_spread_AB` across surviving pairs
- Mean correlation per regime partition
- Conditional cost: `avg RT cost when corr > 0.7` vs `avg RT cost when corr < 0.3`

**Implication:** High cross-leg correlation means both legs widen simultaneously → RT cost compounds beyond the sum of individual leg costs. This is a portfolio-level microstructure risk invisible in per-leg analysis.

### EXT-3 — Book-Walk Model (Position Size > L1 Depth)

When the trade's share count exceeds L1 available size, the fill walks into deeper levels:

```
order_shares = position_notional / mid_px

if order_shares <= l1_bid_sz:
    fill_px = l1_ask_px
    effective_spread_bps = spread_l1_bps

elif order_shares <= l1_bid_sz + l2_bid_sz:
    filled_l1 = l1_bid_sz
    filled_l2 = order_shares - l1_bid_sz
    fill_px   = (filled_l1 × l1_ask_px + filled_l2 × l2_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px × 10,000 × 2

else:    # walks to L3
    filled_l1 = l1_bid_sz
    filled_l2 = l2_bid_sz
    filled_l3 = order_shares - l1_bid_sz - l2_bid_sz
    fill_px   = (filled_l1 × l1_ask_px + filled_l2 × l2_ask_px + filled_l3 × l3_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px × 10,000 × 2
```

**Integration:** When book-walk triggers, `C_spread(t)` in §2.1 is replaced by `0.5 × effective_spread_bps`. The impact component still applies on top.

**Diagnostic:** Track `spread_leakage = effective_spread - quoted_spread`. If consistently large → position sizing exceeds available depth.

**Note:** Book-walk results are diagnostic only because LOB sizes are symmetric/synthetic. They should not be used as the primary cost model.

---
