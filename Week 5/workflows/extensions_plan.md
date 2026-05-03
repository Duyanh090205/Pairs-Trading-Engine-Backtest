# Extensions Plan (Run if Time Permits)

> [!NOTE]
> Extensions are self-contained and do not affect the core deliverable. They are run AFTER Plan 3 is approved and the report is assembled. No code is built until Plan 3 is complete.

## Scope
1. EXT-1: Latency Stress Test with Real Spread Cost
2. EXT-3: Book-Walk Model (Position Size > L1 Depth)

*(EXT-2: Cross-Leg Spread Correlation is deprioritized)*

---

## EXT-1 — Execution Latency Stress Test

**Objective:** Extend Week 4 latency sweep by charging the actual spread at the delayed execution bar, not the signal bar.

**Implementation (`src/plan3_validation/ext1_latency.py`):**
```python
for lag in [1, 2, 5, 10]:
    t_exec = t_signal + lag
    fill_price    = price(t_exec)                          # already in Week 4
    fill_cost_bps = C_spread(t_exec) + C_impact(t_exec)    # NEW: cost at execution bar

# Output: alpha decay curve with two y-axes:
#   y1 = Sharpe (net of dynamic cost at execution bar)
#   y2 = Avg cost at execution bar vs cost at signal bar
```

**Pass Criterion:**
- Net Sharpe at t+5 > 0 AND cost at t+5 < 1.5× cost at t+1

---

## EXT-3 — Book-Walk Model

**Objective:** When the trade's share count exceeds L1 available size, calculate effective spread by walking the book into deeper levels. Diagnostic only.

**Implementation (`src/plan3_validation/ext3_book_walk.py`):**
```python
order_shares = position_notional / mid_px

if order_shares <= l1_bid_sz:
    fill_px = l1_ask_px
    effective_spread_bps = spread_l1_bps

elif order_shares <= l1_bid_sz + l2_bid_sz:
    filled_l1 = l1_bid_sz
    filled_l2 = order_shares - l1_bid_sz
    fill_px   = (filled_l1 * l1_ask_px + filled_l2 * l2_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px * 10000 * 2

else:    # walks to L3
    filled_l1 = l1_bid_sz
    filled_l2 = l2_bid_sz
    filled_l3 = order_shares - l1_bid_sz - l2_bid_sz
    fill_px   = (filled_l1 * l1_ask_px + filled_l2 * l2_ask_px + filled_l3 * l3_ask_px) / order_shares
    effective_spread_bps = (fill_px - mid_px) / mid_px * 10000 * 2
```

**Diagnostic Tracking:**
- Track `spread_leakage = effective_spread - quoted_spread`. If consistently large, position sizing exceeds available depth.
