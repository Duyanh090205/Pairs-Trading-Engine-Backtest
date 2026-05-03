# Pipeline Week 5: The Engine — The "Microstructure" Reality

## 0. Objective

**Theme:** The Friction.

**Deliverable:** A "Net-of-Fees Performance Report."

**Core Question:** The Week 4 strategy showed profitability under a flat 60 bps cost assumption. When we replace that with empirical bid-ask spreads that widen during spread instability, does alpha survive — or was it an illusion eaten by friction?

**Scope Boundary:** This week **measures** friction on the existing strategy. We do NOT modify the strategy itself (no sizing changes, no new entry/exit gates, no regime suppression). Same signals, same positions — different cost model.

**Data:** `orderbook.parquet` — 213M rows, 504–526 tickers, 1-min frequency, 2022-01-03 to 2026-03-19. 3-level limit order book (L1/L2/L3 bid/ask price + size).

---

## 0.1 Data Profile

| Property | Value |
|---|---|
| Total rows | 212,975,144 |
| Frequency | 1-minute (dominant; some 2–5 min gaps) |
| Tickers | 504 (2022-01) → 526 (2026-03) |
| Columns | `timestamp`, `ticker`, `l{1,2,3}_bid_px`, `l{1,2,3}_bid_sz`, `l{1,2,3}_ask_px`, `l{1,2,3}_ask_sz` |
| NaN count | 0 across all columns |
| Date range | 2022-01-03 09:00 → 2026-03-19 23:59 |

**L1 Spread (bps):** Universe median ~10; SPY ~4.7; VTRS ~32; max observed >150 (stress prints).

**Depth:** L2 spread ~20 bps (1.5× L1 size), L3 ~43 bps (2.5× L1 size).

**Known Limitation:** `bid_sz == ask_sz` at all levels (symmetric/synthetic LOB). No order-flow imbalance signals can be derived. Size data is treated as representative liquidity at each price level only.

---
