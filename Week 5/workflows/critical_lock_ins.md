## Critical Lock-Ins

21. **Dynamic slippage replaces static 60 bps** as primary cost model; static 60 bps retained as comparison anchor
22. **Three-component cost:** half-spread + volatility-scaled impact + borrow
23. **κ is pre-specified by liquidity tier** using formation-window median full L1 spread. L2−L1 widening is used only as validation, not for performance fitting.
24. **Spread data is real-time observable** — no forward-looking spread statistics in execution
25. **Symmetric LOB sizes acknowledged as limitation** — no order-flow imbalance signals derived
26. **All three cost regimes reported side-by-side** — no cherry-picking
27. **Net Sharpe is the decision metric** — gross Sharpe is informational only
28. **Strategy unchanged** — Week 5 measures friction only, does not modify signals/sizing/gates
29. **NC validation under dynamic costs** — proves cost model is directionally neutral
30. **DSR/PBO recomputed on net returns** — statistical significance must survive friction

---
