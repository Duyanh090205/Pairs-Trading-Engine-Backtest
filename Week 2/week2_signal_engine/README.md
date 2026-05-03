# Week 2 Z-Score Signal Engine

## Purpose
This project is a vectorized Z-score signal engine for pairs trading on 1-minute stock data. It handles:
- Data loading and alignment
- Formation/trading period splitting
- OLS hedge ratio estimation (formation only)
- Spread and rolling Z-score computation
- Stateful entry/exit position tracking
- Comprehensive diagnostics and visualizations

**IMPORTANT NOTE:**
This is Week 2 of the quant program and is strictly for signal generation and trade timing validation. It is *not* a profitability backtest. We are using candidate pairs and near-misses from Week 1 to validate the engine's behavior, not to make claims about validated, tradable pairs. Transaction costs, stop-losses, and overnight gap handling are deferred to Week 3.

## Workflow
1. Configure run parameters in `configs/`.
2. Execute the full end-to-end pipeline via `scripts/run_all_pairs_diagnostics.py` or individually via `src/pipeline/run_week2.py`.
3. Output tables, figures, and tracking logs will be generated in `outputs/`.
4. Detailed rules, diagnostics, and Kalman-filter logic are documented in `docs/methodology_summary.md`.
