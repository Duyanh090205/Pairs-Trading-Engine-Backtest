# Week 3 Workflow Checkpoint

This checklist reflects the methodology outlined in `Week3_Final_Plan_v3.md` and should be used to track progress through Deliverables 1 & 2.

## Deliverable 1: Creating "Bad Data"
- [ ] **CP1: Confirm Clean Original Data**
  - [ ] Convert `window_start` nanoseconds to datetime ET (DST-aware)
  - [ ] Assert timestamps are strictly monotonic per ticker
  - [ ] Assert no duplicate `(ticker, window_start)` records
  - [ ] Assert OHLC logically valid (`high >= close >= low`, `high >= open >= low`)
  - [ ] Assert no NaNs in OHLC
  - [ ] Filter session to valid trading hours (09:30–15:59 ET)
  - [ ] Log outputs: total files, total tickers, time range, and assertion results
- [ ] **CP2: 4 Biased Ideas & Write-up**
  - [ ] Define Look-ahead bias and its realistic implications
  - [ ] H1: Random future-close substitution (Vendor buffer delay proxy)
  - [ ] H2: Timestamp backdating by 60s (Vendor open vs close timestamp proxy)
  - [ ] H3: Spread-level injection (Direct alpha leakage into derived column)
  - [ ] H4: Full-dataset normalization leak (Most common ML preprocessing error using full 2022 mean/std)
- [ ] **CP3: Implement 4 Flawed Datasets + Sweep**
  - [ ] Setup script matching 4 methods × 5 k-values (10%, 20%, 30%, 40%, 50%)
  - [ ] Set global random seed = 42 for reproducibility
  - [ ] Validate assertions for all 4 methods (e.g. H1 close matches future close exactly)
  - [ ] Save 20 explicit CSVs using convention `flawed_h1_k10.csv`

## Deliverable 2: Verified Backtest Engine
- [ ] **CP4: Data Pipeline Standardization**
  - [ ] Vectorized data loader: concat all sources, fix timestamps, sort
  - [ ] Output structural checks for integrity over the whole dataset
- [ ] **CP5: Signal Engine & Sizing**
  - [ ] Signal Engine Logic (OLS α=−0.6956, β=1.0487, spread computation) 
  - [ ] Z-score implementation (680 rolling, ddof=1, warmup=30 bars/day, burn-in=340)
  - [ ] Handling constraints: Executed at `t+1`, NaN-handling keeps prior pos
  - [ ] **Version A Sizing:** Cấp 1 (OLS constant β=1.0487)
  - [ ] **Version B Sizing:** Cấp 3 (Kalman β monthly rebalance updating at month ends)
  - [ ] **PnL & Costs:** Daily Mark-To-Market equity curve, Split 30/30bps cost format
  - [ ] Perform Timestamp Verification pass (assert `exec_ts > signal_ts` always)
- [ ] **CP6: Performance Benchmarking & Sweeps**
  - [ ] CP6a: Populate Sharpe sensitivity table across 20 flawed datasets
  - [ ] CP6b: Run engine-level execution lag comparative test
  - [ ] CP6c: Net metrics extraction (Version A vs Version B impact analysis)
  - [ ] CP6d: Threshold parameter sweep (Z=2.0 vs Z=2.57)
  - [ ] CP6e: Negative control execution (CVNA/ISRG vs INTC/JPM)
- [ ] **CP7: Final Verified Backtest Log**
  - [ ] Header formatting
  - [ ] Timestamp pass statement block
  - [ ] Trade log export (20-30 rows sample)
  - [ ] 2D Sharpe table documentation
  - [ ] Summary metrics table (A vs B)
  - [ ] Comparative analysis and Red Flag review notes
  - [ ] Final audit trail verification string match

## Red Flags / Validation Warnings
- [ ] Sharpe Ratio > 5 on 'Clean' Dataset (Failure if True)
- [ ] Sharpe value not cleanly monotone across increasing `k%` range for H1/H3
- [ ] Kalman β (Version B) underperforms Version A on clean dataset
- [ ] Any trade executed at `t` rather than strictly `t+1`
- [ ] CVNA/ISRG Sharpe > 0 with significant edge (Invalidates pairs signal logic)
