# Week 3 Implementation Planning

- **Cascade ID**: `5380efcc-ac33-4375-b19b-57f86da6ec27`
- **Steps**: 56
- **Status**: CASCADE_RUN_STATUS_IDLE
- **Created**: 2026-04-20T02:53:46.058751200Z
- **Last Modified**: 2026-04-20T03:09:04.441707200Z
- **Last User Input**: 2026-04-20T03:08:34.115757500Z
- **Workspace**: file:///d:/Quant%20Finance/Quant%20Program/Week%203
- **Exported**: 2026-04-20 22:58:17

---

## 🧑 User  `2026-04-20T02:53:46`
Read the @[# Week 3 — Final Implementation Plan v3] carefully, they include all the steps and checkpoints also methodology needed for this week projects. Please write an implementation plan include new folder structure, data needed and also methodology follow the plan provided strictly. Also create a workflow .md file as checkpoint for later work checking

### 🔧 Tool: `view_file`  `2026-04-20T02:53:51`
`file:///d:/Quant%20Finance/Quant%20Program/Week%203/Week3_Final_Plan_v3.md` (546 lines, 22721 bytes)

### 🔧 Tool: `code_edit`  `2026-04-20T02:54:56`
[Code Edit] file:///C:/Users/nguye/.gemini/antigravity/brain/5380efcc-ac33-4375-b19b-57f86da6ec27/implementation_plan.md
Created Week 3 implementation plan artifact based on the final plan v3.

```diff
+# Week 3: Flawed Data Generation and Verified Backtest Engine
+
+Based on the `Week3_Final_Plan_v3.md` document, this plan details the folder structure, data requirements, and step-by-step methodology for executing Deliverable 1 (Bad Data creation) and Deliverable 2 (Verified Backtest Engine).
+
+## User Review Required
+
+> [!IMPORTANT]
+> Please review the proposed **folder structure** below to ensure it aligns with your project workspace preferences.
+> Also, confirm if the raw `.csv` data for the **entire year 2022** (Jan-Dec) is readily available in the workspace, particularly for the normalization leak (H4) which requires both formation (Jan-Jun) and trading (Jul-Dec) data.
+
+## Proposed Folder Structure
+
+```text
+d:\Quant Finance\Quant Program\Week 3\
+├── data/
+│   ├── raw/                 # Original 1-minute CSV files ({TICKER}_{DATE}.csv)
+│   ├── clean/               # Validated, session-filtered, concatenated clean data
+│   └── flawed/              # 20 internally generated flawed datasets (H1-H4 x 5 k-levels)
+├── notebooks/
+│   ├── 01_Data_Verification_and_Flawed_Injection.ipynb  # CP1, CP2, CP3
+│   └── 02_Verified_Backtest_Sweeps.ipynb                # CP4, CP5, CP6a-6e, CP7
+├── scripts/
+│   ├── engine.py            # Vectorized state machine, PnL, Sizing logic
+│   └── utils.py             # Data loading, verifying and session filtering
+├── Week3_Workflow_Checkpoint.md  # Workflow checklist (already created)
+└── logs/
+    └── Verified_Backtest_Log.txt # CP7 output log
+```
+
+## Data Needed
+
+- **Tickers:** CMS, DUK, DOW, LYB, CVNA, ISRG, INTC, JPM.
+- **Timeframe:** Required trading period is Jul–Dec 2022. Entire 2022 (Jan-Dec) is required for the normalization bias (H4) implementation.
+- **Columns:** `ticker, volume, open, close, high, low, window_start, transactions`.
+- **Formats:** Data timestamps (`window_start`) are in Unix nanoseconds and must be converted to datetime ET (DST-aware).
+- **Session:** Valid data limited to 09:30–15:59 ET.
+
+## Methodology
+
+The implementation follows the exact checkpoints specified in the plan strictly:
+
+### Deliverable 1: Creating the "Bad Data"
+- **CP1 (Data Verification):** Load clean data, confirm monotonic timestamps, no `(ticker, ts)` duplicates, validate OHLC relationships (`high >= close >= low`, etc.), check nan values, and filter valid session hours. Log verification outputs.
+- **CP2 (Methodology Write-up):** The four biases are:
+  - H1: Random future-close substitution.
+  - H2: Timestamp backdating by 60 seconds (1 bar).
+  - H3: Spread-level injection (computing clean spread and injecting future spread).
+  - H4: Full-dataset normalization leak (normalizing close price with whole year's mean/std).
+- **CP3 (Flawed Datasets Generation):** Sweep 5 `k%` values (10%, 20%, 30%, 40%, 50%) for each of the 4 methods using random seed = 42. Generate and save 20 datasets (`flawed_h1_k10.csv`, etc.). Validate specific assertions for each method before saving.
+
+### Deliverable 2: Verified Backtest Engine
+- **CP4 (Data Pipeline):** Build robust pipeline to load, concat, timestamp convert, sort, verify, and session filter dataset efficiently.
+- **CP5 (Signal & PnL Engine):**
+  - Use custom vectorized engine (pandas + numpy + Numba state machine).
+  - Z-score Entry/Exit (default Z=2.0) on rolling 680-bar window.
+  - Implement **Version A (OLS Cấp 1)** sizing with constant β = 1.0487.
+  - Implement **Version B (Kalman Cấp 3)** sizing with monthly rebalanced β.
+  - Daily mark-to-market PnL engine with 30bps entry + 30bps exit transaction costs.
+  - Timestamp verification (`exec_ts > signal_ts`).
+- **CP6a-e (Sensitivities):** 
+  - CP6a: Sharpe sensitivity table for 20 flawed datasets + 1 clean.
+  - CP6b: Engine-level comparative injection (execution lag).
+  - CP6c: Net metrics comparing Version A vs Version B sizing.
+  - CP6d: Threshold sensitivity comparing Z=2.0 vs Z=2.57.
+  - CP6e: Negative controls evaluation.
+- **CP7 (Verified Backtest Log):** Generate the comprehensive 7-part text report with findings, audit trails, and explicit warnings about testing layers.
+
+## Open Questions
+
+> [!NOTE]
+> 1. Should I begin implementing Deliverable 1 (CP1-CP3) scripts right away, or generate the complete skeleton and folder structure together upon approval?
+> 2. Are the `raw` data files already loaded in a specific subdirectory, or should I write a script to download/generate them first?
+
+## Verification Plan
+
+- Once executing, run assertions built into CP1 and CP3 to ensure data mutations strictly follow the H1-H4 specs without unintended side effects.
+- Complete execution of CP4-CP6e and generate CP7 log by dynamically checking values against the listed red flags (e.g., if Sharpe > 5.0 on clean dataset, it triggers an explicit failure).
```

### 🔧 Tool: `code_edit`  `2026-04-20T02:55:10`
[Code Edit] file:///d:/Quant%20Finance/Quant%20Program/Week%203/Week3_Workflow_Checkpoint.md
Created Week 3 Workflow Checkpoint markdown file as requested.

```diff
+# Week 3 Workflow Checkpoint
+
+This checklist reflects the methodology outlined in `Week3_Final_Plan_v3.md` and should be used to track progress through Deliverables 1 & 2.
+
+## Deliverable 1: Creating "Bad Data"
+- [ ] **CP1: Confirm Clean Original Data**
+  - [ ] Convert `window_start` nanoseconds to datetime ET (DST-aware)
+  - [ ] Assert timestamps are strictly monotonic per ticker
+  - [ ] Assert no duplicate `(ticker, window_start)` records
+  - [ ] Assert OHLC logically valid (`high >= close >= low`, `high >= open >= low`)
+  - [ ] Assert no NaNs in OHLC
+  - [ ] Filter session to valid trading hours (09:30–15:59 ET)
+  - [ ] Log outputs: total files, total tickers, time range, and assertion results
+- [ ] **CP2: 4 Biased Ideas & Write-up**
+  - [ ] Define Look-ahead bias and its realistic implications
+  - [ ] H1: Random future-close substitution (Vendor buffer delay proxy)
+  - [ ] H2: Timestamp backdating by 60s (Vendor open vs close timestamp proxy)
+  - [ ] H3: Spread-level injection (Direct alpha leakage into derived column)
+  - [ ] H4: Full-dataset normalization leak (Most common ML preprocessing error using full 2022 mean/std)
+- [ ] **CP3: Implement 4 Flawed Datasets + Sweep**
+  - [ ] Setup script matching 4 methods × 5 k-values (10%, 20%, 30%, 40%, 50%)
+  - [ ] Set global random seed = 42 for reproducibility
+  - [ ] Validate assertions for all 4 methods (e.g. H1 close matches future close exactly)
+  - [ ] Save 20 explicit CSVs using convention `flawed_h1_k10.csv`
+
+## Deliverable 2: Verified Backtest Engine
+- [ ] **CP4: Data Pipeline Standardization**
+  - [ ] Vectorized data loader: concat all sources, fix timestamps, sort
+  - [ ] Output structural checks for integrity over the whole dataset
+- [ ] **CP5: Signal Engine & Sizing**
+  - [ ] Signal Engine Logic (OLS α=−0.6956, β=1.0487, spread computation) 
+  - [ ] Z-score implementation (680 rolling, ddof=1, warmup=30 bars/day, burn-in=340)
+  - [ ] Handling constraints: Executed at `t+1`, NaN-handling keeps prior pos
+  - [ ] **Version A Sizing:** Cấp 1 (OLS constant β=1.0487)
+  - [ ] **Version B Sizing:** Cấp 3 (Kalman β monthly rebalance updating at month ends)
+  - [ ] **PnL & Costs:** Daily Mark-To-Market equity curve, Split 30/30bps cost format
+  - [ ] Perform Timestamp Verification pass (assert `exec_ts > signal_ts` always)
+- [ ] **CP6: Performance Benchmarking & Sweeps**
+  - [ ] CP6a: Populate Sharpe sensitivity table across 20 flawed datasets
+  - [ ] CP6b: Run engine-level execution lag comparative test
+  - [ ] CP6c: Net metrics extraction (Version A vs Version B impact analysis)
+  - [ ] CP6d: Threshold parameter sweep (Z=2.0 vs Z=2.57)
+  - [ ] CP6e: Negative control execution (CVNA/ISRG vs INTC/JPM)
+- [ ] **CP7: Final Verified Backtest Log**
+  - [ ] Header formatting
+  - [ ] Timestamp pass statement block
+  - [ ] Trade log export (20-30 rows sample)
+  - [ ] 2D Sharpe table documentation
+  - [ ] Summary metrics table (A vs B)
+  - [ ] Comparative analysis and Red Flag review notes
+  - [ ] Final audit trail verification string match
+
+## Red Flags / Validation Warnings
+- [ ] Sharpe Ratio > 5 on 'Clean' Dataset (Failure if True)
+- [ ] Sharpe value not cleanly monotone across increasing `k%` range for H1/H3
+- [ ] Kalman β (Version B) underperforms Version A on clean dataset
+- [ ] Any trade executed at `t` rather than strictly `t+1`
+- [ ] CVNA/ISRG Sharpe > 0 with significant edge (Invalidates pairs signal logic)
```

## 🤖 Assistant  `2026-04-20T02:55:13`
<details><summary>💭 Thinking</summary>

**Refining Operational Protocols**

I've been meticulously refining my operational protocols. I'm actively integrating specialized tools to manage files and data efficiently, and am completely avoiding general-purpose commands that can be a source of problems. My focus now is on robust and accurate execution.

</details>

I have read through the `Week3_Final_Plan_v3.md` document and generated the requested materials:

1. **Implementation Plan** (see the new artifact): I have constructed a complete implementation plan detailing the proposed new folder structure, the precise data specifications required, and the step-by-step methodology for executing Deliverable 1 and Deliverable 2 strictly following your provided guidelines.
2. **Workflow Checkpoint**: I have generated a structured `.md` checklist mapping directly to the project checkpoints. You can find this file in your workspace at: `d:\Quant Finance\Quant Program\Week 3\Week3_Workflow_Checkpoint.md`.

Please **review the implementation plan** and address the open questions regarding data availability inside the artifact. Once you approve the plan, we can begin implementing the Deliverable 1 scripts!

*Model: `MODEL_PLACEHOLDER_M37` | Stop: `STOP_REASON_STOP_PATTERN` | Think: `1.006706600s`*

### 🔧 Tool: `list_dir`  `2026-04-20T03:02:11`
`[List Directory]`
