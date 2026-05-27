# Week 6: The Engine — Live Paper Trading + Engine v2 + Cost Model v2

## 0. Objective

**Theme:** Going Live, Honestly.

Week 6 is two parallel tracks:

1. **Track A — Live paper trading deployment.** Take the frozen Week 4 + Week 5 stack to a paper-trading broker, prove it survives a real WebSocket and a real disconnect, and measure realized-vs-predicted slippage against the Week 5 dynamic cost model.
2. **Track B — Engine v2 and Cost Model v2.** Build the *better* version of the Week 4 engine and Week 4 backtest, and the *better* (more dynamic, more realistic) version of the Week 5 cost model. Re-run the 45-fold walk-forward under v2 and compare against the V2.0 frozen baseline before going live.

Both tracks must finish before the Track A deliverable is filed. Engine v2 / Cost v2 results are gated by Acceptance Criteria in §7 — if v2 underperforms V2.0 on the head-to-head, we deploy V2.0 frozen and document v2 as research.

---

## 1. Scope & Integration Context

### 1.1 Inputs from Week 4 (Signal Engine, frozen V2.0)

| Input | Value | Source |
|---|---|---|
| Strategy | Cointegration pairs trading, 2D Kalman filter, prior-spread signal | [pipeline_results.md](../Week 4/results/pipeline_results.md) |
| Frozen config | persistence gate + HL ≤ 6d + CORR25 ≥ 0.25 (intra-session) + no-EOS + Z=3.5 + δ=1e-7 | whitepaper §3 |
| Walk-forward | 45 folds, 6-month formation → 1-month trading, monthly roll | whitepaper §2.1 |
| Baseline result | Mean SR **+0.995**, 25/45 folds, 48% positive, worst MaxDD **−3.15%**, 90 trades, 16.7% from β<0 pairs | whitepaper §4.2 |
| Regime profile | Bear +2.41 / Early Bull +2.84 / Mid Bull +1.34 / **Late Bull −0.75** | whitepaper §5.2 |
| Outputs consumed live | Frozen pair list, α_PCA / β_PCA, R, δ, Z thresholds, trade_log/rebalance_log schemas | [run_final_pipeline.py](../Week 4/run_final_pipeline.py) |
| Open W4 limitations carried in | Position carry-forward at fold boundary (deviation #9), aggressive decide-at-close fill-at-close latency (deviation #10), δ grid floor at 1e-7 (100% of folds) | pipeline_results.md §"Documented Deviations" |

### 1.2 Inputs from Week 5 (Friction Model, dynamic)

| Input | Value | Source |
|---|---|---|
| Cost model | `C_total = C_spread + C_impact + C_borrow` with κ-tier impact and Actual/365 borrow | [pipeline_week5.md §2.1](../Week 5/workflows/pipeline_week5.md) |
| Realized RT cost | **45.3 bps** dynamic vs **90.87 bps** static-60 — the "60 bps" anchor was 2× too conservative | [net_of_fees_report.md §4](../Week 5/reports/net_of_fees_report.md) |
| Net-of-fees Sharpe | Gross 0.503 / Static 0.365 / **Dynamic 0.443** (equity-curve corrected, C2 fix) | net_of_fees_report.md §1 |
| Cost decomposition | Spread 72%, Impact 24%, Rebalance 3%, Borrow <1% | net_of_fees_report.md §5 |
| κ tier distribution | 30% Tier 1, 65% Tier 2, 5% Tier 3 (post-formation calibration) | net_of_fees_report.md §3 |
| Kill zones | 11:00–11:30, 11:30–12:00, 15:30–15:59 are net-negative | net_of_fees_report.md §8 |
| Open W5 limitations carried in | Symmetric LOB (no order-flow imbalance), no book-walking, fixed-per-fold κ, exit-day debit only, fixed 50 bps borrow, DSR collapse at 50 trials | [methodology_results.md §11](../Week 5/reports/methodology_results.md) |
| W5 deferred-to-W6 list | Dynamic-cost NC, L2 sensitivity, liquidity-gated sizing, cost-aware rebalance gate, regime gateway, VPIN, TWAP/VWAP, adaptive κ, per-pair δ, cross-tertile pairs | [deferred_to_week_6.md](../Week 5/workflows/deferred_to_week_6.md) |

### 1.3 What "Live Mode" Actually Inherits

The live engine ingests 1-minute OHLCV + L1/L2/L3 LOB snapshots and updates Kalman state `θ̂(t) = [α(t), β(t)]ᵀ` bar-by-bar. It executes threshold rebalances using the **frozen formation-window parameters** from the most recent completed fold; only the Kalman posterior and live spread features evolve in real time. Strategy parameters do not adapt during the live window.

---

## 2. Self-Audit & Risk Disclosures Before Going Live

These are the load-bearing concerns surfaced by the Week 4 + Week 5 evidence. Each must have a documented answer in the live deployment, not just an implementation:

| # | Concern | Evidence | Required answer in this spec |
|---|---|---|---|
| R1 | The most recent fold (45 / 2026-03) is **Late Bull**, the only structurally negative regime (mean SR −0.75) | whitepaper §5.2 | §5 regime kill-switch + §3 sizing reduction |
| R2 | Live sample is ~22 trading days; DSR will be 0 trivially | net_of_fees_report.md §10 | §7 — live is deployment evidence, not significance |
| R3 | Static "60 bps" anchor is 2× the realized cost. Misleading benchmark in dashboard | net_of_fees_report.md §4 | §4.4 — primary anchor is dynamic 45 bps |
| R4 | Symmetric LOB limitation goes away with broker data. Order-flow imbalance and book-walk become possible | methodology_results.md §11.1 | §4.4 lifts the EXT-3 book-walk model into the production cost path |
| R5 | Outlier 14:00–14:30 trade with 7,923 bps gross was never reviewed | methodology_results.md §11.5 | §4.5 enforces a per-trade gross-bps sanity ceiling |
| R6 | β<0 pairs were the BUG-3 fault line. 16.7% of trades come from this class. Live engine must enforce signed `shares_b` and the `side_A`/`side_B` log columns | whitepaper §3.3 | §4.3 freezes these contracts |
| R7 | Week 4 starts each fold flat. Live has no fold boundary — must specify monthly re-formation behavior without flattening open positions | pipeline_results.md deviation #9 | §4.3 (E2 fix) |
| R8 | Week 4 latency is decide-at-close / fill-at-close. Live is decide-at-bar-close / fill-on-next-bar at minimum. Cost model and Sharpe baseline must be recalibrated | pipeline_results.md deviation #10 | §4.3 (E3 fix), §4.4, §6 |
| R9 | δ collapses to grid floor (1e-7) in 100% of W4 folds → Kalman is effectively static. v2 should validate this on live data, not assume it | pipeline_results.md §11 | §4.3 (E1 — adaptive δ candidate) |
| R10 | DSR fails at 50 trials. Adding a Week 6 v2 increases the trial count. Every v2 variant tested expands E[max SR] | net_of_fees_report.md §10 | §7 — pre-register v2 variants |

---

## 3. Track B — Engine v2 (Better Week 4)

Goal: Build a successor backtest engine that addresses the highest-priority W4 limitations and is the same engine the live deployment will run. Re-run the same 45 folds, same data, same pair selection, and compare head-to-head against V2.0.

### 3.1 Variants to test (pre-registered)

Each is a focused change; we test them in isolation and as the bundled v2 candidate.

| ID | Change | Priority | Source |
|---|---|---|---|
| **E1** | **Adaptive δ via dual-rate Kalman** — slow δ for parameter, fast δ for residual variance estimate, instead of frozen-per-fold | High | W4 §11, §6 |
| **E2** | **Position carry-forward across fold boundaries** — open trades survive monthly re-formation; pair list refreshes but Kalman θ̂ persists for any pair that survives the new persistence gate | High | W4 deviation #9 |
| **E3** | **Latency convention fix** — decide-at-bar-close, fill-at-next-bar-open (or VWAP of next bar). Re-run full backtest under realistic latency. | High | W4 deviation #10 |
| **E4** | **Z-velocity entry filter** — `z_velocity_K = (Z[t] − Z[t-K]) / K` for K ∈ {60, 195}. Reject entries below threshold. Targets Late Bull factor-drift failures | Highest (untested) | W4 §7.2 |
| **E5** | **Liquidity-gated position sizing** — cap shares at fraction (e.g. 10%) of L1 depth at entry bar. Lifts W5 deferred item | Medium | [deferred_to_week_6.md](../Week 5/workflows/deferred_to_week_6.md) |
| **E6** | **Cost-aware rebalance gate** — defer rebalance when current dynamic cost > rebalance gain estimate. Lifts W5 deferred item | Medium | deferred_to_week_6.md |
| **E7** | **Per-trade outlier ceiling** — trades with gross > 99th percentile of formation-window gross are flagged and routed to manual review log; no automatic suppression | Low (audit) | W5 §11.5 (R5) |

E1–E4 are the "better engine"; E5–E6 are the "better cost-aware execution"; E7 is a defensive audit hook.

### 3.2 Acceptance for Engine v2 (head-to-head vs V2.0)

The bundled v2 must clear **all** of these on the same 45 folds:

- Mean Sharpe ≥ V2.0 +0.995 (no regression)
- **Late Bull mean Sharpe materially > V2.0 −0.750** (the v2 motivation)
- Worst MaxDD ≤ V2.0 −3.15% (no tail-risk regression)
- NC pass rate ≥ V2.0 32% on the same 25 active folds
- PBO ≤ 10% (V2.0 was 0.030)
- Lookahead audit: 0 violations (same as V2.0)

If E4 (Z-velocity) is the only variant that clears Late Bull, ship just E4. Document failed variants in `results_week6/v2/ablations.md`.

### 3.3 What v2 must NOT change

To keep the comparison clean: same pair-selection pipeline (Phase 1 + persistence gate + HL≤6d + CORR25), same Z=3.5 entry, same dollar-neutral $20k/pair sizing **before** the E5 liquidity gate, same trade_log / rebalance_log schemas (Week 5 contract).

---

## 4. Track A — Live Paper Trading

### 4.1 Broker & Data Connectivity (Dev Role)

**Decision (open — to be filled by user comments):**
- [ ] Broker = `Alpaca paper` | `Interactive Brokers Paper`
- [ ] Market data feed = same broker | external (Polygon / Databento)

Spec applies to either; the only difference is L2/L3 depth quality (IB > Alpaca free).

**Required:**
- Auth via env-var-loaded API keys; no keys in repo. `.env.example` committed.
- WebSocket stream for: trade prints, L1 quotes (always), L2/L3 quotes if the broker supports.
- REST fallback for: order submission, position query, account state.
- 1-minute bar builder from trade prints (do not trust broker-aggregated bars before validating against tick-replay; bar timing skew has bitten production engines before).
- All inbound data persisted to a local append-only log (`data/live/{date}/quotes.parquet`, `bars.parquet`, `book.parquet`) before being consumed by the engine — enables exact replay during incident review.

### 4.2 Resilience & State Persistence (AI Audit Target)

Beyond the 5-minute disconnect drill, the engine must survive an **arbitrary** crash (kill -9) and resume cleanly. Required:

- **State store** — Kalman `θ̂(t)`, posterior covariance `P(t)`, open positions, in-flight orders, last bar processed. SQLite or a single append-only Parquet per fold; flush at every bar close.
- **Idempotent order submission** — every order carries a deterministic `client_order_id = hash(pair_id, bar_ts, leg, side)`. Reconnect re-submits will be deduped by the broker.
- **Position reconciliation on reconnect** — broker positions vs local store. If divergent, halt new entries and surface a manual-review alert; do *not* auto-reconcile by trading.
- **Bar-gap detection** — if the next 1-minute bar is more than 90s late, mark the engine as "stale" and refuse new entries until a fresh bar arrives. Existing positions continue to be priced at last good bar.
- **Disconnect drill** — at a pre-scheduled window during paper trading, kill the WebSocket for 5 minutes. Engine must (a) detect within 30s, (b) refuse new entries, (c) keep state, (d) reconnect, (e) re-subscribe, (f) reconcile positions, (g) resume — without crash and without duplicate orders.

### 4.3 Live Execution Engine (Engine v2 in production mode)

The engine v2 from §3 runs unchanged in live mode, with these additions:

- **Bar source switch** — historical Parquet → live WebSocket bar builder.
- **E2 fold-boundary handling** — at month rollover, re-run pair selection on the new formation window. Pairs that survive the new persistence gate AND were already open: continue with carried-forward Kalman state. Pairs that drop out of the new universe: close at next bar (NOT immediately at rollover instant — avoid an artificial concentrated-exit moment). New pairs that pass: eligible for entry signals. Document the rollover ledger.
- **E3 latency** — order is sent at bar-close + 100ms; assumed fill at next bar's first tick (market-on-bar-open) or VWAP of first 30 seconds for thinner names. Live realized latency goes into the Drift Monitor.
- **EOS rules** — Track A inherits W4's no-EOS configuration. Strict adherence to 09:30–15:59 ET means *no entries* outside that window; positions opened intraday may carry overnight per the Week 4 design. If user wants EOS=ON for paper-trading risk control, set a config flag and re-run the W4 backtest under EOS=ON for an apples-to-apples comparison first (don't flip live without re-baselining).
- **Per-pair sizing** — start at $20k/pair × max 50 pairs = $1M paper notional, then apply E5 liquidity gate. Reduce paper capital to $250k–$500k for live disclosure conservatism unless the user opts otherwise.

### 4.4 Cost Model v2 — The Better Dynamic, Realistic Friction Model

This is the headline Week 6 cost-model deliverable. Goals: (i) keep the Week 5 three-component shape, (ii) replace static-per-fold parameters with live-observed estimates, (iii) lift the W5 deferred items that are now feasible with broker LOB data.

#### 4.4.1 What carries over from Week 5
- Three-component decomposition: spread + impact + borrow.
- κ-tier framework as the *prior*; live observations update it.
- Full Actual/365 borrow accrual on calendar days held.
- C2 Sharpe convention (bar-level equity curve, cost as exit-day debit).
- B4 spread-floor `clip(lower=0)` to avoid crossed-quote revenue artifacts.
- Schema continuity: every live trade emits a `cost_log` row with the same columns as W5 plus the new `realized_*` and `predicted_*` fields below.

#### 4.4.2 What v2 changes (dynamic, realistic)

| ID | Change | Why |
|---|---|---|
| **C1** | **Adaptive κ on rolling 5–10-day window**, not fixed-per-fold tier. Use exponentially-weighted realized impact / σ to update κ; tier still serves as the prior. | W5 §11; deferred-list "Adaptive κ via online learning" |
| **C2** | **Asymmetric LOB live** — record bid_sz vs ask_sz; expose order-flow imbalance signal `OFI = (bid_sz − ask_sz) / (bid_sz + ask_sz)`. Use as a cost adjustment, not as a strategy signal. | W5 §11.1, deferred "VPIN" |
| **C3** | **Book-walking model in production path** (W5's EXT-3 promoted): when order shares > L1 depth, compute effective spread from L2/L3 walk. | W5 EXT-3 |
| **C4** | **Live borrow rate from broker API**, not fixed 50 bps. Fallback to 50 bps on API miss. | W5 deferred |
| **C5** | **Burn-in fix** — replace W5's 15 bps fallback during the 390-bar `min_periods` window with a Welford-style incremental rolling std seeded from historical formation data. Day-1 trades get a real σ estimate, not a constant. | W5 hooks.py:17 |
| **C6** | **Latency-aware spread cost** — cost is computed at the *fill* bar, not the *signal* bar (W5 EXT-1). Material if quotes autocorrelate. | W5 EXT-1 |
| **C7** | **Intraday financing** — small per-bar accrual on short notional (~1 bps/day = 0.0026 bps/min) for trades held intraday. Negligible at current sizes but documented for completeness. | W5 §11; deferred |
| **C8** | **Per-tier per-bucket `spread_seasonality` table refreshed weekly** rather than only at formation. | W5 §11 |
| **C9** | **Realized-vs-predicted residual capture** — every fill records `(realized_slippage_bps − predicted_slippage_bps)`; this is the input to the Drift Monitor in §4.5. | New |

#### 4.4.3 Cost Model v2 acceptance

Replay Week 5's 90 trades through v2 and assert:
- All math-violation invariants from W5 §10 still hold (`total_cost ≥ 0`, `net ≤ gross`, component additivity).
- v2 mean RT cost is within ±10 bps of W5's 45.3 bps on the same trade set (sanity check that the headline number doesn't drift unintentionally).
- κ time-series under C1 is bounded within {0.2, 1.0} (no runaway).
- Asymmetric-LOB OFI is non-degenerate (var > 0 across 90 trades).
- Book-walk triggers >0% on Tier 3 trades (otherwise C3 is dead code).

### 4.5 Drift Monitor (Strategist Role)

Drift = realized cost − predicted cost from §4.4. The monitor is *not* aggregate-only; it must decompose:

- **Per leg:** A vs B fills, entry vs exit.
- **Per intraday bucket:** the 13 buckets from W5 §8. Watch for new kill zones.
- **Per κ tier:** Tier 1 / 2 / 3.
- **Per fill type:** marketable-limit, market, IOC.
- **Cumulative drift budget:** running sum of (realized − predicted) in $; halt new entries when cumulative drift > 25% of cumulative gross PnL (configurable).

**Statistical test** (run end-of-day): paired t-test over the day's fills, H₀: realized = predicted. p < 0.01 with cumulative drift in the same direction → flag. Do not auto-act on a one-day signal; require 3 consecutive days of one-sided drift before suspending.

**Outlier guard (R5)** — any fill with realized slippage > 99.5th percentile of W5's 90 trades (≈ 200 bps RT) is flagged and the trade is force-flattened at next bar; the trade is logged as `manual_review_required = True`.

### 4.6 Live Trading Dashboard

**Sections (all mandatory):**

1. **Connectivity strip** — broker status, market-data heartbeat (last bar age in seconds), last reconcile pass timestamp, kill-switch state.
2. **P&L** — Live Gross, Live Net (using v2 cost model), W5 Static-60 anchor (for comparison only — not the primary number; flag as "conservative legacy" anchor).
3. **Open pairs table** — pair_id, side_A, side_B, β, current Z, predicted slippage at exit if flat now, time-in-trade, intraday-bucket alarm if in a known kill zone.
4. **Drift panel** — today's realized vs predicted (per leg / per bucket / per tier), cumulative drift $, paired-t p-value, days in same-side drift streak.
5. **Regime panel** — current macro regime tag (Bear / Early Bull / Mid Bull / **Late Bull**, manually labeled at deploy time), the W4 Sharpe for that regime, kill-switch threshold and current realized live Sharpe vs threshold.
6. **Engine state** — Kalman snapshot (α, β, P diag) for top-3 active pairs, last δ value (for E1 adaptive variant), last fold rollover timestamp, position-carry-forward count.
7. **Audit log** — disconnect events, reconnect duration, reconciliation outcomes, outlier flags, manual-review queue.

**Form:** local web app (FastAPI + simple HTMX or React) is sufficient; full hosting is optional. Snapshot the dashboard hourly into `dashboard_screenshots/{ts}.png` for the deliverable.

---

## 5. Lifting Deferred Items from Week 5

For each item in [`deferred_to_week_6.md`](../Week 5/workflows/deferred_to_week_6.md), state Week 6's stance:

| Deferred item | W6 stance |
|---|---|
| Liquidity-gated position sizing | **Lifted** (E5) |
| Cost-aware rebalance gate | **Lifted** (E6) |
| HMM regime-detection gateway | **Partial** — §4.6 dashboard surfaces a manually-labeled regime; full HMM deferred to W7. The §5 kill-switch acts as a coarse regime gate based on realized live Sharpe |
| Order-flow toxicity (VPIN / Kyle's λ) | **Partial** — C2 captures OFI as cost input; full VPIN deferred to W7 |
| Optimal execution (TWAP / VWAP) | **Deferred to W7** — at $20k/leg, market-order single-bar execution is acceptable per W5 §11.3 |
| Adaptive κ via online learning | **Lifted** (C1) |
| Per-pair Kalman δ optimization | **Partial** — E1 is dual-rate, not per-pair. Per-pair deferred to W7 |
| Cross-tertile pairs analysis | **Deferred to W7** — strategy unchanged in Track A, and v2 (Track B) does not modify the universe |

W5 deferred priorities 1–5 from `methodology_results.md §14`:

| W5 priority | W6 stance |
|---|---|
| P1 Dynamic-cost NC | **Lifted** — re-run W4 NC pairs through v2 cost model; report alongside live deployment |
| P2 L2 sensitivity sweep | **Lifted** — broker LOB is asymmetric; the W5 blocker is gone |
| P3 Rename `check_dsr_degradation` | **Cleanup task** during v2 implementation |
| P4 Outlier review (14:00–14:30 trade) | **Lifted** (R5 / §4.5 outlier guard) |
| P5 DSR re-evaluation | Open — accumulate live observations; rerun DSR after 60+ days, with `n_trials` updated to include W6 v2 variants |

W4 open items:

| W4 open item | W6 stance |
|---|---|
| Z-velocity entry filter (W4 §7.2 P1) | **Lifted** (E4) |
| n_c25 ≤ 12 fold-skip hypothesis | Open — only validated by accumulating future folds; do not deploy as a hard filter in v2 |
| Factor orthogonalisation | **Deferred to W7** |
| 16-combination Option A factorial | **Deferred to W7** |
| Point-in-time universe (CRSP) | **Deferred to W7** |

---

## 6. Pre-Deployment Checklist (one-time, gated)

Before the first paper-trading session can begin:

- [ ] Engine v2 head-to-head report committed to `results_week6/v2/headtohead.md` and meets §3.2 acceptance, OR engine v2 is shelved and live runs V2.0 frozen
- [ ] Cost Model v2 replay against W5's 90 trades passes §4.4.3 acceptance
- [ ] Broker auth verified end-to-end on a single small order, then cancelled
- [ ] State persistence drill: kill -9 → restart → state recovered; documented in `results_week6/resilience/restart_drill.md`
- [ ] 5-minute disconnect drill: documented in `results_week6/resilience/disconnect_drill.md`
- [ ] Position reconciliation: deliberately desync local store from broker by 1 share, confirm engine halts with manual-review alert (does NOT auto-trade to reconcile)
- [ ] Idempotency: re-submit the same `client_order_id` and confirm broker dedupes
- [ ] Kill-switch wiring: trigger threshold manually, confirm engine refuses new entries and (configurably) flattens existing
- [ ] Regime label set explicitly in dashboard config (likely `Late_Bull_2025_Q1_2026` for an early-2026 deploy) — visible to anyone reading the dashboard

---

## 7. Acceptance Criteria for the Live Deliverable

The Week 6 grade is based on infrastructure resilience + analytical honesty, not on live PnL.

**Hard requirements:**
- Engine survives a 5-minute disconnect cleanly (no crash, no duplicate orders, state preserved).
- Engine survives a process kill cleanly (state recovered from store).
- All trades during the live window are logged with both predicted (v2) and realized cost.
- No lookahead audit violations during live (same automated check as W4).
- Dashboard sections from §4.6 all populated and updating.

**Soft requirements (reported, not pass/fail):**
- Realized vs predicted slippage residual mean within ±15 bps.
- Cumulative drift below 25% of cumulative gross PnL.
- Live mean Sharpe is *informational only* — N is too small for inference (R2). Report it; do not gate on it.

**Anti-acceptance (auto-fail):**
- Any trade that bypasses §4.5 outlier guard.
- Any reconnect path that auto-trades to reconcile a desync.
- Any deployment under a regime label of "unknown" — operator must commit to a regime tag at start.

---

## 8. Final Deliverable Format

A folder `results_week6/live_run_{YYYYMMDD}/` containing:

1. `dashboard_url.txt` — if web-hosted; or `dashboard_screenshots/` directory with hourly snapshots otherwise.
2. `trade_log.parquet` — same schema as W4 + `predicted_cost_*`, `realized_cost_*`, `client_order_id`.
3. `cost_log.parquet` — same schema as W5 + `realized_*` columns.
4. `drift_report.md` — §4.5 monitor's end-of-window summary, with the per-bucket / per-tier breakdown.
5. `resilience/disconnect_drill.md`, `resilience/restart_drill.md` — written before live, referenced from this doc.
6. `live_vs_predicted.md` — short post-mortem: did W5's 45.3 bps RT match live? Where did the model under/over-predict?
7. `regime_card.md` — single page: which W4 regime are we live in, what is its W4 Sharpe, what was the kill-switch threshold, did it trigger.
8. (If Track B shipped) `v2/headtohead.md` and `v2/ablations.md` — the engine and cost-model v2 results.

**Grading note:** A clean dashboard screenshot + a successful disconnect drill is the minimum bar; the analytical honesty of `live_vs_predicted.md` and `regime_card.md` is what separates a passing submission from a strong one.

---

## 9. Open Decisions (waiting on user comments)

These are flagged for the user's later round of comments — do not assume defaults silently:

- **D1.** Broker choice (Alpaca paper vs IB paper).
- **D2.** Paper notional ($1M like W4, or reduced for risk).
- **D3.** EOS=ON or =OFF for live (W4 frozen config is OFF; live overnight risk profile may push toward ON).
- **D4.** Engine v2 ambition — ship E1+E2+E3+E4 bundled, or only the variant(s) that clear §3.2 individually?
- **D5.** Cost Model v2 ambition — ship C1–C9 in full, or split C1+C5+C9 as the v2 minimum and defer the rest to W7?
- **D6.** Kill-switch threshold — fixed (e.g. cumulative drift > 25% of gross), or regime-conditional (tighter in Late Bull)?
- **D7.** Live window length — one trading week, one calendar month, until-disconnect-drill-passes?
- **D8.** Regime tag for the dashboard at deploy time.

Section §3 and §4 default to the most ambitious interpretation; user comments will trim scope where appropriate.
