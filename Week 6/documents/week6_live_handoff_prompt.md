# Week 6 Live Paper Trading — Handoff Prompt

Paste this entire file as the first message in a new chat session to start Week 6 live deployment work.

---

## Project context

- **Working dir**: `d:\Quant Finance\Quant Program\Week 6\`
- **Backtest research COMPLETE** as of 2026-05-25. Strategy V4 daily stat-arb shipped with production-verified results (Sharpe +1.28). Bit-identical verification done.
- **This week's deliverable**: Live Paper Trading Dashboard URL + screenshot, with 5-min API disconnect resilience proven.
- **Grading**: Infrastructure resilience + dashboard health. NOT trading P&L.

## Production ship config (LOCKED — do not re-tune)

```bash
python scripts/run_v4_pipeline.py --folds all \
    --use-dynamic-cost \
    --entry-z 3.0 --hard-sl-z 5.0 \
    --use-composite-filter
```

| Parameter | Value | Note |
|---|---|---|
| Entry Z | 3.0 | Sweet spot from full 39-fold sweep |
| Hard SL Z | 5.0 | 2σ buffer |
| β-cap | \|β\| ≤ 5 | Removes 27 numerical artifacts |
| Cost model | Week 5 dynamic | per-ticker per-day spread + impact + commission + borrow |
| Regime filter | Composite z-score SIMPLE | Skip month if `stress_z > q67_trailing_252d` |
| Discovery | PCA(5) + Johansen + BH-FDR + HL[5,30] | Standard V4 |

## Verified backtest metrics (`results/v4/z30_composite/fold_metrics.csv`)

- Mean per-fold Sharpe: **+1.279**
- Monthly Sharpe annualized: **+0.660**
- Sum return: +2.32% over 26 active months
- Win rate: 17/26 = 65%
- After survivorship adjustment (~−0.2): ~+1.08

## Honest expectations for live (don't oversell)

- **Annualized return ~+0.71%** on $100k = +$700/year
- **Strategy does NOT beat T-bill (5%)** at any leverage level (margin interest dominates)
- Real fills will likely make it WORSE (slippage, bid-ask, rejections)
- This is **academic/infrastructure deliverable**, not investment vehicle
- Sharpe +1.28 = low absolute return + very low variance, not high return

## Deliverable requirements (Week 6 assignment)

1. Connect to Brokerage API (paper trading)
2. WebSocket connection + API authentication
3. Drift Monitor — Live trades vs backtest predictions
4. AI Audit: simulate 5-min API disconnect — bot must reconnect, not crash
5. Live Trading Dashboard URL + screenshot showing:
   - Live P&L
   - Open positions
   - API connection status

## 5-Phase plan (detail in `documents/pipeline_week6_live.md`)

| Phase | Effort | Where |
|---|---|---|
| 1. Setup + Connection | 2-3 days | Local |
| 2. Strategy live engine | 2-3 days | Local |
| 3. Resilience drills (GRADED) | 1-2 days | Local |
| 4. Dashboard (FastAPI+HTMX) | 1-2 days | Local |
| **4.25. Monitoring + alerting** | **1-2 days** | **Local** |
| 4.5. Cloud deploy (Render.com) | 4-6h | Cloud |
| 5. Live run | 1-2 weeks | Cloud |

### Monitoring system (§6.5 of pipeline_week6_live.md)

Beyond just dashboard — full coverage:
- **Discord webhook alerts** (4 tiers: CRITICAL / ERROR / WARN / INFO)
- **Auto kill-switch** (drawdown >3%, drift loss >25% gross PnL, reconcile mismatch, stale data)
- **Scheduled reports** (daily 16:30 ET, weekly Fri 17:00, monthly EOM)
- **Trade journal** (per-trade attribution: predicted vs realized cost)
- **Live-vs-backtest rolling comparison** (21-day Sharpe, win rate, drift)
- **Anomaly detection** (fill slippage > 50bps, order latency > 2s, depth > 80% L1)

User watches Discord (real-time) + dashboard (daily) + reports (weekly). NOT screen-watching during market hours.

## Architecture stack

| Layer | Choice |
|---|---|
| Broker | **Alpaca Paper** (free, Python SDK, WebSocket+REST) |
| Language | Python 3.12 (reuse `engine_daily/`) |
| State | SQLite (single file, persistent disk) |
| Dashboard | FastAPI + HTMX (no React) |
| Logging | Loguru |
| Cloud | **Render.com $7/month** (after Phase 4) |

## Scaled-down universe for paper

- Backtest used 528 tickers, gross $2.5M-8M (3-5x implicit leverage)
- Paper margin (Alpaca Reg T 2x) cannot support full universe
- **Use top 50 most-liquid tickers**
- **Notional $5-10k per pair** (scaled from $25k backtest)
- Total gross ~$250-500k on $100k paper capital

## LOCKED decisions (do not relitigate)

1. Strategy params unchanged from backtest (Z=3.0, SL=5.0, β≤5)
2. Composite z-score filter ON (binary halt)
3. Static β only — Kalman REJECTED at Z=3.0 (median lift -0.15)
4. No carry-forward — REJECTED (Sharpe -0.66 vs +1.28)
5. No size dampener — REJECTED (-0.42 vs binary halt)
6. Universe: top 50 (scaled from 528)
7. Notional: $5-10k per pair (scaled from $25k)
8. Cloud: Render.com $7/month
9. Dashboard: FastAPI + HTMX (no React)
10. State: SQLite single file
11. Run duration: 1 month paper trade
12. Bar frequency: daily (end-of-day decisions at 15:55 ET, execute at 16:00 close)

## Pre-deployment checklist (all 10 must pass before submit)

1. Broker auth verified ✓
2. WebSocket connected and streaming ✓
3. State store writable ✓
4. State recovers after kill -9 ✓
5. **5-min disconnect drill passed (GRADED)** ✓
6. Position reconcile mismatch handling ✓
7. Idempotent order submission (client_order_id dedup) ✓
8. Composite z-score regime features buildable live ✓
9. Dashboard all endpoints load ✓
10. Cloud deployment accessible ✓

## Critical files to read first

- `documents/pipeline_week6_live.md` — full implementation spec
- `documents/log.md` — backtest journey + ship config
- `engine_daily/engine_daily.py` — V4 state machine logic to port
- `engine_daily/regime_detector.py` — composite z-score to reuse
- `scripts/run_v4_pipeline.py` — fold loop + filter wiring as reference
- `results/v4/z30_composite/fold_metrics.csv` — production-verified results

## Memory references (auto-loaded)

- `[[week6-v4-status]]` — ship config detail
- `[[week6-rejected-approaches]]` — what NOT to try
- `[[week6-key-files]]` — file path index
- `[[quant-decision-rules]]` — methodology requirements
- `[[quant-quick-test-preference]]` — testing approach

## First action (start here)

1. Create Alpaca paper account at https://alpaca.markets — generate API key + secret
2. Install: `pip install alpaca-py fastapi uvicorn loguru sqlalchemy schedule python-dotenv`
3. Create `.env` (in `.gitignore`): `ALPACA_API_KEY=...`, `ALPACA_SECRET_KEY=...`, `ALPACA_PAPER=true`
4. Test REST: `from alpaca.trading.client import TradingClient; client = TradingClient(...); print(client.get_account().cash)`
5. Pick top-50 universe: sort `engine_daily` ticker cache by avg daily $ volume, take top 50
6. Create `live/` folder structure per `pipeline_week6_live.md` §8
7. Start Phase 1 Day 2: WebSocket connection + bar builder

## Anti-patterns (DON'T)

- ❌ Re-run backtest (production results verified bit-identical)
- ❌ Tune strategy params (Z, SL, β cap all locked)
- ❌ Try Kalman / carry-forward / sizer (all rejected with evidence in `log.md`)
- ❌ Maximize P&L (graded on infrastructure)
- ❌ Skip disconnect drill (THIS is the graded test)
- ❌ Commit `.env` to git (will leak Alpaca keys)
- ❌ Use Oracle Cloud free tier (per Discord: signup hell)
- ❌ Build React dashboard (HTMX is enough for screenshot)
- ❌ Deploy cloud before Phase 4 (debug locally first, save $)

## User preferences

- Vietnamese, non-engineer — prefer plain-VN explanations (`/explain` skill)
- Pre-committed decision rules required (avoid p-hacking)
- Honest disclosure required (flag bugs even when inconvenient)
- Tiered testing: analytical → subset → full
- Sample size flags when relevant
- Update memory (`MEMORY.md`) when major findings/decisions

## Status banner for new chat

> Week 6 backtest research COMPLETE (Sharpe +1.28, ship-ready).
> Now starting Week 6 LIVE DEPLOYMENT phase.
> Goal: Live paper trading dashboard with 5-min disconnect resilience proven.
> Effort: ~2-3 weeks active dev + 1-2 weeks live run.
> Cloud: Render.com $7/month.
> Broker: Alpaca paper.

---

End of handoff prompt. Begin Phase 1 Day 1.
