# Week 6 Live Paper Trading — Implementation Spec

**Status**: Backtest research COMPLETE (2026-05-25). Live deployment phase next.
**Deliverable**: Live Trading Dashboard URL + screenshot showing P&L, open positions, API status.
**Grading focus**: Infrastructure resilience (5-min disconnect drill) + dashboard visual health. NOT trading P&L.

---

## 1. Inherited from backtest (LOCKED — do not modify)

### Production ship config

```bash
python scripts/run_v4_pipeline.py --folds all \
    --use-dynamic-cost \
    --entry-z 3.0 --hard-sl-z 5.0 \
    --use-composite-filter
```

- Entry threshold: |Z| ≥ 3.0
- Hard SL: |Z| ≥ 5.0
- β-cap: |β| ≤ 5
- Cost model: Week 5 dynamic (spread + impact + commission + borrow)
- Regime filter: composite z-score SIMPLE binary halt
- Discovery: PCA(5) + Johansen + BH-FDR + HL[5,30]

### Verified backtest metrics (`results/v4/z30_composite/fold_metrics.csv`)

| Metric | Value |
|---|---|
| Mean per-fold Sharpe | +1.279 |
| Monthly Sharpe annualized | +0.660 |
| Sum return | +2.32% over 26 active months |
| Win rate | 17/26 = 65% |
| After survivorship adj | ~+1.08 |

### Honest expectations for live

- Expected annualized return: **~+0.7%/year on $100k** = +$700/year
- **Strategy does NOT beat T-bill (5%) at any leverage** (margin interest dominates)
- Sharpe high because variance low, not because return high
- Live results may be WORSE due to: real slippage, real bid-ask cost, real order rejections, real broker data quirks
- **Deliverable is graded on infrastructure quality, not P&L**

---

## 2. Architecture stack

| Layer | Choice | Rationale |
|---|---|---|
| Broker | **Alpaca Paper** | Free, Python SDK, WebSocket+REST, no IB hassle |
| Language | Python 3.12 | Reuse `engine_daily/` code |
| State store | SQLite (single file) | ACID, no server, easy backup |
| Dashboard | FastAPI + HTMX | Lightweight, no React build |
| Logging | Loguru | Structured, easy grep |
| Scheduler | Python `schedule` library | Run end-of-day at 15:55 ET |
| Cloud (Phase 4.5+) | **Render.com $7/month** | MVP-friendly, GitHub auto-deploy |

### Why Alpaca paper (not IB)
- Free, easy signup
- Python SDK `alpaca-py` is well-maintained
- WebSocket streaming included
- Paper account is FULL functionality (margin, options, limit/market orders)

### Why Render.com (not Oracle/DigitalOcean)
- 15-min sign-up (vs Oracle setup hell)
- GitHub auto-deploy: push commit → cloud redeploys
- Persistent disk add-on for SQLite
- $7/month tier removes 15-min inactivity sleep
- Per Discord discussion: Candy confirmed it works

---

## 3. Implementation plan (5 phases + 1 deploy)

### Phase 1 — Setup + Connection (2-3 days, LOCAL)

**Day 1**:
- Create Alpaca paper account at https://alpaca.markets
- Generate API key + secret
- Install: `pip install alpaca-py fastapi uvicorn loguru sqlalchemy schedule`
- Test REST auth: `account = api.get_account(); print(account.cash)`
- Create `.env` file (not committed): `ALPACA_API_KEY=...`, `ALPACA_SECRET_KEY=...`

**Day 2**:
- WebSocket connection to Alpaca real-time data
- Subscribe to top-50 ticker quote stream (L1)
- Build 1-minute bar from trade prints (in-memory)
- Compare built bars vs Alpaca's official bars (should match within tolerance)

**Day 3**:
- SQLite state store schema:
  ```
  positions(pair_id, side_a, side_b, beta, notional, entry_date, entry_z, ...)
  orders(client_order_id, broker_order_id, status, timestamp, ...)
  bars(ticker, ts_minute, open, high, low, close, volume)
  ticks(ticker, ts, price, size)  # for bar audit
  audit_log(ts, event, level, message)
  ```
- Smoke test: kill process mid-stream → restart → state recovers

**Pre-Phase-2 gate**:
- Account balance retrievable ✓
- Bars match official ✓
- State recovers after kill -9 ✓

### Phase 2 — Strategy live engine (2-3 days, LOCAL)

**Day 4-5**:
- Port `engine_daily.engine_daily.run_pair_daily` to streaming mode (`live/engine_live/live_pair.py`)
- Z-score: load last 60 daily bars from backtest cache + append live daily bars
- Per-pair state machine: same logic as backtest (entry |Z|>3.0, exit zero-cross, hard SL |Z|>5.0)
- Order placement:
  - Marketable-limit at next-bar open
  - Idempotent client_order_id: `hash(pair_id, bar_ts, leg, side)`
  - Submit via REST, track fills via WebSocket

**Day 6**:
- Daily scheduler: at 15:55 ET, decide entries based on EOD bar
- Order execution at 16:00 close (or next-day open if MOO)
- Composite filter check at start of each calendar month:
  - Load regime features from `engine_daily/regime_detector.py`
  - Compute stress_z + q67 trailing 252d
  - If halt: skip new entries this month, hold existing
- Mark-to-market positions per minute

**Pre-Phase-3 gate**:
- 1 pair (e.g., JPM/BAC) trades end-to-end in 1 simulated day ✓
- Idempotency: re-submit same order ID → broker dedups ✓
- Composite filter halts month correctly when stress high ✓

### Phase 3 — Resilience drills (1-2 days, LOCAL)

**This is what's GRADED. Invest heavily.**

| Drill | Pass criteria | Effort |
|---|---|---|
| **5-min WebSocket disconnect** | Detect within 30s, halt new entries, reconnect, reconcile positions, resume cleanly | 6 hours |
| **State persistence**: `kill -9` mid-day | Restart recovers all state, no duplicate orders | 3 hours |
| **Position reconcile mismatch**: off-by-1 share | Engine halts + alerts, does NOT auto-trade | 2 hours |
| **Idempotency**: re-submit same order ID | Broker dedups, no double order | 1 hour |
| **Bar gap > 90s** | Engine pauses new entries until fresh bar | 2 hours |
| **Order rejection** (insufficient buying power) | Engine logs + alerts, continues with other pairs | 1 hour |

Each drill = pytest unit test in `live/drills/`. Run before any deploy.

**Critical implementation detail**:
- WebSocket reconnect: exponential backoff 1s → 2s → 4s → 8s → cap 60s
- On reconnect: query broker positions, compare to local SQLite, halt if mismatch >1 share
- Stale data threshold: 90 seconds since last bar → pause new entries

### Phase 4 — Dashboard (1-2 days, LOCAL)

FastAPI + HTMX (server-rendered HTML, no React):

**Endpoints**:
```
GET  /              # main dashboard page (HTML)
GET  /api/status    # connectivity + heartbeat JSON
GET  /api/pnl       # live P&L (gross, net, vs backtest predicted)
GET  /api/positions # open positions table
GET  /api/drift     # drift monitor (realized vs predicted cost)
GET  /api/log       # last 50 events
WS   /ws/updates    # push updates to dashboard
```

**Required sections (from assignment + spec §4.6)**:

1. **Connectivity strip** — Broker status, last bar age (s), reconcile timestamp, kill-switch state
2. **P&L** — Live gross, live net (V4 cost model), Backtest predicted (control for drift)
3. **Open positions table** — pair_id, side_A, side_B, β, current Z, time-in-trade
4. **Drift panel** — realized vs predicted per leg/per ticker, cumulative drift $
5. **Regime panel** — Current stress_z, q67 threshold, halted? (true/false), days since last halt
6. **Engine state** — top-3 active pairs Z snapshot, last cron run timestamp
7. **Audit log** — Last 50 events: disconnect, reconnect, fills, alerts

**Screenshot capture**: hourly cron job snapshot dashboard → `dashboard_screenshots/{ts}.png`. Used for deliverable.

### Phase 4.5 — Cloud deployment (4-6 hours)

**Steps**:
1. Push code to GitHub **private** repo (don't commit `.env`!)
2. Create Render.com account, link GitHub
3. Create new "Web Service" from repo
4. Set environment variables (Alpaca keys, log paths)
5. Add **persistent disk** (1GB) for SQLite state file
6. Configure health check endpoint `/api/status`
7. Set auto-deploy on push to `main` branch
8. **Upgrade to $7/month tier** before Phase 5 (removes 15-min inactivity sleep)
9. Verify dashboard accessible at `https://<your-app>.onrender.com`
10. Smoke test: 1-hour live run on cloud, monitor for issues

**Render.com config tips**:
- Build command: `pip install -r requirements.txt`
- Start command: `python live/main.py & uvicorn live.dashboard.server:app --host 0.0.0.0 --port $PORT`
- Region: pick closest to broker (Alpaca = US East)
- Disk mount path: `/data` (mount SQLite here so it persists across deploys)

### Phase 5 — Live run + report (1-2 weeks, CLOUD)

**Window**: 1 calendar month paper trading (or until deadline).

**Daily monitor checklist**:
- Connectivity: 100% uptime expected
- Halts: ~30% of trading days expected halted by composite filter
- Trades: ~3-5 per day during active months
- Drift: |realized − predicted| within ±15 bps

**Weekly report**:
- `drift_report.md` — per-bucket realized vs predicted slippage
- `regime_card.md` — which composite regime, how often halted
- `live_vs_predicted.md` — did backtest predictions match live?

**Deliverable artifacts**:
- Dashboard URL: `https://<your-app>.onrender.com`
- Screenshot: `dashboard_screenshots/<final_date>.png`
- Reports above
- `disconnect_drill.md` — proof of 5-min cut handled
- `restart_drill.md` — proof of kill -9 recovery

---

## 4. Scaled-down universe for paper

| Setting | Backtest | Paper |
|---|---|---|
| Universe | 528 tickers | **Top 50 by liquidity** |
| Notional per pair | $25k (vol-target) | **$5-10k** (paper margin friendly) |
| Total gross exposure | $2.5M | **$250-500k** |
| Capital required | $1M (3-5x leverage implicit) | **$100k paper** |

**Why scale down**:
- Alpaca paper allows ~2x margin (Reg T)
- 528-ticker universe needs prime-broker margin (not available in paper)
- 50 most-liquid maintains diversification while fitting paper margin

**Top-50 selection** (verify before Phase 1):
- Sort by Alpaca's average daily volume × price
- Filter only NYSE/NASDAQ common stocks (not ETFs, ADRs)
- Cross-reference with V4 backtest universe (must have data for last 252 trading days)

---

## 5. Pre-deployment checklist (hard gate before submit)

```python
@dataclass
class PreflightCheck:
    broker_auth_ok: bool          # 1. API key valid + balance reachable
    websocket_connected: bool      # 2. Live data flowing
    state_store_writable: bool     # 3. SQLite write test
    state_store_recoverable: bool  # 4. Read-after-write test
    disconnect_drill_passed: bool  # 5. 5-min cut handled (GRADED)
    reconcile_drill_passed: bool   # 6. Off-by-1 reconcile works
    idempotency_test_passed: bool  # 7. Duplicate order ID dedups
    regime_features_buildable: bool # 8. composite z-score computes
    dashboard_loads: bool          # 9. All endpoints respond
    cloud_deployment_ok: bool      # 10. Render dashboard accessible
```

All 10 must be True before deliverable submission.

---

## 6. Failure modes + mitigations (risk register)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Alpaca paper data quality issues | Medium | Medium | Cross-check vs Yahoo bars; fallback to backtest-quality bars |
| WebSocket reconnect storm (rate limits) | Low | High | Exponential backoff capped at 60s |
| State race condition (concurrent bar + reconcile) | Low | High | Single-threaded event loop + state mutex |
| Strategy doesn't trade in paper window (composite halts all) | Medium | Low | Document the halt as correct behavior; show halt mechanism in dashboard |
| Disconnect mid-order submission | Medium | High | Idempotent client_order_id + retry after reconnect |
| Render.com cold start at month-start cron | Medium | Medium | $7/month tier removes 15-min sleep; add warmup ping |
| Strategy actually loses money in paper (drift > backtest) | Likely | Low (graded on infra) | Document honestly in `live_vs_predicted.md` |

---

## 6.5. Live trade monitoring (full coverage)

Dashboard (§Phase 4 above) shows current state. Monitoring goes **beyond display** — it includes alerting, automated actions, and review workflows.

### 6.5.1 — Alert system (push notifications)

**Channel options**:
- **Discord webhook** (recommended — free, instant, mobile push via Discord app)
- Slack webhook (alternative)
- Email via SMTP (fallback for critical alerts)
- SMS via Twilio (overkill for paper, optional)

**Alert tiers** (per `live/monitor/alerts.py`):

| Severity | Trigger | Action | Channel |
|---|---|---|---|
| 🔴 **CRITICAL** | Broker auth fail / engine crash / margin call | Halt new entries + page user | Discord + Email |
| 🟠 **ERROR** | WebSocket disconnect >30s / order rejection / reconcile mismatch | Halt + Discord notify | Discord |
| 🟡 **WARN** | Drift > 15bps / unusual fill price / cron miss | Log + Discord (rate-limited) | Discord |
| 🟢 **INFO** | Successful fold rollover / month-start halt decision | Log only | (none) |

**Implementation** (Discord webhook):
```python
import requests
def alert(severity: str, message: str, dedupe_key: str = None):
    # Rate-limit duplicates: same dedupe_key within 5 min suppressed
    if recently_sent(dedupe_key, within_minutes=5):
        return
    payload = {"content": f"[{severity}] {message}"}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)
    log_to_db(severity, message, dedupe_key)
```

### 6.5.2 — Automated kill-switch

**Hard auto-halt conditions** (engine refuses new entries until manual restart):

| Condition | Threshold | Rationale |
|---|---|---|
| Cumulative drift loss | > 25% of cumulative gross PnL | Cost model has departed from reality |
| Daily drawdown | > 3% of account equity | Far beyond backtest max DD (1.8%) |
| Consecutive losing trades | > 10 in a row | Strategy may be broken |
| Reconcile mismatch | > 1 share unexplained | Possible state corruption |
| Stale data | bar gap > 5 minutes | Data quality degraded |

When triggered:
1. Halt all new entries immediately
2. Existing positions held (close at hard SL or natural exit)
3. CRITICAL alert sent
4. Engine state dumped to disk for forensic analysis
5. Manual operator approval required to resume

### 6.5.3 — Scheduled monitoring reports (auto-generated)

**Daily report** (cron 16:30 ET, end-of-day):
- `daily_summary_{YYYYMMDD}.md` generated automatically
- Sent to Discord as file attachment
- Contents: trades today (count, P&L), open positions, drift vs predicted, regime status, any alerts

**Weekly report** (cron Friday 17:00 ET):
- `weekly_summary_{YYYYWW}.md`
- Rolling metrics: 7-day Sharpe, win rate, drift trend
- Cumulative P&L vs backtest expected
- Alert summary

**Month-end report** (last trading day):
- Comprehensive review for the month
- Triggers next-month regime decision visibility
- Sent to Discord

### 6.5.4 — Trade-level journal

Every trade logged to `trade_journal.parquet` with:
- Entry: pair_id, ts, Z, β, notional, predicted cost
- Exit: ts, exit type (zero-cross / hard SL / EOM), realized cost
- Slippage: actual fill price vs decision price
- Drift attribution: bp difference per cost component (spread, impact, commission, borrow)

**Dashboard `/api/trades`** endpoint returns last N trades with attribution.

### 6.5.5 — Backtest vs live comparison (running metrics)

**Dashboard panel** showing rolling 21-day comparison:

| Metric | Backtest predicted | Live actual | Difference |
|---|---|---|---|
| Sharpe (21-day) | +1.28 (from backtest) | (calculated) | (delta) |
| Win rate | 65% | (calculated) | (delta) |
| Avg trade duration | 7 days | (calculated) | (delta) |
| Avg cost per trade | 22 bps | (calculated) | (delta) |
| Halt rate | 31% of months | (calculated) | (delta) |

If `|live - backtest|` exceeds 2σ on any metric → WARN alert.

### 6.5.6 — Anomaly detection (basic)

Per-trade sanity checks:

| Check | Threshold | Action |
|---|---|---|
| Fill price vs decision price | > 50 bps slip | Flag trade in journal |
| Trade size vs L1 depth | > 80% of L1 | Reject entry, log |
| Order latency | > 2s | Flag, may indicate broker issue |
| Borrow fee unexpected | > 100 bps/yr | Halt that pair |

### 6.5.7 — Monitoring infrastructure files

```
live/monitor/
├── alerts.py              # Discord webhook + tier routing
├── kill_switch.py         # Auto-halt logic
├── reports.py             # Daily/weekly/monthly auto-generation
├── trade_journal.py       # Per-trade logging
├── drift_monitor.py       # Realized vs predicted cost
├── anomaly_detector.py    # Per-trade sanity checks
└── live_vs_backtest.py    # Rolling metric comparison
```

### 6.5.8 — What user monitors actively (vs auto)

| Frequency | Channel | What user looks at |
|---|---|---|
| **Real-time** | Discord push | Critical/Error alerts |
| **Daily** | Dashboard + daily report email | P&L, positions, drift, regime |
| **Weekly** | Weekly report + dashboard screenshot | Trend metrics, kill-switch status |
| **Monthly** | Month-end report | Performance attribution, lessons |

User does NOT need to watch dashboard during market hours — alerts surface anomalies. Dashboard is for **forensic review** + **screenshot deliverable**.

---

## 7. Out of scope (deferred to phase 2 / Week 7+)

- True production trading (real $)
- Multi-broker support
- Sub-minute bar resolution
- HMM regime detector (rejected during backtest research)
- Kalman β (rejected during backtest research)
- Crypto universe (V6, future)
- Slack/email alerts (optional polish)

---

## 8. File structure

```
Week 6/
├── live/                                 # NEW for live trading
│   ├── broker/
│   │   ├── alpaca_client.py             # auth, REST wrappers
│   │   └── websocket_handler.py         # stream + reconnect logic
│   ├── execution/
│   │   ├── order_manager.py             # idempotent submit, fill tracking
│   │   └── reconciliation.py            # broker vs local mismatch
│   ├── engine_live/
│   │   ├── live_pair.py                 # streaming version of run_pair_daily
│   │   ├── z_tracker.py                 # rolling Z state
│   │   └── regime_check.py              # composite filter live
│   ├── state/
│   │   ├── schema.sql                   # SQLite tables
│   │   ├── persist.py                   # SQLite read/write
│   │   └── recovery.py                  # restore from disk
│   ├── monitor/                          # §6.5 — beyond dashboard display
│   │   ├── alerts.py                    # Discord webhook + tier routing
│   │   ├── kill_switch.py               # Auto-halt logic (drawdown / drift / reconcile)
│   │   ├── reports.py                   # Daily/weekly/monthly auto-generation
│   │   ├── trade_journal.py             # Per-trade logging with attribution
│   │   ├── drift_monitor.py             # Realized vs predicted cost
│   │   ├── anomaly_detector.py          # Per-trade sanity checks
│   │   └── live_vs_backtest.py          # Rolling metric comparison
│   ├── dashboard/
│   │   ├── server.py                    # FastAPI app
│   │   ├── templates/                   # HTMX HTML
│   │   └── static/                      # CSS
│   ├── drills/
│   │   ├── test_disconnect_5min.py      # GRADED drill
│   │   ├── test_restart_recovery.py
│   │   └── test_idempotency.py
│   ├── preflight.py                     # Hard gate (PreflightCheck)
│   └── main.py                          # entry point: live trading loop
├── render.yaml                          # Render.com deployment config
└── requirements-live.txt                # additional deps for live
```

---

## 9. End-state deliverable form

```
results_week6_live/
├── dashboard_url.txt                    # https://<app>.onrender.com
├── dashboard_screenshots/
│   ├── 20260601_0930.png                # market open
│   ├── 20260601_1200.png                # midday
│   └── 20260601_1600.png                # close
├── disconnect_drill.md                  # 5-min cut proof
├── restart_drill.md                     # kill -9 proof
├── trade_log.parquet                    # all trades during window
├── drift_report.md                      # per-bucket drift analysis
├── live_vs_predicted.md                 # honest post-mortem
└── regime_card.md                       # which regime, halt history
```
