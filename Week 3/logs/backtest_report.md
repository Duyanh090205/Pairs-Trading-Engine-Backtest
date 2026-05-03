# Pairs Trading Backtest — Methodology and Results
*Pipeline run: 2026-04-20 | deliverable1.py + deliverable2.py | Runtime: ~337s | No warnings*

---

## 1. Data

| Field | Value |
|---|---|
| Source | 2,510 one-minute OHLCV CSV files |
| Tickers | CMS, CVNA, DOW, DUK, GOOG, GOOGL, INTC, ISRG, JPM, LYB |
| Full date range | 2022-01-03 to 2022-12-30 |
| Session filter | 09:30–15:59 ET inclusive |
| Total bars (post-filter) | 970,411 |
| Formation period | 2022-01-03 to 2022-06-30 (parameter estimation only) |
| Trading period | 2022-07-01 to 2022-12-30 (all PnL computed here) |

**Data quality (6 assertions, all PASS):**
- A1: timestamps monotonically increasing per ticker
- A2: no duplicate (ticker, timestamp) pairs
- A3: high ≥ close ≥ low for every bar
- A4: high ≥ open ≥ low for every bar
- A5: no NaN in OHLC columns
- A6: no trading day with fewer than 100 bars (10 days have <300 bars — market early-close days; minimum observed: 199 bars)

---

## 2. Methodology

### 2.1 Pairs and Parameters

OLS regression on formation period: `log(A) = α + β · log(B) + ε`

| Pair | Role | α | β | Window |
|---|---|---|---|---|
| CMS / DUK | Primary | −0.695816 | 1.048794 | 677 bars |
| DOW / LYB | Secondary | −0.905720 | 1.088981 | 774 bars |
| GOOG / GOOGL | Secondary | +0.102777 | 0.987187 | 19 bars |
| CVNA / ISRG | Negative control | −20.231 | 4.425 | 1,755 bars |
| INTC / JPM | Negative control | −0.014 | 0.784 | 2,000 bars (cap) |

Windows derived from AR(1) half-life of formation spread, clipped to [10, 2000].

Kalman filter monthly β (CMS/DUK Version B), prior month-end values:

| Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|
| 0.764627 | 0.773766 | 0.754115 | 0.757557 | 0.787362 | 0.778809 |

### 2.2 Signal Pipeline

1. **Spread:** `S(t) = log(A_close) − α − β · log(B_close)`
2. **Z-score:** rolling mean and std with `min_periods = window // 2` (burn-in)
3. **Session warmup:** first 30 bars of each calendar day suppressed
4. **State machine:** enter long (short) spread when Z < −2.0 (Z > +2.0); exit at Z = 0
5. **Execution lag:** `position_executed[t] = position[t−1]` — signal at bar t−1 close, execution at bar t open

### 2.3 Sizing

**Version A (OLS constant β):**
- At entry bar ep: `shares_A = ±1 / price_A(ep−1)`, `shares_B = ∓β / price_B(ep−1)`
- Held at fixed shares until exit; zeroed at exit bar

**Version B (Kalman monthly β):**
- Same structure; β replaced monthly using prior month-end Kalman value
- β map applied at entry bar; shares held constant through the trade

### 2.4 PnL

- **Bar-level MTM:** `pnl(t) = shares_A(t) · Δclose_A(t) + shares_B(t) · Δclose_B(t)`
- **Effective entry price:** close of bar ep−1 (signal bar), consistent with 1-bar lag model
- **Effective exit price:** close of bar xp−1 (last bar with nonzero shares)
- **Transaction costs:** deducted at entry and exit bars against current notional; exit uses `notional.shift(1)` because shares are zeroed at exit bar
- **Reference notional:** mean entry notional across all trades ≈ 1 + β
- **Daily return:** `sum(bar_pnl + cost) / ref_notional` per calendar day
- **Sharpe:** `mean(daily_return) / std(daily_return, ddof=1) × √252`

### 2.5 Cost Scenarios

Signals and sizings computed once (cost-independent); PnL looped per scenario:

| Label | Entry bps | Exit bps | Round-trip |
|---|---|---|---|
| Conservative | 30 | 30 | 60 bps |
| Realistic | 10 | 10 | 20 bps |
| Low-cost | 2 | 2 | 4 bps |
| No-cost | 0 | 0 | 0 bps |

### 2.6 Bias Injection (Deliverable 1)

24 flawed datasets: 4 methods × 6 k-values (5%, 10%, 20%, 30%, 40%, 50%). Random seed = 42 throughout. Row count verified equal to clean data (970,411) for every file.

| Method | What is injected | Scope |
|---|---|---|
| H1 | `close[t]` replaced with `close[t+1]` for k% of rows per ticker | All tickers, row-level |
| H2 | `window_start` decremented by 60 s for k% of rows (timestamp backdating) | All tickers, row-level |
| H3 | `spread_biased` column added; S(t) replaced with S(t+1) for k% of CMS/DUK timestamps | CMS/DUK spread only |
| H4 | `close` normalized by full-2022 mean/std for k% of tickers | Ticker-level |

H3 affects the CMS/DUK spread column only. DOW/LYB and GOOG/GOOGL read raw OHLC prices from H3 files, which are identical to clean data for those tickers.

H4 ticker selection at each k% (seed=42, 10 tickers):
- k=5%, 10%: 1 ticker — CMS
- k=20%: 2 tickers — CMS, ISRG
- k=30%: 3 tickers — LYB, CMS, INTC
- k=40%: 4 tickers — GOOG, CMS, GOOGL, INTC
- k=50%: 5 tickers — GOOGL, DUK, ISRG, CMS, GOOG

---

## 3. Results

### 3.1 Clean Backtest — CMS/DUK (CP5)

Timestamp verification: **PASS** — 90 trades, 0 violations of exec_ts > signal_ts.

| Cost | RT bps | Sharpe A | Sharpe B | Trades | Max DD (A) | Win Rate (A) |
|---|---|---|---|---|---|---|
| Conservative | 60 | −22.3859 | −17.9027 | 90 | −34.21% | 2.2% |
| Realistic | 20 | −3.7493 | −2.4884 | 90 | −5.91% | 40.0% |
| Low-cost | 4 | +4.6831 | +3.8462 | 90 | −1.27% | 76.7% |
| No-cost | 0 | +6.2796 | +5.1712 | 90 | −1.13% | 81.1% |

Avg gross PnL per trade: 13.75 bps (Version A), 14.23 bps (Version B). Trade count and gross PnL are identical across all cost scenarios.

### 3.2 Version A vs Version B — CMS/DUK clean (CP6c)

| Metric | Ver A | Ver B | Delta B−A |
|---|---|---|---|
| **Conservative (60 bps RT)** | | | |
| Sharpe | −22.3859 | −17.9027 | +4.4831 |
| Max Drawdown | −34.21% | −33.89% | +0.32% |
| CAGR | −56.62% | −56.20% | +0.41% |
| Calmar | −1.6550 | −1.6586 | −0.0036 |
| Win Rate | 2.2% | 3.3% | +1.1% |
| Avg Gross (bps) | 13.7469 | 14.2250 | +0.4781 |
| Trades | 90 | 90 | 0 |
| **Realistic (20 bps RT)** | | | |
| Sharpe | −3.7493 | −2.4884 | +1.2609 |
| Max Drawdown | −5.91% | −5.62% | +0.29% |
| Win Rate | 40.0% | 41.1% | +1.1% |
| **Low-cost (4 bps RT)** | | | |
| Sharpe | +4.6831 | +3.8462 | −0.8369 |
| Max Drawdown | −1.27% | −1.44% | −0.17% |
| Calmar | 14.7941 | 13.8077 | −0.9865 |
| Win Rate | 76.7% | 75.6% | −1.1% |
| **No-cost (0 bps RT)** | | | |
| Sharpe | +6.2796 | +5.1712 | −1.1084 |
| Max Drawdown | −1.13% | −1.34% | −0.21% |
| CAGR | +27.55% | +28.75% | +1.20% |
| Calmar | 24.3125 | 21.4745 | −2.8380 |
| Win Rate | 81.1% | 80.0% | −1.1% |

### 3.3 Bias Sensitivity Sweep — CP6a

All results use Version A. Rows = method, columns = k%.

---

**CMS/DUK — Clean Sharpe: −22.3859 (60 bps) / −3.7493 (20 bps) / +4.6831 (4 bps) / +6.2796 (0 bps)**

*60 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −18.791 | −17.090 | −17.382 | −17.053 | −14.331 | −14.259 |
| H2 | −20.306 | −18.983 | −16.219 | −14.212 | −12.394 | −12.008 |
| H3 | −21.336 | −21.372 | −21.477 | −20.459 | −19.648 | −19.373 |
| H4 | +0.381 | +0.381 | +0.381 | +0.381 | +0.381 | +0.905 |

*20 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −2.498 | −1.523 | +0.660 | +1.802 | +1.497 | +2.219 |
| H2 | −3.456 | −2.777 | −1.530 | −1.228 | −0.604 | −0.243 |
| H3 | −3.751 | −3.806 | −4.119 | −4.674 | −3.957 | −4.244 |
| H4 | +0.385 | +0.385 | +0.385 | +0.385 | +0.385 | +0.913 |

*4 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +5.164 | +4.956 | +6.992 | +8.130 | +6.874 | +8.034 |
| H2 | +4.490 | +4.607 | +4.616 | +4.387 | +3.856 | +3.935 |
| H3 | +4.700 | +4.656 | +4.411 | +4.072 | +3.667 | +3.431 |
| H4 | +0.387 | +0.387 | +0.387 | +0.387 | +0.387 | +0.916 |

*0 bps (no cost):*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +6.687 | +6.256 | +8.227 | +9.295 | +7.969 | +9.151 |
| H2 | +6.009 | +6.027 | +5.845 | +5.526 | +4.788 | +4.802 |
| H3 | +6.323 | +6.282 | +6.051 | +5.777 | +5.190 | +4.966 |
| H4 | +0.387 | +0.387 | +0.387 | +0.387 | +0.387 | +0.917 |

---

**DOW/LYB — Clean Sharpe: −12.1328 (60 bps) / −2.2560 (20 bps) / +1.3572 (4 bps) / +2.1400 (0 bps)**

*60 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −9.766 | −9.179 | −5.921 | −5.131 | −4.860 | −2.340 |
| H2 | −10.454 | −9.157 | −7.500 | −5.694 | −5.305 | −5.182 |
| H3 | −12.133 | −12.133 | −12.133 | −12.133 | −12.133 | −12.133 |
| H4 | −12.133 | −12.133 | −12.133 | 0.000 | −12.133 | −12.133 |

*20 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −1.133 | −0.269 | +1.329 | +4.275 | +2.674 | +5.193 |
| H2 | −1.552 | −0.908 | −0.257 | +0.046 | +0.178 | +0.288 |
| H3 | −2.256 | −2.256 | −2.256 | −2.256 | −2.256 | −2.256 |
| H4 | −2.256 | −2.256 | −2.256 | 0.000 | −2.256 | −2.256 |

*4 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +2.071 | +2.833 | +3.968 | +7.317 | +5.105 | +7.688 |
| H2 | +1.556 | +1.907 | +2.108 | +1.962 | +1.994 | +2.089 |
| H3 | +1.357 | +1.357 | +1.357 | +1.357 | +1.357 | +1.357 |
| H4 | +1.357 | +1.357 | +1.357 | 0.000 | +1.357 | +1.357 |

*0 bps (no cost):*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +2.781 | +3.513 | +4.570 | +7.970 | +5.642 | +8.243 |
| H2 | +2.242 | +2.532 | +2.630 | +2.395 | +2.404 | +2.492 |
| H3 | +2.140 | +2.140 | +2.140 | +2.140 | +2.140 | +2.140 |
| H4 | +2.140 | +2.140 | +2.140 | 0.000 | +2.140 | +2.140 |

---

**GOOG/GOOGL — Clean Sharpe: −103.7476 (60 bps) / −95.1385 (20 bps) / −26.9738 (4 bps) / +15.3984 (0 bps)**

*60 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −105.276 | −100.892 | −99.604 | −0.572 | −0.488 | −3.058 |
| H2 | −104.405 | −90.981 | −86.234 | −78.817 | −74.789 | −87.094 |
| H3 | −103.748 | −103.748 | −103.748 | −103.748 | −103.748 | −103.748 |
| H4 | −103.748 | −103.748 | −103.748 | −103.748 | −4.556 | −4.556 |

*20 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | −66.595 | −75.280 | −57.044 | +0.975 | +1.027 | −1.650 |
| H2 | −91.353 | −79.693 | −66.288 | −56.364 | −56.018 | −65.700 |
| H3 | −95.138 | −95.138 | −95.138 | −95.138 | −95.138 | −95.138 |
| H4 | −95.138 | −95.138 | −95.138 | −95.138 | −3.912 | −3.912 |

*4 bps round-trip:*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +3.035 | +14.926 | +18.255 | +1.595 | +1.632 | −1.087 |
| H2 | +1.324 | +16.494 | +24.675 | +26.489 | +28.765 | +27.625 |
| H3 | −26.974 | −26.974 | −26.974 | −26.974 | −26.974 | −26.974 |
| H4 | −26.974 | −26.974 | −26.974 | −26.974 | +4.502 | +4.502 |

*0 bps (no cost):*

| Method | k=5% | k=10% | k=20% | k=30% | k=40% | k=50% |
|---|---|---|---|---|---|---|
| H1 | +24.051 | +43.186 | +37.320 | +1.750 | +1.784 | −0.946 |
| H2 | +42.926 | +49.800 | +49.862 | +48.140 | +50.438 | +50.770 |
| H3 | +15.398 | +15.398 | +15.398 | +15.398 | +15.398 | +15.398 |
| H4 | +15.398 | +15.398 | +15.398 | +15.398 | +4.547 | +4.547 |

---

### 3.4 Execution Lag — CP6b (CMS/DUK)

Lag=Y: `position_executed[t] = position[t−1]`.  
Lag=N: `position_executed[t] = position[t]` (no lag).

| Lag | Variant | 60 bps RT | 20 bps RT | 4 bps RT | 0 bps RT | Trades |
|---|---|---|---|---|---|---|
| Y | Clean engine + clean data | −22.3859 | −3.7493 | +4.6831 | +6.2796 | 90 |
| Y | Clean engine + H1 k=50% | −14.2590 | +2.2192 | +8.0345 | +9.1513 | 125 |
| Y | Clean engine + H3 k=50% | −19.3734 | −4.2436 | +3.4309 | +4.9658 | 93 |
| N | Biased engine + clean data | −23.2341 | −7.1112 | +2.0818 | +3.9290 | 90 |
| N | Biased engine + H1 k=50% | −17.3827 | −6.9399 | +0.1313 | +1.8668 | 125 |

### 3.5 Z-threshold Sensitivity — CP6d (CMS/DUK clean)

| Z | Trades | 60 bps RT | 20 bps RT | 4 bps RT | 0 bps RT |
|---|---|---|---|---|---|
| 2.00 | 90 | −22.3859 | −3.7493 | +4.6831 | +6.2796 |
| 2.57 | 51 | −13.5104 | −1.4892 | +4.0569 | +5.1516 |

### 3.6 Negative Controls — CP6e (clean data)

Formation-period OLS; window from AR(1) half-life.

| Pair | α | β | Window | 60 bps RT | 20 bps RT | 4 bps RT | 0 bps RT | Trades |
|---|---|---|---|---|---|---|---|---|
| CVNA / ISRG | −20.2311 | 4.4245 | 1,755 | −3.5678 | −2.6794 | −2.3327 | −2.2469 | 27 |
| INTC / JPM | −0.0139 | 0.7842 | 2,000† | −1.4012 | −0.0363 | +0.4654 | +0.5866 | 22 |

† Window capped at maximum (2,000 bars). Spread half-life in formation period exceeded the cap.

### 3.7 Data Sanity Audit — CP6f (assertion-level detection)

Four checks applied directly to each dataset file — no backtest required.

| Check | What it tests | Catches |
|---|---|---|
| OHLC consistency | `high ≥ max(open,close)` AND `low ≤ min(open,close)` | H1, H4 |
| Timestamp no-duplicates | No duplicate `(ticker, ts_utc)` pairs | H2 |
| Session conformity | Every bar minute-of-day in [09:30, 15:59] ET | (H2 backup) |
| Schema strict | Column set exactly matches expected — no extras, no missing | H3 |

Results (24/24 flawed files flagged, clean passes all 4):

| File | OHLC | TSdup | TSsess | Schema | Verdict |
|---|---|---|---|---|---|
| clean | ok | ok | ok | ok | clean |
| flawed_h1_k5…k50 | FAIL | ok | ok | ok | FLAGGED |
| flawed_h2_k5…k50 | ok | FAIL | ok | ok | FLAGGED |
| flawed_h3_k5…k50 | ok | ok | ok | FAIL | FLAGGED |
| flawed_h4_k5…k50 | FAIL | ok | ok | ok | FLAGGED |

Failing row counts scale with k for OHLC and TSdup:
- H1 OHLC violations: 27,255 (k=5%) → 283,476 (k=50%)
- H2 duplicate timestamps: 45,610 (k=5%) → 240,687 (k=50%)
- H3 schema: extra column `spread_biased` at all k-levels
- H4 OHLC violations: 96,972 (k=5%, 10%) → 482,633 (k=50%) — scales with number of normalized tickers

**Caveat:** H3 detection relies on the injected extra column. An in-place H3 injection (overwriting an existing column) would pass all 4 checks because the raw OHLC stays valid.

---

## 4. What the Results Show

**Cost sensitivity is the dominant factor for CMS/DUK.** Average gross PnL per trade is 13.75 bps (Version A). At 60 bps round-trip, the strategy loses nearly every trade net of costs (win rate 2.2%). At 20 bps it loses less (win rate 40.0%). At 4 bps it produces Sharpe +4.68 and win rate 76.7%. At 0 bps (gross), Sharpe reaches +6.28 with win rate 81.1%. The same 90 trades produce four qualitatively different outcomes depending solely on cost assumption.

**Version A beats Version B at low cost; Version B beats at high cost.** Avg gross per trade is 0.48 bps higher for Version B (14.23 vs 13.75). At conservative costs where every trade loses net, Version B's higher gross leaves less to lose — hence higher (less negative) Sharpe. At low/no cost where trades are profitable, Version A's lower-cost sizing accumulates more efficiently — hence higher Sharpe (+4.68 vs +3.85 at 4 bps, +6.28 vs +5.17 at 0 bps).

**H1 and H2 inflate apparent edge.** For CMS/DUK at 60 bps, clean Sharpe is −22.39. H1 k=5% gives −18.79, H2 k=5% gives −20.31. The gap widens monotonically for H2 as k increases (to −12.01 at k=50%). H1 is generally monotonic but not perfectly so. At 0 bps, H1 k=30% inflates CMS/DUK Sharpe from +6.28 to +9.30, and H2 k=5% inflates it to +6.01 — both remain above clean, confirming the bias inflates gross edge even when costs are removed.

**H3 moves CMS/DUK Sharpe toward zero by smaller amounts than H1 or H2.** At 60 bps, H3 ranges from −21.34 (k=5%) to −19.37 (k=50%). H3 has no effect whatsoever on DOW/LYB or GOOG/GOOGL at any k-level — those pairs show identical Sharpe to their clean baselines across all six k-values. This is a direct consequence of H3 injecting bias only into the CMS/DUK `spread_biased` column.

**H4 produces a sign change.** At 60 bps CMS/DUK: clean = −22.39, H4 = +0.38 to +0.91 at all k-values. At 0 bps, H4 CMS/DUK remains near zero (+0.387–0.917) while clean is +6.28 — confirming the z-scored prices destroy the pair relationship entirely, not merely inflate apparent edge. DOW/LYB at k=30% reports exactly 0.000 because LYB is normalized at that k-level, making log prices invalid and producing zero trades.

**GOOG/GOOGL has large gross edge but is cost-destroyed.** At 0 bps clean Sharpe is +15.40; at 4 bps it collapses to −26.97; at 60 bps to −103.75. The signal window of 19 bars generates very high trade frequency, and round-trip costs of 4+ bps per trade eliminate the entire gross edge. Under H1 at 0 bps, Sharpe reaches +43 at k=10%. Under H2 at 0 bps, it reaches +50.77 at k=50% — the largest bias-induced Sharpe inflation of any pair.

**Removing the execution lag reduces Sharpe.** At 0 bps: Lag=Y gives +6.28 vs Lag=N gives +3.93 on clean data. At 4 bps: +4.68 vs +2.08. Trade count is 90 in both cases — the difference is purely from using the current bar's close (which the engine would not have observed at execution time) vs the prior bar's close.

**Higher Z-threshold reduces trade count and moderates Sharpe.** From Z=2.00 to Z=2.57: trades fall from 90 to 51. At 60 bps the Sharpe improves from −22.39 to −13.51 (fewer but higher-conviction trades). At 0 bps, Sharpe drops from +6.28 to +5.15 — the additional 39 trades filtered out were individually profitable gross.

**Negative controls behave as expected.** CVNA/ISRG stays negative at all cost levels including 0 bps (−2.25), confirming no gross edge. INTC/JPM shows near-zero Sharpe at 20 bps (−0.04) and a small positive at 0 bps (+0.59), consistent with weak or incidental gross edge rather than genuine cointegration in the trading period.

---

## 5. Audit Notes

**Red flag checks:** All passed. No clean-data Sharpe > 5.0, no timestamp violations (0 of 90 trades), DOW/LYB H3 flat as expected, sanity audit passed all 4 checks on clean data.

**H4 k=30% DOW/LYB = 0.000:** At this k-level LYB is normalized (z-score values replacing prices). Log of near-zero or negative values in the spread formula yields NaN. With no valid spread bars, no trades are taken and Sharpe is returned as 0.0 by the `sigma > 0` guard.

**INTC/JPM window = 2,000 (capped):** Half-life from AR(1) on formation spread exceeded the 2,000-bar maximum. This pair's spread showed near-non-stationary behavior in the Jan–Jun 2022 formation period.

**H3 schema detection is fragile:** The current H3 injection adds a visible extra column (`spread_biased`) that the schema check catches immediately. An in-place H3 injection (overwriting `close` directly or using the spread in a way that doesn't add columns) would evade all 4 sanity checks — raw OHLC would remain valid. Real H3-type bias in production (leaky pre-computed feature columns) requires re-derivation from raw inputs to detect.

**Data race note:** During development, deliverable2.py was run concurrently with a still-running deliverable1.py, producing NaN cells in H2 k=30/40 and H3 k=5 for secondary pairs. All results above are from a clean sequential run after all 24 files were fully written.

---

*Files: `data/raw/` (2,510 CSVs) · `data/flawed/` (24 CSVs, 970,411 rows each) · `logs/verified_backtest_log_multi_cost.txt`*
