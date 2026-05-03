# Pipeline Tuần 4: Cointegration Pairs Trading — Multi-Regime Defense

## 0. Mục tiêu tuần này

Defend trước AI Investment Committee một strategy cointegration pairs trading hoạt động được qua **cả** Bear (2022) và Bull (2023–2026). Deliverable: Strategy Whitepaper.

**Câu hỏi cốt lõi:** Pairs cointegrate trong 2022 có generalize sang 2023–2026 không? Strategy có robust qua regime transition không?

**Dữ liệu:** 03/01/2022 – 19/03/2026, 1-min OHLCV, S&P 500 universe.

---

## Phase 0 — Data Quality Gateway

Module standalone, vectorized. Mọi phase downstream phải đọc data đã pass gateway này.

### 0.1 Pipeline chính

1. **Timestamp:** UTC ns → `US/Eastern` (DST tự động qua `pytz`)
2. **Session filter:**
   - Phase 1 (cointegration): 09:35–15:55 ET
   - Phase 2 (execution): 09:30–15:59 ET
3. **Resample:** Phase 1 → 5-min `.last()` cho price, `.sum()` cho volume; Phase 2 → 1-min raw

### 0.2 Outlier Treatment (Z-Score Based)

Áp dụng trên 1-min minute-level returns trước khi resample:

```
return_t = ln(close_t / close_{t-1})
Z_t = (return_t - rolling_mean) / rolling_std    # rolling window = 1 day
```

**Rule:**
- |Z_t| > 10 → flag as bad print → set return_t = NaN
- NaN → forward-fill với limit=1 bar
- Per ticker: nếu fraction outlier > 1.0% trong toàn sample → drop ticker hoàn toàn

### 0.3 Hard Assertions (fail-fast)

- Monotonic timestamps
- No duplicate (ticker, timestamp) rows
- OHLC valid: O ∈ [L, H], C ∈ [L, H], H ≥ L
- Non-negative price và volume
- Volume không all-zero trong session (ngoại trừ market-halt days)

### 0.4 Bad-Data Flags (log only, không drop)

- Stale price: ≥10 bars liên tiếp identical close
- Intra-session gap: missing bar GIỮA session
- Cross-ticker freeze: ≥30% tickers đứng giá cùng timestamp → market-halt day
- Volume-price coherence: |vol Z| > 10 với |return Z| < 1 → suspect tick

### 0.5 Output

```
data/validated/
├── 5min_phase1.parquet   # 09:35–15:55 ET, log-prices + volume, post-outlier
├── 1min_phase2.parquet   # 09:30–15:59 ET, OHLCV, post-outlier
└── meta_flags.parquet    # bad-data audit trail
```

KHÔNG thực hiện universe-wide inner join ở Phase 0. Data join logic ở Phase 1 (pairwise) — xem 1.2.1.

---

## Phase 1 — Cointegration Discovery

**Input:** 5-min log-prices + volume từ Phase 0.

### 1.1 Universe Hard Screens

Áp dụng độc lập từng ticker trên formation window:
- Median price ≥ $5
- Average daily dollar volume ≥ $1M (computed: $\overline{\text{ADV}}_\$ = \overline{\text{close} \times \text{volume}}$)
- Completeness ≥ 90%
- Zero-return fraction < 50%

→ Survivor universe count cần verify empirically lần đầu chạy. Document actual count và identify which filter binds. Initial estimate: ~400-500 tickers (S&P 500 mostly passes).

**Survivorship bias:** Universe = **point-in-time S&P 500 membership** (constituents tại từng ngày, không phải current list). Tickers bị delist (vd FRC, SBNY tháng 5/2023) → close position tại last available price, exit reason = `"delisted"`. Nếu data source chỉ có current list → acknowledge là known limitation trong whitepaper Discussion.

### 1.2 Pair Formation

All-pairs trong survivor universe (~80k–125k candidates).

KHÔNG filter, bucket, hay tag theo volume ở phase này. Volume-related analysis hoàn toàn ở Phase 4 (post-hoc stratification).

### 1.2.1 Pairwise Inner Join với Min-Overlap

Mỗi pair (A, B) trước khi đi vào cointegration test:

```
df_pair = inner_join(df_A, df_B, on='timestamp')   # chỉ 2 columns
overlap_ratio = len(df_pair) / len(formation_window_bars)

if overlap_ratio < 0.80:
    skip pair (insufficient overlap)
else:
    proceed to 1.3
```

Pairwise (not universe-wide) để retain data; Johansen cần T×2 per pair, không cần T×N rectangular matrix.

### 1.3 Hedge Ratio: PCA (Secondary Eigenvector)

Trên formation window (post inner-join), β qua PCA của centered log-price matrix:

```
X = [ln(A) - mean(ln(A)),  ln(B) - mean(ln(B))]    # T×2 centered
Cov = X^T @ X / (T-1)
eigenvalues, eigenvectors = eig(Cov)              # sorted descending

# Cointegrating direction = SECONDARY eigenvector (orthogonal to common trend)
v_2 = eigenvectors[:, 1]    # secondary

β_PCA = -v_2[0] / v_2[1]
α_PCA = mean(ln(A)) - β_PCA × mean(ln(B))
```

**Lý do secondary eigenvector:**
- Dominant eigenvector v_1 ≈ common-trend direction, capture variance lớn nhất
- Secondary eigenvector v_2 ⊥ v_1, là cointegrating direction (linear combination cho stationary residual)
- Đây là Avellaneda-Lee convention

**Caveat:** PCA = TLS chỉ khi noise variances bằng nhau giữa A và B. Với pairs có variance ratio lệch lớn (>5×), β có bias.

### 1.4 Cointegration Test: Johansen

- Trace statistic + max eigenvalue, k=1
- Bỏ Engle-Granger (asymmetric, double-test bias)

### 1.5 BH-FDR Multiple Testing Correction

q = 0.05 trên Johansen p-values toàn bộ tested pairs trong fold đó.

### 1.6 OU Half-Life Filter

Fit Ornstein-Uhlenbeck trên spread:
$$dS_t = \theta(\mu - S_t)dt + \sigma dW_t$$

Half-life raw = $\ln(2)/\theta$ trong units của 5-min bars.

```
half_life_days = (ln(2) / θ_per_5min_bar) / 78    # 78 = 5-min bars per day
```

**Range:** [1, 10] trading days.

Lower bound 1 day (78 5-min bars) đảm bảo Z-window đủ samples cho std estimator (sampling error <12%). [0.5, 10] trước đó quá aggressive — std estimator trên 39 bars có sampling error ~16% → Z whipsaw.

### 1.7 Surviving Pairs

Pass tất cả 1.1 → 1.6 → enter execution universe cho fold đó.

No cap trên số surviving pairs. Fallback: nếu fold compute time > X giờ, cap top-K theo Johansen p-value.

Discarded: GICS sector filter, economic logic filter (purely quantitative).

---

## Phase 2 — Signal Generation & Execution

**Input:** 1-min log-prices + surviving pairs từ Phase 1.

### 2.1 Kalman Filter Setup (2D State)

State vector: $\theta_t = [\alpha_t, \beta_t]^T$

**State equation:**
$$\theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$Q = \delta \cdot I_2$$

**Observation equation:**
$$\ln(A_t) = \alpha_t + \beta_t \ln(B_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R)$$

**Init:**
- $\theta_0 = [\alpha_{PCA}, \beta_{PCA}]^T$ từ Phase 1.3
- $R$ = realized residual variance từ PCA fit trên formation window
- $P_0 = R \cdot I_2$ (**intentional informative prior** — trust PCA estimates ngay từ đầu; nếu muốn vague prior dùng $P_0 = 1000 \cdot R \cdot I_2$, filter sẽ mất nhiều bars hơn để converge)

**Lý do thêm α:** Cointegration của log-prices có level offset non-zero. Nếu omit α, spread $\ln(A_t) - \beta_t \ln(B_t)$ có mean ≠ 0 và mean drift theo β → rolling Z absorb cái drift này → bias signal.

**Lý do spec R explicit:** Filter dynamics điều khiển bởi ratio δ/R, không phải δ một mình. Selection rule (2.1.1) thực ra select δ/R ratio.

### 2.1.1 Auto-Selection Rule cho δ (PRE-REGISTERED, MULTI-CRITERION)

Trên formation window của mỗi fold:

```
For each δ in {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}:
    For each surviving pair:
        Fit Kalman với (δ, R) trên formation
        Compute prior spread series
        Compute kurtosis(spread_prior)
        Compute half_life(spread_prior) via OU fit
        Compute lag1_ACF(spread_prior)

    metric_kurt[δ]  = median(|kurtosis_pair - 3|) across pairs
    median_HL[δ]    = median(half_life_pair) across pairs
    median_ACF78[δ] = median(|ACF(spread_prior, lag=78)|) across pairs
                      # lag=78 = 1 trading day trên 5-min data

# Multi-criterion selection
optimal_δ = argmin{ metric_kurt[δ] }
            subject to:
              median_HL[δ]    ∈ [1, 10] trading days
              median_ACF78[δ] > 0.7    # > 0.5 redundant với HL≥1d; 0.7 ≈ HL > 1.66d, binds independently
```

**Lý do dùng lag-78 thay vì lag-1:**
- Lag-1 ACF của OU process trên 5-min bars ≈ exp(−θ·Δt). Với HL ∈ [1, 10] days:
  `HL=1d → ACF(lag-1) ≈ 0.991`, `HL=10d → ACF(lag-1) ≈ 0.999`
- ACF lag-1 luôn > 0.99 trong vùng HL hợp lý → threshold 0.3 không bind, constraint vô nghĩa
- Lag-78 (= 1 ngày) cho range [0.5, 0.93] → threshold 0.5 có bind, discriminate pairs ổn vs noisy
- ACF78 > 0.5 ≈ exclude pairs có HL < 0.83 day (dưới lower bound của 1.6, consistent)

**Lý do multi-criterion:**
- Kurtosis = 3 alone không discriminate signal vs white noise (cả hai đều có kurt=3)
- Half-life constraint đảm bảo spread vẫn mean-reverting
- ACF78 constraint đảm bảo spread có temporal structure tại time-scale 1 ngày

**Edge cases auto-flag:**
- Optimal δ ở biên grid → expand grid cho fold tiếp theo
- Không có δ nào pass cả 3 constraints → kick fold's universe, log warning
- δ jump >2 grid steps consecutive folds → parameter instability flag

δ được re-select mỗi fold (không carry forward), log vào audit trail.

### 2.2 Spread Construction (Kalman PRIOR State)

Mỗi bar t:
1. **Predict:** $\hat{\theta}_{t|t-1}$ từ $\hat{\theta}_{t-1|t-1}$
2. **Prior spread:** $S_t^{prior} = \ln(A_t) - \hat{\alpha}_{t|t-1} - \hat{\beta}_{t|t-1} \cdot \ln(B_t)$
3. Z-score = (S_t^prior − rolling_mean) / rolling_std
4. Generate signal từ Z-score
5. **Update:** $\hat{\theta}_{t|t}$ sau khi quan sát A_t
6. $\hat{\theta}_{t|t}$ dùng cho sizing và threshold rebalance

**Critical:** Spread tính từ PRIOR (chưa nhìn A_t) → genuine OOS, kurtosis ~3–5. Posterior kéo về 0 → kurtosis 13.49 (leptokurtic, untradeable).

### 2.3 Z-Score Window (Unit Conversion Explicit)

```
half_life_days        = (ln(2) / θ_per_5min_bar) / 78
Z_window_phase2_bars  = half_life_days × 390    # 390 = 1-min bars per day
```

Cap 2,000 bars, burn-in = window // 2.

### 2.4 State Machine

```
state = 0 (flat)
if Z > +Z_entry:  state = -1 (short A, long B)
if Z < -Z_entry:  state = +1 (long A, short B)
if state ≠ 0 và |Z| crosses 0:  state = 0 (exit)
no re-entry until cross zero
```

**Default:** Z_entry = ±2.0. Sensitivity: {1.75, 2.0, 2.25}.

**Max holding period:** Default = end-of-session (force flatten 15:55 ET). Sensitivity: {EOS, 1d, 3d, 5d}.

### 2.5 Microstructure Safeguards

- Session warmup: 30 bars đầu mỗi session = NaN
- Execution lag: `position_executed[t] = position[t-1]`
- NaN handling: `signal_valid` mask thay vì dropna
- Engine: Numba `@njit(cache=True, fastmath=False)`

### 2.6 Position Sizing & Threshold Rebalance

**Dollar normalization:**
```
N_open_pairs_max = 50   # pre-defined cap, sensitivity: {20, 50, 100, uncapped}
per_pair_dollar  = total_capital / N_open_pairs_max

long_notional    = per_pair_dollar × 0.5
short_notional   = per_pair_dollar × 0.5

shares_A = long_notional / price_A(entry)
shares_B = (short_notional × β̂(entry|entry)) / price_B(entry)
```

Dollar-neutral: $0.5 long A + $0.5 × β̂ short B. N_open_pairs_max cap prevents 1/N → 0 khi nhiều pairs mở đồng thời.

**Threshold rebalance rule:** β̂ tại entry KHÔNG hard-locked. Mỗi bar trong khi position open:

```
β_drift = (β̂(t|t) - β_ref) / β_ref

if |β_drift| > X% AND position_open:
    # Compute delta units of B to rebalance
    delta_β        = β̂(t|t) - β_ref
    delta_shares_B = |delta_β| × (short_notional / price_B(entry))
                     # short_notional = 0.5 × per_pair_dollar, scale bằng price_B
                     # KHÔNG dùng shares_A (sẽ sai khi price_A ≠ price_B)

    # Cost: one-side commission (30bps) on delta only, NOT round-trip
    rebalance_cost = 0.30% × delta_shares_B × price_B(t)

    # Update reference with hysteresis dead band = X/2%
    β_ref ← β̂(t|t)
    # Next trigger only when |drift from new β_ref| > X%
    # Drift < X/2% from β_ref → ignore (dead band prevents thrash near boundary)
```

**Default X = 10%.** Sensitivity grid (Phase 4.5): {5%, 10%, 20%, ∞}. X = ∞ tương đương hard-lock.

**Lý do threshold thay vì hard-lock:**
- Hard-lock mâu thuẫn premise của Kalman (Kalman estimate β(t) là dynamic — nếu trust nó cho signal thì cũng phải trust cho hedge)
- Threshold capture cost-saving cho 80–90% trades (β drift nhỏ), vẫn re-hedge khi material drift
- Hysteresis (dead band X/2%) prevent β oscillating near boundary → thrash cost
- Defensible khi auditor hỏi "tại sao dùng Kalman nếu lock β?"

---

## Phase 3 — Backtest & Validation Framework

### 3.1 Validation Architecture: Rolling Walk-Forward

| Parameter | Value |
|---|---|
| Formation window | 6 months (~9,800 bars 5-min) |
| Trading window | 1 month (~7,800 bars 1-min) |
| Roll frequency | Monthly |
| Embargo gap | max(1 day, 0.5 × Z_window) |
| Tổng folds | ~45 (07/2022 → 03/2026) |

Mỗi fold = full Phase 1 pipeline re-run trên formation window mới của fold đó.

**Frozen trong fold (research-level):** pair list, [α_PCA, β_PCA] init, Z thresholds, Kalman δ (auto-selected), OU half-life range, R (measurement noise).

**Updated trong fold (execution-level):** $\hat{\theta}(t)$ Kalman update mỗi bar, rolling Z stats, threshold rebalance khi β drift > X%.

**Boundary handling khi sang fold mới:**
- Open position từ fold cũ: carry forward đến natural exit. KHÔNG force flatten.
- Pair drop khỏi universe mới: position cũ chạy hết, KHÔNG mở entry mới.
- New pair: enable entry từ bar đầu fold mới.
- Kalman state: re-init từ $\theta_0 = [\alpha_{PCA}, \beta_{PCA}]$ của fold mới.

**Aggregation:** report cả concatenated equity curve VÀ fold-equal-weighted Sharpe distribution.

### 3.2 Transaction Cost Model

Static 30 bps entry + 30 bps exit = 60 bps round-trip.

**Honest framing:** 60 bps là **upper end of realistic intraday range** cho S&P 500 pairs. Literature intraday-specific (Frazzini-Israel-Moskowitz 2018; Engle-Ferstenberg-Russell) cho 8–50 bps round-trip range cho institutional execution. 60 bps phản ánh assumption kém thuận lợi (slippage cao, smaller institutional access). Strategy survive 60 bps = robust to high-cost regime, KHÔNG phải evidence của large gross alpha.

**Sensitivity grid:** {30, 45, 60, 75 bps}.

**Borrow cost (short leg):**
```
borrow_rate_bps_annual = 50    # default; sensitivity {30, 50, 100}
borrow_cost_daily = (borrow_rate_bps_annual / 10_000) / 252
                    × short_notional_$
```
Accrual: daily, trên short_notional tại close mỗi ngày. Charged ngay cả khi position không rebalance.

**Sensitivity grid:** {30, 50, 100} bps annual.

**Threshold rebalance cost:** charged explicitly mỗi lần re-hedge xảy ra (Phase 2.6). Cost decomposition trong reporting tách: (1) entry/exit commission, (2) borrow cost, (3) re-hedge cost from threshold breaches.

### 3.3 PnL Mechanics

- **Sharpe:** Daily returns × √252 (consistent với annualization convention)
- **Equity curve:** Bar-level (1-min) compound từ 1.0
- **MaxDD:** Computed trên bar-level equity (không phải daily MTM) để capture intraday drawdowns
- **Calmar:** CAGR / bar-level MaxDD

Daily MTM hide intraday DD cho 1-min strategy → Calmar daily MTM bias upward.

### 3.4 Performance Metrics

Per fold + aggregated: Sharpe (daily), MaxDD (bar-level), CAGR, Calmar, Win Rate, Trade Count, Avg Holding Time, Turnover.

### 3.5 Negative Control Validation

**Hai controls qua mọi fold:**
1. Empirical NC: CVNA/ISRG (known non-cointegrated trong 2022)
2. Synthetic NC: 2 random walks i.i.d. simulate

**Pass criterion — Bootstrap implementation:**
```
# Per fold: NC return series đã có (từ running NC qua fold)
# Bootstrap block resampling (block size = 1 day = 390 bars, preserve autocorrelation)
NC_bootstrap_sharpes = []
for k in range(1000):
    resampled_returns = block_bootstrap(NC_daily_returns, block_size=1)
    NC_bootstrap_sharpes.append(annualized_sharpe(resampled_returns))

threshold_2sigma = mean(NC_bootstrap_sharpes) + 2 × std(NC_bootstrap_sharpes)
```

Primary strategy Sharpe (aggregate across folds, concatenated) phải > `threshold_2sigma`.

**Lý do block bootstrap thay vì i.i.d.:** Daily returns của pairs strategy có serial correlation (open positions span nhiều ngày). Block bootstrap preserve structure này, tránh underestimate NC variance.

### 3.6 Latency Parameterization

Đo alpha decay khi execution delay tăng — kiểm tra signal sống nhờ cointegration thật hay microstructure noise.

- **Primary:** t+1 lag (baseline)
- **Sweep:** {t+2, t+5, t+10} bars
- **Stress:** random latency ~ Uniform(1, 5) bars (mô phỏng execution uncertainty)
- **Output:** alpha decay curve — trục x = latency (bars), trục y = Sharpe

**Pass criterion:** Sharpe ở t+5 vẫn dương VÀ degrade gracefully (không cliff). Nếu t+2 đã giết Sharpe → signal sống nhờ microstructure noise, không phải cointegration thật → fail.

**Implementation:** wrapper quanh execution engine, shift `position_executed[t]` thêm lag. Zero thay đổi signal logic. Chỉ run trên default config, không cross với OAT grid.

### 3.7 Red Flag Triggers (Auto-Audit)

| Trigger | Condition | Action |
|---|---|---|
| Lookahead leak | Sharpe > 5.0 | Halt, audit execution lag |
| NC discrimination fail | Primary Sharpe within 2σ của NC bootstrap | Halt, audit signal logic |
| Kalman spread degenerate | var(spread_prior) > 2× var(static_PCA_spread) OR < 0.1× | Flag δ too large or too small |
| δ boundary | Auto-selected δ ở biên grid | Expand grid next fold |
| δ instability | δ jump >2 grid steps consecutive folds | Flag parameter instability |
| Universal constraint fail | Không δ nào pass kurt+HL+ACF cho mọi pair | Kick fold universe |

### 3.8 Audit Log

Mỗi fold tạo `.txt` 7-section: parameter hash, selected δ + multi-criterion metrics (kurt/HL/ACF78), trade log sample, comparative metrics, timestamp verification proof, NC bootstrap results, red flag status, environment hash.

---

## Phase 4 — Multi-Regime Defense

### 4.1 Regime-Conditional Partition

Layer 1 folds tách theo regime:

| Regime | Fold range | N folds |
|---|---|---|
| Late Bear 2022 | Folds 1–6 (07/2022 – 12/2022) | 6 |
| Early Bull 2023 | Folds 7–18 (01/2023 – 12/2023) | 12 |
| Mid Bull 2024 | Folds 19–30 | 12 |
| Late Bull 2025–Q1 2026 | Folds 31–45 | 15 |

**Report:** mean/median Sharpe, % positive folds, IQR per regime.

**Caveat:** Bear sample N=6 under-power cho strong inference. Late 2022 = peak Bear stress — pairs identified trên distressed data có thể không cointegrate trong calmer regime ngay sau. Acknowledge trong Discussion section của whitepaper.

**Pass criterion:** Distribution stable across regimes. Nếu Sharpe collapse trong Bull → strategy bear-specific (honest finding, không phải failure).

### 4.2 Pair Persistence Test

1. Trên formation 6 tháng cuối 2022 (07/2022 – 12/2022, consistent với rolling architecture), identify set $\mathbf{P}_{2022}$ pass tất cả Phase 1 filters
2. Tại mỗi formation window của Phase 3.1 từ fold 7 trở đi (01/2023+), re-test Johansen mỗi pair $\in \mathbf{P}_{2022}$ (chỉ đo persistence của P_2022, không re-discover)
3. **Output:** line chart "% pairs trong $\mathbf{P}_{2022}$ còn pass Johansen tại t" theo thời gian

**Mục đích:** Đo cointegration decay rate empirically.

### 4.3 Volume Sensitivity Analysis (Hussein-Inspired)

**Reframed từ "Hussein replication" — methodology divergent từ Hussein, không phải replication chính xác.**

| Aspect | Hussein 2025 | This pipeline |
|---|---|---|
| Volume measure | Share volume | **Share volume (matched)** |
| Bucket granularity | Decile (~50 stocks) | Tertile (~100–150 stocks, smaller universe) |
| Pair formation | Within-decile only | **Within-tertile only (matched design)** |
| Universe | Full S&P 500 | Survivor (post 1.1 filter) |
| Time scale | Daily, 1-month hold | Intraday 5-min, max EOS |

**Compute volume metadata per fold:**
1. Trên formation window của fold đó, compute **share-volume ADV** (raw share count per day, average) cho mỗi ticker trong survivor universe
2. Phân tertile T1 (low) / T2 / T3 (high) trên distribution share-volume ADV
3. **Within-tertile pair formation only:** tag pairs T1-T1, T2-T2, T3-T3. KHÔNG analyze cross-tertile pairs (không có baseline trong Hussein).

**Stratify Layer 1 fold results:**
- Sharpe distribution per same-tertile bucket
- Compare T3-T3 vs T1-T1 — đây là Hussein's main test (high-vol vs low-vol decile)

**Outcome chấp nhận được:** hold (T3-T3 > T1-T1), không hold, hoặc partial — cả 3 đều là contribution. Document explicit trong whitepaper rằng đây là OUT-OF-DOMAIN extension, không phải replication.

**Cross-tertile pairs:** defer to Week 5+ as separate exploratory experiment.

### 4.4 Overfitting Diagnostics

- **Deflated Sharpe Ratio (DSR):** Bailey & López de Prado
- **Probability of Backtest Overfitting (PBO):** k-fold combinatorial path testing

### 4.5 Sensitivity Grid (Pre-Registered, OAT Strategy)

**Vấn đề full-grid:** 9 parameters × multiple values = 20,000+ configs × 45 folds = infeasible. Thay bằng **One-At-a-Time (OAT)** quanh default, cộng 3 combo runs cho interaction check.

**Default config (anchor):**

| Parameter | Default |
|---|---|
| Formation window | 6m |
| Trading window | 1m |
| Z entry | ±2.0 |
| Kalman δ | auto (multi-criterion) |
| Transaction cost | 60 bps |
| Borrow rate | 50 bps/year |
| Stop-loss | None |
| Threshold rebalance X% | 10% |
| Max holding | EOS |
| N_open_pairs_max | 50 |

**OAT sweeps — vary one parameter, hold all others at default:**

| Parameter | Values swept |
|---|---|
| Formation window | {3m, **6m**, 9m} |
| Trading window | {2w, **1m**, 6w} |
| Z entry | {±1.75, **±2.0**, ±2.25} |
| Transaction cost | {30, 45, **60**, 75 bps} |
| Borrow rate | {30, **50**, 100 bps/year} |
| Stop-loss | {**None**, -2.5%, -5%} |
| Threshold rebalance X% | {5%, **10%**, 20%, ∞} |
| Max holding | {**EOS**, 1d, 3d} |
| N_open_pairs_max | {20, **50**, 100} |

→ 9 params × ~3 values mỗi param = ~27 additional runs × 45 folds = ~1,215 fold-runs. Feasible.

**3 Interaction Combo Runs (targeted):**
1. **Tight signal:** Z=1.75 + formation=3m + SL=-2.5% (test aggressive entry trong short window)
2. **Conservative signal:** Z=2.25 + formation=9m + SL=None (test patience strategy)
3. **High cost stress:** TC=75bps + borrow=100bps + X%=5% (test cost sensitivity under churn)

**Decision rule:** Final config = median net Sharpe across folds của default run, KHÔNG pick best OAT config. OAT chỉ để report robustness, KHÔNG để tune.

### 4.6 Stop-Loss Robustness Test (Smoothed)

Per-pair cumulative return từ entry, smoothed để tránh whipsaw trên 1-min bars:

```
rolling_pnl_5bar = rolling_mean(pair_pnl_pct, window=5)

if state ≠ 0:
    if rolling_pnl_5bar < SL_threshold:
        state = 0
        exit_reason = "SL"
```

Smoothing window K=5 bars (default). Sensitivity: K ∈ {3, 5, 10}.

**Lý do smoothing:** Per-bar SL trên 1-min trigger trên noise. Hussein apply SL trên DAILY cum return (smoother bậc magnitude). 5-bar rolling mean filter intra-trade noise nhưng vẫn react với material drawdown.

**Reporting:**
- Sharpe distribution với mỗi SL config (None, -2.5%, -5%) × K {3, 5, 10}
- Exit reason breakdown: % zero-cross / % SL / % max-holding / % EOS
- Average return per exit reason

**KHÔNG tuyên bố tương đương Hussein Figure 23.** Mapping exit reasons không 1-1.

---

## Reporting Standards (Whitepaper §-by-§)

| § | Content |
|---|---|
| 1 | Layer 1 main result: 45-fold Sharpe distribution (mean, median, IQR, % positive, min, max) |
| 2 | Regime breakdown: Sharpe per regime (4.1) — caveat Bear N=6 |
| 3 | Pair persistence decay curve (4.2) |
| 4 | Volume sensitivity analysis: Hussein-inspired, methodology divergence documented (4.3) |
| 5 | Overfitting: Raw Sharpe + DSR + PBO (4.4) |
| 6 | OAT sensitivity: robustness across parameters (4.5) + 3 combo runs |
| 7 | Negative control: empirical + synthetic, bootstrap distribution (3.5) |
| 8 | Latency stress test: alpha decay curve (3.6) |
| 9 | Stop-loss robustness + exit reason breakdown (4.6) |
| 10 | Cost decomposition: gross → commission → borrow (50bps default) → threshold-rebalance → net |
| 11 | Auto-selected δ trajectory across folds + multi-criterion metrics (kurt/HL/ACF78) |
| 12 | Universe count verification: actual survivor count, binding filter |

---

## Critical Lock-Ins

1. **PCA cho cointegration test, Kalman cho execution** (different roles)
2. **PCA convention: secondary eigenvector** cho cointegrating direction
3. **Kalman 2D state [α, β]** với measurement noise R explicit
4. **Kalman PRIOR state cho spread, POSTERIOR θ cho sizing**
5. **Kalman init [α_PCA, β_PCA]** từ Phase 1.3
6. **Threshold rebalance** (default 10% drift), KHÔNG hard-lock
7. **Multi-criterion δ selection** (kurtosis + half-life + ACF78), pre-registered
8. **Monthly rolling walk-forward**, không static split
9. **Volume = post-hoc stratification tại Phase 4** (Hussein-inspired, not replication), KHÔNG pre-filter
10. **60 bps round-trip = upper-realistic, not conservative anchor** + borrow 50 bps/year default
11. **No frozen 39-month anchored test** — replaced by regime-conditional partition (4.1) + persistence test (4.2)
12. **No pair count cap** trừ khi compute blow up (fallback rule)
13. **Each fold re-runs full Phase 1 pipeline**
14. **Pairwise inner join với min-overlap 80%**, KHÔNG universe-wide inner join
15. **Bar-level MaxDD** cho intraday strategy
16. **Embargo = max(1 day, 0.5 × Z_window)**
17. **Half-life range [1, 10] days** (1 day lower bound đảm bảo Z-window sample size)
18. **Bootstrap NC threshold** thay vì fixed |Sharpe| < 0.5
19. **Dollar normalization per pair** = capital / N_open_pairs_max (default 50), KHÔNG "1 unit of ln(A)"
20. **Kalman degenerate check via variance ratio** (no double-run baseline) — var(prior) vs var(static PCA spread)

---

## Deferred to Week 5+

- **HMM regime-detection gateway** (Phase 2.5 layer)
- **Dynamic volatility-adjusted slippage**
- **Combinatorial Purged Cross-Validation as primary**
- **Hyperparameter freeze test** (specification robustness experiment)
- **Per-pair Kalman δ optimization** (bucket-level δ as middle ground)
- **Volume-aware execution gate** — motivated nếu Phase 4.3 cho thấy volume effect mạnh
- **2-stage δ grid refinement** (coarse → fine centered on best)
- **Cross-tertile pairs analysis** (T1-T2, T1-T3, T2-T3)
- **Universe size ablation** (relax Phase 1.1 cho Phase 4.3 specifically)
- **Quintile/decile bucket sensitivity** cho Phase 4.3 nếu universe đủ lớn
