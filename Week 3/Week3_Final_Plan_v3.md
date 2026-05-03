# Week 3 — Final Implementation Plan v3
> **Deliverable 1:** Creating the "Bad Data" · **Deliverable 2:** Verified Backtest Engine
> Version: Final v3 — Engine, Signal, Sizing, PnL, Sharpe đã chốt hoàn toàn
> Dựa trên findings từ Week 1 (cointegration scan) và Week 2 (OLS + Kalman signal engine)

---

## Tổng quan tiến độ

| Checkpoint | Mô tả | Deliverable | Trạng thái |
|---|---|---|---|
| CP1 | Xác nhận dữ liệu gốc sạch | D1 | ⬜ |
| CP2 | Write-up: 4 ý tưởng + chọn implement | D1 | ⬜ |
| CP3 | Implement 4 flawed datasets + k% sweep | D1 | ⬜ |
| CP4 | Data pipeline chuẩn | D2 | ⬜ |
| CP5 | Signal engine + timestamp verification | D2 | ⬜ |
| CP6a | Sharpe sensitivity sweep: k% × 4 methods | D2 | ⬜ |
| CP6b | Engine-level injection (comparative) | D2 | ⬜ |
| CP6c | Net metrics: Sizing Cấp 1 vs Cấp 3 | D2 | ⬜ |
| CP6d | Threshold sensitivity: Z=2.0 vs Z=2.57 | D2 | ⬜ |
| CP6e | Negative control sanity check | D2 | ⬜ |
| CP7 | Verified Backtest Log (.txt) | D2 | ⬜ |

---

## Quyết định kiến trúc đã chốt

| Hạng mục | Quyết định |
|---|---|
| **Engine** | Custom vectorized — pandas + numpy + Numba state machine (kế thừa Week 2) |
| **Signal** | OLS Z-score, Z=2.0 fixed, state machine từ Week 2 — không thay đổi |
| **Sizing** | Cấp 1 (OLS β cố định) và Cấp 3 (Kalman β monthly rebalance) — chạy song song |
| **PnL method** | Cách C — Daily mark-to-market |
| **Sharpe** | Daily returns × √252 |
| **Transaction cost** | Split 30/30: deduct 30bps tại entry, 30bps tại exit |
| **Equity curve** | Cumulative product của daily returns, bắt đầu = 1.0 |

---

## Context từ Week 1 & 2

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Primary pair | CMS / DUK | Week 1 — EG t=−5.268 |
| Secondary pair | DOW / LYB | Week 1 — EG t=−4.861 |
| Negative control | CVNA / ISRG | Week 1 — EG t=+0.778 |
| OLS β (CMS/DUK) | 1.0487 | Week 2 — formation period only |
| OLS α (CMS/DUK) | −0.6956 | Week 2 — formation period only |
| Kalman β cuối Jun 2022 | 0.7629 | Week 2 — Section 8 |
| Kalman β cuối Dec 2022 | 0.7819 | Week 2 — Section 8 |
| Rolling window | 680 bars | Week 2 — window = clip(HL, 10, 2000) |
| Entry threshold (default) | Z = 2.0 | Week 2 — Section 6 |
| Entry threshold (adaptive) | Z = 2.57 | Week 2 — 95th pct \|Z\| |
| Session warmup | 30 bars/ngày | Week 2 — Section 5.2 |
| Execution lag | 1 bar | Week 2 — position_executed[t] = position[t−1] |
| Trades baseline (Z=2.0) | 90 | Week 2 — CMS/DUK trading period |
| Round-trip cost | 60 bps | 4 legs × 15 bps — split 30 entry + 30 exit |

---

## Dữ liệu

- **Columns:** `ticker, volume, open, close, high, low, window_start, transactions`
- **Timestamp:** `window_start` = Unix nanosecond → bắt buộc convert sang datetime ET
- **File structure:** `{TICKER}_{DATE}.csv` — một file mỗi cổ phiếu mỗi ngày
- **Session hợp lệ:** 09:30–15:59 ET (DST-aware)
- **Quy tắc vàng:** signal tính tại close bar `t` → chỉ execute tại bar `t+1`

---

---

# DELIVERABLE 1 — Tạo "Bad Data"

**Nguyên tắc cốt lõi:**
Bias phải nằm trong **file data**, không phải engine code.
Flawed dataset là file độc lập — ai cầm file đó chạy engine sạch cũng bị ảnh hưởng.
Inject **randomly** (k% rows/tickers, seed=42 cố định) để trông tự nhiên và reproducible.

---

## CP1 — Xác nhận dữ liệu gốc sạch

1. Convert `window_start` nanosecond → datetime ET
2. Timestamp tăng đơn điệu theo từng ticker
3. Không có cặp `(ticker, window_start)` trùng
4. `high ≥ close ≥ low` và `high ≥ open ≥ low` trên mọi bar
5. Không có NaN trong OHLC
6. Sau session filter, đủ bars mỗi ngày giao dịch

**Output cần ghi lại:** tổng file, tổng tickers, time range, 6 assertion results.

---

## CP2 — Write-up: 4 ý tưởng + implement

### Phần 1 — Định nghĩa look-ahead bias

Look-ahead bias xảy ra khi signal tại bar `t` dùng thông tin chỉ có thể biết được sau bar `t`. Strategy trông xuất sắc trong backtest vì đã "biết tương lai" — nhưng sụp đổ ngay khi live vì thông tin đó không tồn tại lúc đặt lệnh. Đây là loại bias nguy hiểm nhất: lookahead bias bị lộ ngay ngày đầu tiên go-live, khác với overfitting hay survivorship bias (phát hiện sau vài tuần).

---

### Phần 2 — 4 ý tưởng, xếp theo độ realistic

**Hướng 1 — Random future-close substitution**
*(Realistic: Cao)*

Cơ chế: Chọn ngẫu nhiên k% rows. Thay `close[t]` bằng `close[t+1]` của cùng ticker trong file CSV.

Tại sao realistic: Lỗi vendor data phổ biến nhất — buffer flush delay ghi nhầm close phút sau vào phút hiện tại. Cũng xảy ra khi dùng pandas `.shift(-1)` thay vì `.shift(1)`.

Detectability: Thấp — một dòng check `close[t] == close_original[t+1]` là phát hiện ngay.

Sharpe impact: Cao — trực tiếp cho signal biết giá sẽ đi đâu.

---

**Hướng 2 — Timestamp backdating**
*(Realistic: Trung bình)*

Cơ chế: Chọn ngẫu nhiên k% rows, trừ 60 giây (= 1 bar) khỏi `window_start` trong file. Bar của thời điểm `t` trông như `t−1`. Engine sort theo timestamp sẽ đọc data của `t` nhưng nghĩ đó là `t−1`.

Tại sao realistic: Xảy ra khi vendor dùng bar-open timestamp thay vì bar-close timestamp, hoặc timezone conversion off-by-one. Timestamp verification (`exec_ts > signal_ts`) vẫn pass — chỉ là timestamp sai.

Detectability: Cao — phải cross-check với nguồn data độc lập mới phát hiện.

Sharpe impact: Thấp — giá liền kề 1 phút thường rất gần nhau.

---

**Hướng 3 — Spread-level injection**
*(Realistic: Cao)*

Cơ chế: Tính spread `S(t) = log(CMS_t) − α − β·log(DUK_t)` cho toàn timeline. Chọn ngẫu nhiên k% timestamps, ghi `S(t+1)` vào column `spread_biased` tại row `t` trực tiếp trong file.

Tại sao realistic: Tấn công trực tiếp vào thứ signal engine consume. Raw OHLC vẫn hợp lệ — bias chỉ trong derived column. Xảy ra khi pre-computed feature column được cache với lag offset sai.

Detectability: Trung bình — phải recompute spread từ raw OHLC và compare.

Sharpe impact: Rất cao — inject trực tiếp vào signal input, bypass mọi tính toán trung gian.

---

**Hướng 4 — Full-dataset normalization leak**
*(Realistic: Rất cao — lỗi phổ biến nhất trong production pipeline)*

Cơ chế: Chọn ngẫu nhiên k% tickers. Với các tickers đó, normalize cột `close` bằng mean và std của **toàn bộ năm 2022** (bao gồm future Jul–Dec):

```
global_mean = mean(close_entire_2022)   ← bao gồm tương lai
global_std  = std(close_entire_2022)    ← bao gồm tương lai
close_biased[t] = (close[t] − global_mean) / global_std
```

Với (100−k)% tickers còn lại: giữ nguyên close gốc.

Tại sao realistic nhất: `StandardScaler().fit_transform(df['close'])` trông như data preprocessing bình thường. Nhưng fit trên toàn bộ column = mean/std được tính có bao gồm Jul–Dec, trong khi strategy chỉ được phép biết Jan–Jun. Đây là lỗi được documented rộng rãi nhất trong production ML pipeline — nhiều fund đã publish exceptional alpha rồi fail ngay khi live vì lý do này.

Tại sao khó detect nhất: Không có future price nào xuất hiện trực tiếp. Không có timestamp nào sai. Để phát hiện, auditor phải đọc code preprocessing và hỏi "window nào để tính statistics?" — không thể suy ra từ file data.

Sharpe impact: Trung bình — tích lũy qua nhiều trades, không lớn per-bar.

---

### Phần 3 — Implement cả 4, so sánh qua k% sweep

Mỗi hướng test tại 5 mức k% (10%, 20%, 30%, 40%, 50%). Tổng 20 flawed datasets. Core output là bảng Sharpe(k%) × method.

### Phần 4 — Dự đoán tác động

H1 và H3: Sharpe tăng gần tuyến tính với k% — mỗi injected bar trực tiếp cho signal biết giá tương lai. H2: slope thấp nhất vì giá liền kề ít thay đổi. H4: tăng phi tuyến — chậm ở k% thấp nhưng tích lũy khi nhiều tickers bị inject. Crossover point giữa H2 và H4 là kết quả thú vị nhất.

---

## CP3 — Implement 4 flawed datasets + k% sweep

**Thiết kế:** 4 methods × 5 k-values (10/20/30/40/50%) = 20 flawed datasets.
Random seed = 42 cố định trên tất cả runs.

Naming convention:
```
flawed_h1_k10.csv, flawed_h1_k20.csv, ..., flawed_h4_k50.csv
```

**Hướng 1:** Chọn k% rows ngẫu nhiên. Với mỗi row tại bar `t`, thay `close[t]` = `close[t+1]` cùng ticker. Bỏ qua bar cuối mỗi ticker (không có `t+1`).

**Hướng 2:** Chọn k% rows ngẫu nhiên. Trừ 60,000,000,000 nanoseconds khỏi `window_start`. Re-sort theo `(ticker, window_start)` sau inject.

**Hướng 3:** Tính `S_clean[t]` trước. Chọn k% timestamps ngẫu nhiên. Ghi `S_clean[t+1]` vào column `spread_biased` tại row `t`. Lưu file với cả OHLC gốc và column `spread_biased`.

**Hướng 4:** Chọn k% tickers ngẫu nhiên (không phải rows). Với mỗi ticker được chọn, tính `global_mean` và `global_std` trên toàn bộ năm 2022, normalize cột `close`. Tickers còn lại không đổi.

**Validate mỗi method trước khi lưu:**
- H1: `close_biased[t] == close_original[t+1]` trong k% rows injected
- H2: k% rows có timestamp lệch 60s khỏi vị trí tự nhiên
- H3: `spread_biased[t] == S_clean[t+1]` trong k% timestamps injected
- H4: `mean(close_biased) ≈ 0`, `std(close_biased) ≈ 1` trên tickers bị inject

---

---

# DELIVERABLE 2 — Verified Backtest Engine

---

## CP4 — Data pipeline chuẩn

1. Load tất cả CSV, concat một lần (không loop append)
2. Convert `window_start` nanosecond → datetime ET (DST-aware)
3. Sort theo `(ticker, window_start)`
4. 3 assertions: timestamp monotonic, no duplicate `(ticker, ts)`, OHLC valid
5. Filter session 09:30–15:59 ET

Output in ra: tổng bars, số tickers, time range, kết quả 3 assertions.

---

## CP5 — Signal engine + sizing + PnL engine + timestamp verification

### 5.1 Signal pipeline (kế thừa Week 2)

| Bước | Logic |
|---|---|
| Spread | `S(t) = log(CMS_t) − α − β·log(DUK_t)` · α=−0.6956, β=1.0487 |
| Spread (H3 only) | Đọc `spread_biased` column thay vì recompute từ raw price |
| Z-score | Rolling window 680 bars, ddof=1, burn-in = 340 bars đầu |
| Warmup | 30 bars đầu mỗi calendar session → Z = NaN |
| State machine | Entry: Z < −2.0 → LONG · Z > +2.0 → SHORT · Z crosses 0 → FLAT |
| NaN-safety | Z = NaN → giữ position hiện tại, không trigger exit |
| Execution lag | `position_executed[t] = position[t−1]` |
| PnL gating | `signal_valid` flag — không dùng `.dropna()` |

### 5.2 Sizing — 2 versions chạy song song

**Version A — Cấp 1: OLS β cố định**

```
β_hedge = 1.0487  (không đổi suốt Jul–Dec 2022)

Long spread entry:
  Long  1 unit notional CMS  →  shares = 1.0 / P_CMS_entry
  Short β_hedge units DUK    →  shares = 1.0487 / P_DUK_entry

Short spread entry:
  Short 1 unit notional CMS
  Long  β_hedge units DUK
```

**Version B — Cấp 3: Kalman β monthly rebalance**

```
Dùng Kalman β tại cuối tháng trước làm hedge ratio cho tháng hiện tại:

  Jul 2022: β_hedge = Kalman_β[end of Jun] ≈ 0.7629
  Aug 2022: β_hedge = Kalman_β[end of Jul]
  Sep 2022: β_hedge = Kalman_β[end of Aug]
  ...
  Dec 2022: β_hedge = Kalman_β[end of Nov]

Long spread entry:
  Long  1 unit notional CMS  →  shares = 1.0 / P_CMS_entry
  Short β_kalman units DUK   →  shares = β_kalman / P_DUK_entry
```

Signal (entry/exit timing) giống nhau cho cả hai versions.
Chỉ có hedge ratio (số shares DUK) thay đổi.

### 5.3 PnL engine — Daily mark-to-market (Cách C)

**Nguyên tắc:** Cuối mỗi ngày, mark tất cả open positions theo close price. Daily P&L = unrealized change trong ngày + realized PnL từ trades đóng trong ngày đó.

**Transaction cost — split 30/30:**

```
Khi ENTRY (bar t+1 open):
  entry_cost = 30 bps   (Leg 1 CMS: 15bps + Leg 2 DUK: 15bps)
  Daily_PnL[entry_day] −= 30bps

Khi EXIT (bar t+k+1 open):
  exit_cost  = 30 bps   (Leg 3 CMS close: 15bps + Leg 4 DUK close: 15bps)
  Daily_PnL[exit_day] −= 30bps

Tổng round-trip = 60 bps
```

**Công thức Daily PnL cho ngày d:**

```
Unrealized_change[d] =
  Σ position_executed[t] × (S(close_d) − S(close_{d−1}))
  cho mọi bar t trong ngày d mà trade đang open

Realized[d] =
  Σ [S(exit) − S(entry)]
  cho mọi trade đóng trong ngày d

Cost[d] =
  −30bps × (số trades mở trong ngày d)
  −30bps × (số trades đóng trong ngày d)

Daily_PnL[d] = Unrealized_change[d] + Realized[d] + Cost[d]
Daily_Return[d] = Daily_PnL[d] / Notional
```

**Equity curve:**

```
equity[0] = 1.0
equity[d] = equity[d−1] × (1 + Daily_Return[d])
```

### 5.4 Sharpe Ratio

```
Sharpe = mean(Daily_Return) / std(Daily_Return) × √252
```

Dùng √252 vì đã aggregate về daily returns — không phải √(252×390).
Risk-free rate = 0.
Nếu Sharpe > 5.0 → RED FLAG, ghi chú bắt buộc.

### 5.5 Timestamp verification

Với mỗi trade, log:
- `signal_ts` = timestamp của bar `t` (khi Z-score trigger signal)
- `exec_ts` = timestamp của bar `t+1` (khi lệnh được executed)

Assert: `exec_ts > signal_ts` trên 100% trades.

**Ghi chú bắt buộc trong log:**
"Timestamp verification PASS ở engine-level không đảm bảo data-level clean. Bias từ D1 nằm trong file data, không phải execution logic. Hai cấp độ này độc lập nhau."

---

## CP6a — Sharpe sensitivity sweep: k% × 4 methods

Chạy backtest engine (CP5, Version A — OLS sizing) trên 20 flawed datasets + 1 clean baseline.

### Sharpe sensitivity table (điền kết quả thực tế)

| k% inject | Clean | H1: Future close | H2: Timestamp | H3: Spread | H4: Normalization |
|---|---|---|---|---|---|
| 0% (clean) | ___ | — | — | — | — |
| 10% | — | ___ | ___ | ___ | ___ |
| 20% | — | ___ | ___ | ___ | ___ |
| 30% | — | ___ | ___ | ___ | ___ |
| 40% | — | ___ | ___ | ___ | ___ |
| 50% | — | ___ | ___ | ___ | ___ |

### 4 câu hỏi phân tích sau khi điền

**Câu 1 — Slope:** Method nào Sharpe tăng nhanh nhất theo k%?
Kỳ vọng: H3 > H1 > H4 > H2.

**Câu 2 — Crossover:** Tại k% nào H4 (normalization) vượt H2 (timestamp)?

**Câu 3 — Target Sharpe ~10:** Method nào và k% nào đạt đầu tiên?

**Câu 4 — Impact/Detectability ratio:** Method nào nguy hiểm nhất theo tỷ lệ Sharpe impact / khó detect?
Kỳ vọng: H4 cao nhất — impact trung bình nhưng gần như không thể detect từ file.

### Secondary metrics per run

- Số trades (thay đổi so với baseline 90?)
- Avg gross PnL/trade (bps)
- Max drawdown (net)

---

## CP6b — Engine-level injection (comparative)

Dùng clean dataset. Thay đổi duy nhất trong engine: `position_executed[t] = position[t]`.
Đây là lỗi execution lag Week 2 đã tường minh tránh. 100% trades bị affect.

| Version | Bias location | k% affect | Net Sharpe | Max DD |
|---|---|---|---|---|
| Clean engine + clean data | None | 0% | ___ | ___ |
| Clean engine + H1 k=50% | Data | 50% | ___ | ___ |
| Clean engine + H3 k=50% | Data | 50% | ___ | ___ |
| Biased engine + clean data | Engine | 100% | ___ | ___ |
| Biased engine + H1 k=50% | Both | 100%+50% | ___ | ___ |

**Insight:** Engine-level bias (100%) vs data-level bias (50%) — cái nào cho Sharpe cao hơn? Lý do tại sao spec yêu cầu inject vào data: data-level bias khó detect hơn vì timestamp verification vẫn pass.

---

## CP6c — Net metrics: Sizing Cấp 1 vs Cấp 3

Chạy clean dataset với cả 2 sizing versions. So sánh:

| Metric | Version A (OLS β=1.0487) | Version B (Kalman β monthly) | Delta |
|---|---|---|---|
| Net Sharpe | ___ | ___ | ___ |
| Max Drawdown | ___ | ___ | ___ |
| CAGR | ___ | ___ | ___ |
| Calmar | ___ | ___ | ___ |
| Avg gross PnL/trade | ___ | ___ | ___ |
| Win rate (net) | ___ | ___ | ___ |

**Kỳ vọng từ Week 2:** Version B có lower drawdown về cuối năm (Kalman β đã converge về 0.78, giảm structural mishedge). PnL delta tích lũy rõ từ tháng 9–12.

---

## CP6d — Threshold sensitivity: Z=2.0 vs Z=2.57

Chạy trên clean dataset, OLS sizing:

| Metric | Z=2.0 (90 trades) | Z=2.57 (66 trades) |
|---|---|---|
| Net Sharpe | ___ | ___ |
| Max Drawdown | ___ | ___ |
| Total cost | 5,400 bps | 3,960 bps |
| Winner | ___ | ___ |

---

## CP6e — Negative control sanity check

| Pair | EG t-stat | Net Sharpe | Verdict |
|---|---|---|---|
| CMS/DUK | −5.268 | ___ | Expected > 0 |
| DOW/LYB | −4.861 | ___ | Expected > 0 |
| CVNA/ISRG | +0.778 | ___ | Expected ≈ 0 hoặc âm |
| INTC/JPM | +0.132 | ___ | Expected ≈ 0 hoặc âm |

Nếu CVNA/ISRG Sharpe ≈ CMS/DUK → signal không discriminate → không có genuine edge.

---

## CP7 — Verified Backtest Log (.txt)

**Cấu trúc 7 phần:**

**Phần 1 — Header**
```
VERIFIED BACKTEST LOG
=====================
Strategy        : Pairs Trading Z-Score Signal Engine
Run timestamp   : [datetime]
Engine          : Custom vectorized — pandas + Numba
Sizing          : Version A (OLS β=1.0487) + Version B (Kalman β monthly)
PnL method      : Daily mark-to-market (Cách C)
Sharpe factor   : √252 (daily returns)
Transaction cost: 60 bps round-trip, split 30bps entry + 30bps exit
Dataset (clean) : Jul–Dec 2022, CMS/DUK + DOW/LYB + negative controls
Dataset (biased): 20 flawed datasets (4 methods × 5 k%)
Total bars      : [N]
Total trades (clean, Z=2.0): [N]
```

**Phần 2 — Timestamp Verification**
```
Method          : exec_ts > signal_ts required on all trades
Result          : PASSED / FAILED
Trades verified : [N] / [N] (100%)

IMPORTANT: Engine-level timestamp verification PASS does not guarantee
data-level cleanliness. D1 bias resides in data files, not execution logic.
These are two independent layers of verification.
```

**Phần 3 — Trade Log mẫu (20–30 dòng)**
Columns: `signal_ts | exec_ts | pair | side | entry_price | exit_price | gross_pnl_bps | entry_cost_bps | exit_cost_bps | net_pnl_bps`

**Phần 4 — Sharpe Sensitivity Table**
Toàn bộ bảng k% × 4 methods từ CP6a. Đây là core output của D2.

**Phần 5 — Summary Metrics**
Version A (OLS sizing) và Version B (Kalman sizing) side by side:
Sharpe, MaxDD, CAGR, Calmar, Win Rate, Avg Gross PnL/trade.
RED FLAG nếu Sharpe > 5 trên clean dataset.

**Phần 6 — Comparative Analysis**
Engine-level vs data-level Sharpe (CP6b).
Sizing Cấp 1 vs Cấp 3 delta (CP6c).
Threshold sensitivity (CP6d).
Negative control results (CP6e).
Cost sensitivity (40 / 60 / 80 bps scenarios).

**Phần 7 — Audit Trail**
```
Parameter               Value       Source
OLS α (CMS/DUK)       : −0.6956    Week 2, formation Jan–Jun 2022
OLS β (CMS/DUK)       : 1.0487     Week 2, formation Jan–Jun 2022
Kalman β (Jun end)    : 0.7629     Week 2, Section 8
Rolling window        : 680 bars   Week 2, window = clip(HL, 10, 2000)
Entry threshold       : Z = 2.0    Week 2, Section 6
Adaptive threshold    : Z = 2.57   Week 2, 95th pct |Z| formation
Session warmup        : 30 bars    Week 2, Section 5.2
Burn-in               : 340 bars   window // 2
Execution lag         : 1 bar      position_executed[t] = position[t−1]
signal_valid flag     : Yes        PnL gating, not .dropna()
Random seed (D1)      : 42         All 20 flawed datasets
Cost split            : 30+30 bps  Entry + exit, not lump at exit

Known limitations (Week 2 FLAGS):
FLAG 3 — Kalman β drift 26.5%: addressed by Version B (Kalman sizing)
FLAG 2 — 30-bar warmup: 21 trades suppressed vs no-warmup baseline
FLAG 4 — EG evidence individual, not portfolio-level
FLAG 6 — No transaction cost in Week 2 signal: resolved here (30+30 split)
```

---

## Red flags

| Red flag | Ngưỡng | Nguyên nhân | Cách fix |
|---|---|---|---|
| Clean Sharpe > 5 | > 5.0 | Lookahead còn sót | `.dropna()` → `signal_valid`; √252 không phải √(252×390) |
| Biased Sharpe không tăng theo k% | Flat | Inject không vào đúng column engine đọc | H3: đọc `spread_biased` chưa? |
| H2 Sharpe > H1 Sharpe | H2 > H1 | Timestamp logic sai | H2 phải có impact nhỏ hơn H1 |
| H4 Sharpe = clean | Không đổi | Normalization không propagate qua spread | Engine dùng normalized close để tính spread chưa? |
| CVNA/ISRG Sharpe ≈ CMS/DUK | Chênh < 2× | Signal không discriminate | Engine hoặc 2022 quá noisy |
| Version B Sharpe < Version A | B < A | Kalman rebalance implement sai | Monthly rebalance có đúng timing chưa? |
| Avg hold << HL | < 100 bars | NaN handling sai | Kiểm tra NaN-safety trong state machine |
| Entry cost missing | Cost = 60 tại exit | Split 30/30 chưa implement | Deduct 30bps ngay tại entry bar |

---

## Checklist nộp bài

### Deliverable 1
- [ ] Write-up (text box): 4 ý tưởng đủ, dự đoán tác động rõ
- [ ] Code file: CP1 + CP2 + CP3, chạy được không lỗi
- [ ] 20 flawed datasets đã lưu đúng naming convention
- [ ] Validate pass cho cả 4 methods trước khi lưu
- [ ] H1 k=50% và H3 k=50%: Sharpe biased ≥ 5× clean

### Deliverable 2
- [ ] Code file: CP4 → CP6e + CP7, chạy được không lỗi
- [ ] Verified Backtest Log (.txt): đủ 7 phần
- [ ] Sharpe sensitivity table: 20 cells điền đầy đủ
- [ ] Cost split 30+30 bps đã implement (không phải lump 60 tại exit)
- [ ] Daily mark-to-market equity curve (không phải per-trade lump sum)
- [ ] Sharpe dùng √252 (không phải √313.5)
- [ ] Version A và Version B chạy song song, có comparison table
- [ ] Timestamp verification: 100% PASSED
- [ ] Engine-level comparative table (CP6b) đã điền
- [ ] Negative control chạy: CVNA/ISRG và INTC/JPM
- [ ] Clean Sharpe > 5: có giải thích rõ nguyên nhân

---

*Week 3 Final Plan v3 — Tất cả quyết định kiến trúc đã chốt: custom vectorized engine, OLS signal, Cấp 1 + Cấp 3 sizing song song, daily MTM PnL, Sharpe √252, cost 30+30bps split. 4 data-level bias methods × 5 k-values = 20 flawed datasets. Mọi parameter truy nguyên về formation period Jan–Jun 2022.*
