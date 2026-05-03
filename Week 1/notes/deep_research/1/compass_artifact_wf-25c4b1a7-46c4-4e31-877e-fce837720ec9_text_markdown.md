# Nền tảng thống kê của pairs trading: từ cointegration đến Engle-Granger

**Cointegration — chứ không phải correlation — là nền tảng toán học duy nhất đủ chặt chẽ để xác định một cặp tài sản có spread mean-reverting, từ đó xây dựng chiến lược pairs trading có cơ sở khoa học.** Hiểu sai điều này dẫn đến false positives hàng loạt, đặc biệt khi quét ~500 cặp trên dữ liệu minute-level. Bài viết này trình bày chi tiết 8 vấn đề cốt lõi mà một quant practitioner cần nắm vững: từ định nghĩa toán học của cointegration, phương pháp Engle-Granger hai bước, đến các sai lầm phổ biến khi triển khai trên `statsmodels`. Mọi nội dung được xây dựng trên các nguồn học thuật gốc (Engle & Granger 1987, Hamilton 1994, MacKinnon 2010) và tài liệu chính thức của `statsmodels`.

---

## 1. Cointegration là gì — định nghĩa, trực giác, và cơ sở toán học

### Định nghĩa hình thức

Engle và Granger (1987, *Econometrica*, Vol. 55) định nghĩa: các thành phần của vector **y**_t được gọi là **cointegrated bậc (d, b)**, ký hiệu CI(d, b), nếu (i) mỗi thành phần là I(d), và (ii) tồn tại vector **β** ≠ **0** sao cho **β**'**y**_t ~ I(d − b) với b > 0.

Trong trường hợp phổ biến nhất (d = b = 1): hai chuỗi giá Y_t và X_t đều là **I(1)** (non-stationary, chứa unit root), nhưng tồn tại tổ hợp tuyến tính:

> **Y_t − β·X_t ~ I(0)** (stationary)

Vector β được gọi là **cointegrating vector** (trong pairs trading chính là **hedge ratio**). Mối quan hệ dài hạn được biểu diễn qua phương trình cointegrating regression:

> **Y_t = α + β·X_t + ε_t**

trong đó ε_t là **disequilibrium error** — sai số cân bằng. Nếu cointegration tồn tại thì ε_t là I(0): nó dao động quanh giá trị trung bình cố định, không drift đi vô hạn.

### Trực giác

Hình dung hai người say rượu buộc bằng một sợi dây. Mỗi người đi theo đường ngẫu nhiên riêng (random walk — I(1)), nhưng **khoảng cách giữa họ bị giới hạn bởi sợi dây** — nó dao động quanh một giá trị trung bình và luôn bị kéo về. Sợi dây chính là cointegrating relationship, khoảng cách chính là spread. Khi spread lệch xa khỏi trung bình → cơ hội giao dịch; khi spread quay về → chốt lời.

### Granger Representation Theorem và Error Correction

Định lý biểu diễn Granger (Engle & Granger, 1987) thiết lập mối tương đương cơ bản: **nếu Y_t và X_t cointegrated thì tồn tại Error Correction Model (ECM), và ngược lại.** ECM có dạng:

> ΔY_t = γ₀ + λ·(Y_{t−1} − α − β·X_{t−1}) + lags + η_t

Hệ số **λ < 0** chính là tốc độ điều chỉnh — nó kéo Y_t quay về equilibrium khi spread lệch. Half-life of mean reversion được tính: **t₁/₂ = ln(0.5) / ln(1 + λ)**, cho biết cần bao lâu để spread giảm một nửa khoảng cách về trung bình.

Một cách nhìn bổ sung: Stock & Watson (1988) chứng minh rằng các chuỗi cointegrated **chia sẻ common stochastic trend**. Cointegrating vector có tác dụng **loại bỏ trend chung này**, chỉ để lại thành phần stationary.

---

## 2. Correlation và cointegration khác nhau như thế nào

Đây là sự khác biệt quan trọng nhất mà nhiều người bỏ qua. Bảng so sánh:

| Thuộc tính | Correlation | Cointegration |
|---|---|---|
| **Hoạt động trên** | Returns (ΔY, ΔX) — sai phân | Price levels (Y, X) — mức giá |
| **Đo lường** | Đồng biến động ngắn hạn | Cân bằng dài hạn |
| **Toán học** | ρ = Cov(ΔY, ΔX) / (σ_ΔY · σ_ΔX) | Liệu Y_t − β·X_t có stationary? |
| **Tầm nhìn thời gian** | Từ period này sang period kế | Toàn bộ lịch sử giá |

Theo Palomar (HKUST, *Portfolio Optimization*): "Correlation is concerned with the short-term movements... while ignoring the long-term trends. Cointegration focuses on the long term... while being oblivious to short-term variations."

**Điều cốt lõi: correlation và cointegration không có mối quan hệ xác định.** Hudson & Thames minh họa bằng mô phỏng: hai chuỗi giá highly correlated nhưng Engle-Granger ADF statistic chỉ đạt 0.41 — hoàn toàn không reject null hypothesis "no cointegration" kể cả ở mức **90%**. Ngược lại, hai chuỗi cointegrated có thể có rolling correlation **âm** trong một số giai đoạn khi một chuỗi vượt trội tạm thời.

---

## 3. Tại sao high correlation không đủ để chọn cặp giao dịch

### Spurious correlation trong chuỗi non-stationary

Granger & Newbold (1974) và Phillips (1986) đã chứng minh: **hồi quy một random walk lên một random walk độc lập hoàn toàn vẫn cho R² cao và t-statistic có vẻ có ý nghĩa.** Zivot minh họa: hai I(1) processes hoàn toàn độc lập cho OLS t-statistic = 8.035, R² = 0.207. Khi T → ∞, R² hội tụ về 1 và t-statistic phân kỳ — đây là "spurious regression", một bẫy kinh điển.

### Ví dụ cụ thể về thất bại

**Coca-Cola (KO) và Pepsi (PEP):** Return correlation đạt **0.66** (significant thống kê), nhưng Chan (2008) và Palomar chứng minh chúng **không cointegrated** — half-life của residual lên tới 70 ngày, p-value cointegration test "much larger than 0.01". Nếu bạn mở vị thế pairs trade dựa trên correlation, spread có thể drift vô hạn.

**Hai tech stocks 2020–2021:** Cả hai hưởng lợi từ COVID remote-work (correlation > 0.95). Nhưng năm 2021, một công ty pivot thành công sang AI, công ty kia gặp vấn đề pháp lý. Spread phân kỳ vĩnh viễn — không có lực kéo về mean. Cointegration test sẽ phát hiện điều này; correlation test thì không.

**Bài học:** Correlation chỉ nói rằng "hai chuỗi di chuyển cùng hướng trong ngắn hạn." Nó **không đảm bảo** spread sẽ mean-revert — điều kiện tiên quyết duy nhất của pairs trading.

---

## 4. Phương pháp Engle-Granger hai bước — chi tiết từng bước

### Bước 1: Ước lượng cointegrating regression bằng OLS

Chạy hồi quy OLS: **Y_t = α̂ + β̂·X_t + ε̂_t**

Kết quả: β̂ (hedge ratio), α̂ (intercept), ε̂_t (residuals). Nếu cointegration thực sự tồn tại, OLS là **super-consistent** — β̂ hội tụ về true β ở tốc độ T (thay vì √T), nhanh hơn nhiều so với OLS thông thường. Tuy nhiên, t-statistics từ regression này **không thể diễn giải** theo cách thông thường vì phân phối non-standard.

### Bước 2: Kiểm định ADF trên residuals

Chạy ADF test trên ε̂_t:

> **Δε̂_t = ρ·ε̂_{t−1} + Σᵢ δᵢ·Δε̂_{t−i} + v_t**

Giả thuyết:
- **H₀: ρ = 0** → residuals có unit root → KHÔNG cointegrated (spurious regression)
- **H₁: ρ < 0** → residuals stationary → **cointegrated**

Nếu reject H₀, kết luận: tổ hợp tuyến tính Y_t − β̂·X_t là I(0), tức Y và X cointegrated.

### Critical values: dùng MacKinnon, KHÔNG dùng ADF chuẩn

**Đây là chi tiết kỹ thuật cực kỳ quan trọng.** Phillips & Ouliaris (1990) chứng minh: vì β được ước lượng từ dữ liệu (không phải known a priori), phân phối của test statistic khác với Dickey-Fuller chuẩn. Phải dùng **MacKinnon (1991, 2010) critical values** — phụ thuộc vào số biến N trong hệ thống cointegration.

So sánh critical values tại 5% significance (N=2, có constant, mẫu lớn):

| | ADF chuẩn (N=1) | Engle-Granger (N=2) |
|---|---|---|
| **5% critical value** | −2.86 | **−3.34** |
| **1% critical value** | −3.43 | **−3.90** |

**Critical values cointegration nghiêm ngặt hơn nhiều** — dùng nhầm bảng ADF chuẩn sẽ tạo ra hàng loạt false positives.

---

## 5. ADF test phải chạy trên residuals — tại sao không chạy trên prices

Câu hỏi cốt lõi của cointegration **không phải** "liệu từng chuỗi giá có stationary không" (chắc chắn không — giá cổ phiếu là I(1)). Câu hỏi đúng là: **"liệu có tồn tại tổ hợp tuyến tính nào của hai chuỗi giá mà stationary không?"**

Residuals từ cointegrating regression chính xác là tổ hợp tuyến tính đó: **ε̂_t = Y_t − α̂ − β̂·X_t**. Kiểm định ADF trên ε̂_t chính là kiểm tra xem tổ hợp này có I(0) hay không.

Chạy ADF trên Y_t hoặc X_t riêng lẻ sẽ (đúng) cho kết quả "fail to reject unit root" — nhưng thông tin này **hoàn toàn vô nghĩa** đối với câu hỏi cointegration. Đây là bước sai mà beginners thường mắc: họ thấy ADF không reject trên prices và kết luận sai rằng "không có cointegration."

Quy trình đúng:
1. ADF trên Y_t → xác nhận Y_t là I(1) ✓
2. ADF trên X_t → xác nhận X_t là I(1) ✓
3. OLS regression Y_t trên X_t → lấy residuals ε̂_t
4. **ADF trên ε̂_t** → đây mới là cointegration test thực sự

---

## 6. Residual stationarity nghĩa là gì — diễn giải kinh tế và thống kê

### Diễn giải thống kê

Khi ε_t là I(0) (stationary), nó thỏa mãn: **E[ε_t] = μ cố định** (trung bình không drift), **Var(ε_t) = σ² cố định** (phương sai không phát tán), autocovariance chỉ phụ thuộc vào lag. Spread dao động quanh giá trị trung bình ổn định, không drift đi vô hạn, và có cấu trúc autocorrelation dự đoán được.

### Diễn giải kinh tế

Theo Zivot: "I(1) time series with a long-run equilibrium relationship cannot drift too far apart because **economic forces will act to restore the equilibrium.**"

Các lực kinh tế duy trì cointegration bao gồm: **arbitrage** (cùng tài sản trên các sàn khác nhau), **fundamental linkage** (công ty cùng tiếp xúc các yếu tố kinh tế tương tự, ví dụ ETF Australia EWA và Canada EWC đều phụ thuộc tài nguyên thiên nhiên), và **supply substitution** (sản phẩm cạnh tranh không thể có giá phân kỳ vĩnh viễn).

**Stationarity của residuals đồng nghĩa với sự tồn tại của error correction mechanism**: mỗi khi spread lệch khỏi equilibrium, có lực kéo nó về. Granger Representation Theorem đảm bảo rằng cointegration ↔ ECM — hai khái niệm tương đương hoàn toàn. Đây chính là lý do tại sao spread mean-reverts và tại sao pairs trading hoạt động.

---

## 7. Khi nào có thể tuyên bố một cặp là cointegrated

### Sử dụng `statsmodels.tsa.stattools.coint()`

**Luôn dùng `coint()` thay vì tự chạy OLS + `adfuller()`.** Lý do: `coint()` tự động xử lý đúng hai vấn đề kỹ thuật quan trọng:

1. Chạy ADF trên residuals với **`regression='n'`** (không thêm constant vào ADF vì constant đã nằm trong OLS regression)
2. Tính p-value và critical values bằng **MacKinnon N=2** (cointegration-specific), không phải N=1 (standard ADF)

```python
from statsmodels.tsa.stattools import coint
coint_t, pvalue, crit_values = coint(price_a, price_b, trend='c', autolag='aic')
```

Return values: `coint_t` (test statistic), `pvalue` (MacKinnon approximate p-value với N=k_vars), `crit_values` (critical values tại 1%, 5%, 10%).

Nếu cần hedge ratio (mà `coint()` không trả về trực tiếp), dùng manual approach nhưng **phải** sửa p-value:

```python
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
# Sau khi chạy OLS + adfuller(residuals, regression='n')
correct_pvalue = mackinnonp(adf_stat, regression='c', N=2)
correct_crit = mackinnoncrit(N=2, regression='c', nobs=nobs)
```

### Ngưỡng significance và p-value

- **p < 0.05** (5%): ngưỡng chuẩn, được sử dụng phổ biến nhất trong thực hành (Rad et al. 2016; nhiều ETF pairs studies)
- **p < 0.01** (1%): conservative, ưu tiên khi cần high confidence
- **p < 0.10** (10%): borderline, chỉ dùng khi kết hợp với economic rationale mạnh

### Multiple testing correction — bắt buộc khi test ~500 cặp

Với 500 cặp tại α = 0.05, kỳ vọng **~25 false positives** thuần túy do ngẫu nhiên. Nếu quét toàn bộ C(500,2) = 124,750 cặp, con số này lên **~6,237**. Hai phương pháp hiệu chỉnh:

- **Bonferroni:** α_adjusted = 0.05/500 = 0.0001. Rất conservative — bỏ lỡ nhiều cặp thật sự cointegrated.
- **Benjamini-Hochberg (FDR):** kiểm soát tỷ lệ kỳ vọng của false discoveries trong số các cặp được chọn. **Đây là phương pháp được khuyến nghị cho large-scale screening.**

```python
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals_array, alpha=0.05, method='fdr_bh')
```

### Sample size cho minute-level data

Với 390 phút/ngày giao dịch, 1 tuần ≈ 1,950 observations — dữ liệu dồi dào. Tuy nhiên, **nhiều hơn không luôn tốt hơn**: lookback quá dài có thể bao gồm structural breaks đã xảy ra. Khuyến nghị thực tế:

- **Rolling window 1,000–3,000 minute bars** (3–8 ngày giao dịch) cho intraday strategies
- Cân nhắc **downsample xuống 5-min bars** để giảm microstructure noise
- **Cap maxlag** ở mức 20–30 khi quét nhiều cặp (default formula `12*(nobs/100)^0.25` có thể cho maxlag rất lớn với dữ liệu minute)
- Re-test cointegration **định kỳ** (hàng ngày hoặc vài giờ)

---

## 8. Những sai lầm phổ biến mà beginners mắc phải

### Sai lầm #1: Chạy ADF trên prices thay vì residuals
Kết quả: ADF đúng báo "prices là non-stationary" nhưng **không nói gì về cointegration**. Người mới thường thấy kết quả này và kết luận sai "không cointegrated." Phải chạy ADF trên **ε̂_t = Y_t − α̂ − β̂·X_t**.

### Sai lầm #2: Dùng standard ADF critical values thay vì MacKinnon N=2
Nếu dùng `adfuller()` thay vì `coint()` và lấy p-value trực tiếp, p-value sẽ dùng **N=1** — quá liberal, tạo false positives. Critical value 5% chuẩn là −2.86 nhưng cointegration cần **−3.34**. Sai lầm này nghiêm trọng hơn nhiều người nghĩ.

### Sai lầm #3: Nhầm correlation với cointegration
KO-PEP có correlation 0.66 nhưng KHÔNG cointegrated (half-life 70 ngày). Screening bằng correlation alone → chọn nhiều cặp mà spread drift vĩnh viễn. Correlation là điều kiện cần hợp lý để **pre-filter** (giảm số cặp cần test), nhưng quyết định cuối cùng **phải dựa trên cointegration test**.

### Sai lầm #4: Look-ahead bias
Ước lượng hedge ratio trên toàn bộ dataset rồi backtest trên cùng dữ liệu đó. Backtest sẽ đẹp bất thường nhưng live trading thất bại. QuantRocket báo cáo: **chỉ ~40% cặp cointegrated in-sample vẫn cointegrated out-of-sample.** Giải pháp: rolling/expanding window estimation, walk-forward analysis, hoặc Kalman filter.

### Sai lầm #5: Không hiệu chỉnh multiple testing
500 tests × 5% significance = **~25 false positives kỳ vọng**. Phải dùng BH-FDR correction. Thêm vào đó, **pre-filter bằng economic logic** (cùng sector, cùng supply chain) trước khi test thống kê — vừa giảm số tests vừa tăng chất lượng cặp.

### Sai lầm #6: Bỏ qua structural breaks
Cointegration KHÔNG phải vĩnh viễn. Ví dụ thực tế: "Liberation Day" tariffs (4/2025) phá vỡ cointegration S&P 500–Nikkei 225 (Chow test F = 57.50, p ≈ 0); GLD-GDX "went haywire" trong khủng hoảng 2008. Ernie Chan cảnh báo: "It is actually quite hard to detect the breakdown of cointegration except in hindsight maybe a year afterwards."

### Sai lầm #7: Giả định cointegration = mean reversion tức thì
Cointegration chỉ đảm bảo spread **sẽ quay về** equilibrium, nhưng không nói **bao nhanh**. Half-life là thông số quyết định: EWA-EWC có half-life 19 ngày (giao dịch được), KO-PEP có half-life 70 ngày (quá chậm). Cặp có half-life > 60–120 ngày thường không thực tế; < 1 ngày thì gặp vấn đề execution.

### Sai lầm #8: Không kiểm tra I(1) trước khi test cointegration
Cointegration được **định nghĩa** chỉ giữa các chuỗi I(1). Nếu một chuỗi đã I(0), bạn không cần cointegration — trade trực tiếp. Nếu chuỗi là I(2), standard cointegration tests cho kết quả sai lệch. Luôn confirm integration order trước.

---

## (a) Methodology note: Pipeline đúng cho Pairs Selection Report

**Phase 1 — Pre-screening (Economic Logic First):** Xác định universe tài sản có fundamental linkage hợp lý (cùng sector, cùng commodity exposure, cùng supply chain). Pre-filter bằng correlation > 0.7–0.9 để giảm số cặp. Mục đích: giảm multiple testing burden và tăng tỷ lệ true cointegration.

**Phase 2 — Integration Order Check:** Chạy ADF (hoặc KPSS) trên từng chuỗi giá riêng lẻ. Confirm cả hai chuỗi là I(1). Loại bỏ cặp nào có chuỗi I(0) hoặc I(2).

**Phase 3 — Cointegration Test:** Chạy `statsmodels.tsa.stattools.coint(y0, y1, trend='c')` cho mỗi cặp. Lưu p-value, test statistic. Nếu cần hedge ratio: chạy OLS thủ công, lấy residuals, nhưng dùng `mackinnonp(stat, N=2)` cho p-value đúng.

**Phase 4 — Multiple Testing Correction:** Áp dụng Benjamini-Hochberg FDR (`multipletests(pvals, method='fdr_bh', alpha=0.05)`) lên toàn bộ p-values. Chỉ giữ các cặp reject H₀ sau correction.

**Phase 5 — Half-life Filter:** Tính half-life of mean reversion (fit AR(1) hoặc OU process trên spread). Loại cặp có half-life nằm ngoài khoảng thực tế cho trading horizon.

**Phase 6 — Out-of-sample Validation:** Ước lượng hedge ratio trên training window, test cointegration + simulate trading trên validation window hoàn toàn tách biệt. Chỉ giữ cặp vẫn cointegrated OOS.

**Phase 7 — Live Monitoring:** Re-estimate hedge ratio liên tục (Kalman filter hoặc rolling OLS). Monitor structural breaks (CUSUM). Re-test cointegration định kỳ. Sẵn sàng retire cặp mất cointegration.

---

## (b) Checklist "Must-do-correctly"

- ☐ **Dùng `coint()`, không phải `adfuller()` trực tiếp trên residuals** — `coint()` tự động dùng MacKinnon N=2 critical values và `regression='n'` đúng cách
- ☐ **Confirm cả hai chuỗi là I(1)** trước khi test cointegration (ADF hoặc KPSS trên từng chuỗi)
- ☐ **Áp dụng BH-FDR correction** khi test nhiều cặp — `multipletests(pvals, method='fdr_bh')`
- ☐ **Tính half-life** của spread và loại cặp có half-life nằm ngoài trading horizon
- ☐ **Tách biệt in-sample và out-of-sample** — ước lượng hedge ratio trên training data, validate trên unseen data
- ☐ **Pre-filter bằng economic logic** — cùng sector, fundamental linkage rõ ràng — trước khi chạy tests
- ☐ **Dùng log prices nhất quán** — không mix log và level prices
- ☐ **Cap maxlag** (20–30) khi dùng minute-level data để kiểm soát thời gian tính toán
- ☐ **Re-estimate hedge ratio** liên tục trong production (Kalman filter hoặc rolling window)
- ☐ **Monitor structural breaks** và sẵn sàng retire cặp khi cointegration breakdown

---

## (c) Checklist "Common misconceptions to avoid"

- ✗ **"ADF reject trên prices → cointegrated"** — Sai. ADF trên prices chỉ kiểm tra stationarity của từng chuỗi riêng lẻ. Cointegration test phải chạy trên residuals của cointegrating regression.
- ✗ **"Correlation cao = pairs trade tốt"** — Sai. Correlation đo co-movement ngắn hạn của returns; cointegration đo equilibrium dài hạn của prices. Hai khái niệm này độc lập về mặt toán học.
- ✗ **"Dùng critical values ADF chuẩn cho Engle-Granger test"** — Sai. Phải dùng MacKinnon critical values với N = số biến (N=2 cho bivariate). Critical values chuẩn quá liberal, tạo false positives.
- ✗ **"Cointegration là vĩnh viễn"** — Sai. Cointegrating relationships breakdown do M&A, regulatory changes, regime shifts. Chỉ ~40% cặp cointegrated in-sample vẫn giữ OOS.
- ✗ **"Cointegrated = mean revert ngay lập tức"** — Sai. Phải kiểm tra half-life. Cặp có half-life 70+ ngày gần như không thể trade được dù statistically cointegrated.
- ✗ **"Test 500 cặp ở 5% significance là đủ"** — Sai. Kỳ vọng ~25 false positives. Phải dùng FDR correction.
- ✗ **"Backtest trên cùng dữ liệu dùng để estimate hedge ratio"** — Sai. Đây là look-ahead bias. Phải tách training/validation nghiêm ngặt.
- ✗ **"Hedge ratio ước lượng một lần dùng mãi"** — Sai. Hedge ratio time-varying; cần re-estimate liên tục bằng Kalman filter hoặc rolling regression.