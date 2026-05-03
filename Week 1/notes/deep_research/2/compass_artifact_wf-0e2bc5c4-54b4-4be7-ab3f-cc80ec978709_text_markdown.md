# Pairs Trading trên Minute Data: Những cạm bẫy thống kê và quy trình xử lý dữ liệu

**Cointegration test trên dữ liệu phút chứa rủi ro thống kê nghiêm trọng mà dữ liệu daily không gặp phải.** Microstructure noise từ bid-ask bounce, stale quotes và asynchronous trading có thể khiến ADF test reject unit root giả tạo — đặc biệt khi ~98,000 observations/năm cho test power cực lớn nhưng tín hiệu thực lẫn trong nhiễu. Báo cáo này phân tích 8 vấn đề cốt lõi khi chạy Engle-Granger cointegration trên minute-level equity prices, kèm workflow xử lý dữ liệu cụ thể cho Jupyter notebook. Nguồn tham chiếu chính gồm Roll (1984), Epps (1979), Hasbrouck (1993), Aït-Sahalia et al. (2005), Gatev et al. (2006), Vidyamurthy (2004) và Krauss (2017).

---

## 1. Năm vấn đề phổ biến của minute-level price data

Dữ liệu phút mang theo lớp nhiễu mà dữ liệu daily đã "tự lọc" qua aggregation. Hiểu rõ từng nguồn nhiễu giúp tránh kết luận cointegration sai.

**Bid-ask bounce** là hiện tượng giá giao dịch dao động giữa bid và ask dù giá thực (efficient price) không đổi. Roll (1984) chứng minh covariance bậc 1 của price changes bằng **−s²/4** (s = spread). Với spread 1 cent trên cổ phiếu $50, autocorrelation âm ở tần suất phút có thể đạt −0.05 đến −0.15 — lớn hơn nhiều lần so với daily. Khi chạy OLS regression trong Engle-Granger step 1, noise mean-reverting từ bounce lẫn vào residual, khiến ADF test ở step 2 nhầm tưởng spread stationary trong khi thực tế chỉ là nhiễu vi cấu trúc.

**Stale quotes** xảy ra khi không có giao dịch trong một phút — giá hiển thị là giá cũ, đã lỗi thời. Cổ phiếu small-cap có thể đi nhiều phút liên tiếp không trade, đặc biệt giờ trưa (11:30–13:30 ET). Stale quote tạo lead-lag giả tạo: cổ phiếu liquid phản ánh thông tin mới trước, cổ phiếu illiquid "đuổi theo" sau — spread có vẻ mean-reverting nhưng thực chất là artifact của độ trễ thanh khoản.

**Illiquidity spikes** — khi thanh khoản rút đột ngột (tin tức, circuit breaker), spread mở rộng bất thường, tạo structural breaks trong variance. Heteroskedasticity này vi phạm giả định của ADF test. Trong Flash Crash 2010, bid-ask spread của cổ phiếu S&P 500 mở rộng từ ~1 cent lên $1–$5, khiến dữ liệu phút gần như random noise.

**Non-stationarity đặc thù intraday** biểu hiện qua mô hình biến động hình chữ U: volatility cao lúc mở cửa (9:30–10:00), thấp giờ trưa, cao lại trước đóng cửa (15:30–16:00). Andersen & Bollerslev (1997) documeted hiện tượng này chi tiết. Variance thay đổi có hệ thống trong ngày nghĩa là spread giữa hai cổ phiếu cũng có variance không đồng nhất — ADF test giả định homoskedasticity hoặc heteroskedasticity nhẹ, dẫn đến test statistic bị méo. Clinet & Potiron (2021) chứng minh cả Dickey-Fuller lẫn Phillips-Perron đều bị size distortion bởi time-varying variance kiểu này.

**Spurious cointegration từ sample size lớn giả tạo** là vấn đề nguy hiểm nhất. Với 390 phút/ngày × 252 ngày = **~98,000 observations/năm**, so với chỉ ~252 obs cho daily data. ADF test power tăng theo T, nhưng ở tần suất phút, microstructure noise không giảm khi thêm observations — nó cùng magnitude ở mỗi bar. Nếu spread thực z_t là random walk nhưng observed spread ẑ_t = z_t + ε_t (ε_t là noise stationary), với T = 98,000, ADF có đủ power để phát hiện thành phần stationary ε_t dù z_t non-stationary. Kết quả: **reject unit root giả tạo, kết luận cointegration sai**.

---

## 2. Missing timestamps, asynchronous trading và microstructure noise méo ADF test như thế nào

Bốn nguồn lỗi này hoạt động đồng thời, tạo hiệu ứng cộng dồn lên Engle-Granger regression.

**Missing timestamps** xảy ra khi không có trade trong khoảng 1 phút đó. Forward-fill (ffill) là cách xử lý phổ biến nhất — mang giá cuối cùng sang phút tiếp. Nhưng ffill tạo **zero returns giả** (giá đứng yên), giảm measured volatility, và tạo positive autocorrelation giả trong levels. Ví dụ: Stock A giao dịch $50.00 lúc 10:32, không trade đến 10:37. Forward-fill giữ $50.00 cho 5 phút trong khi Stock B liên tục biến động — spread mở rộng giả tạo rồi "snap back" khi A cuối cùng trade lại, tạo ảo giác mean-reversion.

**Asynchronous trading (hiệu ứng Epps)** được Epps (1979) chứng minh: **measured correlation giữa hai cổ phiếu giảm khi tần suất sampling tăng**. Trong cùng 1 phút, Stock A có thể trade lúc 10:30:12 còn Stock B lúc 10:30:48. Cả hai phản ứng cùng thông tin, nhưng ghi nhận ở thời điểm khác nhau. Tóth & Kertész (2009) cho thấy thời gian phản ứng đặc trưng của con người (~vài phút) giải thích hiệu ứng này. Với Engle-Granger regression, Epps effect làm giảm measured correlation → OLS beta estimate noisier → hedge ratio suboptimal → spread residual variance lớn hơn thực → ADF test mất power phát hiện cointegration thật.

**Market hours mismatch** là vấn đề khi hai cổ phiếu có hành vi khác nhau ngoài giờ giao dịch chính. Pre-market (4:00–9:30) và after-hours (16:00–20:00) có thanh khoản cực thấp, spread rất rộng. Nếu bao gồm dữ liệu này, noise tăng đáng kể. Nếu loại bỏ, tạo overnight gap. Nếu một cổ phiếu trade pre-market còn cổ kia không, lúc 9:30 AM spread sẽ phản ánh thông tin bất đối xứng — tạo spurious divergence đầu ngày.

**Microstructure noise** theo mô hình Hasbrouck (1993) phân tách observed price thành efficient price (random walk) và transitory noise. Kết quả đáng chú ý: **variance của noise có thể vượt variance của fundamental price** ở tần suất cao — Hasbrouck ước lượng noise chiếm khoảng **82%** variance returns trong dữ liệu US stocks. Aït-Sahalia, Mykland & Zhang (2005) chứng minh realized variance dùng tất cả tick data hội tụ về 2n·E[ε²] thay vì integrated variance — **phân kỳ** khi tăng tần suất sampling.

**Cơ chế cụ thể ảnh hưởng hai bước Engle-Granger:**

Ở **Step 1** (OLS: P_A,t = α + β·P_B,t + u_t), noise ε_B trên regressor tạo **errors-in-variables problem** cổ điển. β̂ bị attenuation bias — hội tụ về β × σ²_X/(σ²_X + σ²_η), luôn nhỏ hơn β thực. Hedge ratio bị kéo về 0 nghĩa là Stock A insufficiently hedged.

Ở **Step 2** (ADF trên residual û_t), residual kế thừa noise từ cả hai stocks. Bid-ask bounce tạo negative first-order autocorrelation trong residual → **ADF coefficient γ̂ trở nên negative hơn** so với null hypothesis → t-statistic inflated. Với 98,000 obs, bias nhỏ cũng tạo t-statistic rất lớn → **over-rejection of unit root null → spurious cointegration**. Dias et al. (2021) chứng minh bias trong LS estimation of cointegration **không shrink về 0** ngay cả khi giảm sampling frequency.

---

## 3. Raw prices vs log prices: quyết định quan trọng đầu tiên

**Log prices là standard trong cả academia lẫn practice**, và hầu hết trường hợp nên dùng log. Lý do cốt lõi: nếu P_t tuân theo Geometric Brownian Motion (GBM), log(P_t) là I(1) process tự nhiên. Cointegration trên log prices nghĩa là log(P_A) − β·log(P_B) stationary, tương đương P_A/P_B^β mean-reverting — **phản ánh mối quan hệ định giá tương đối**, có ý nghĩa kinh tế hơn dollar spread từ raw prices.

**Variance stabilization** là lý do kỹ thuật chính. Raw prices có heteroscedasticity nội tại: cổ phiếu $100 dao động dollar lớn hơn cổ phiếu $10 dù volatility phần trăm giống nhau. Log transformation loại bỏ vấn đề này vì Δlog(P_t) chính là log return — trực tiếp là percentage change, so sánh được across price levels. Alexander (2002) khẳng định: "it is standard, but not necessary, to perform the cointegration analysis on log prices."

**Academic convention:** Vidyamurthy (2004) dùng log prices, định nghĩa spread = log(a) − n·log(b). Avellaneda & Lee (2010) dùng log prices với PCA. Liu, Chang & Geman (2017) dùng log prices trên 5-minute data. Gatev et al. (2006) là ngoại lệ — dùng normalized raw prices cho distance method (không phải cointegration test). Ernie Chan recommend log prices cho cointegration approach: hedge ratio trên log prices đại diện **dollar values** (market value ratio), không phải số shares.

**Khi nào log gây vấn đề:** Cổ phiếu giá rất thấp (gần $0) — log(P) → −∞ tạo instability số học. Penny stocks khuếch đại noise trong log space. Giá âm (hiếm với equities, xảy ra với một số futures) — log undefined. Interpretation phức tạp hơn raw prices: hedge ratio trên log prices cần rebalance market values, không giữ cố định số shares.

**Quy tắc quyết định:**

| Tiêu chí | Dùng Raw Prices | Dùng Log Prices |
|----------|----------------|-----------------|
| Mức giá hai cổ phiếu | Tương tự nhau (~$50 vs ~$55) | Khác nhau đáng kể ($20 vs $200) |
| Rebalancing | Muốn giữ cố định số shares | Chấp nhận rebalance market values |
| Spread interpretation | Cần P&L bằng dollar | Cần signal dựa percentage |
| Loại instrument | Futures, FX | Equities (default) |
| Cổ phiếu giá thấp | Có penny stocks trong universe | Tất cả > $5 |

---

### Decision note cho notebook

```
# === DECISION NOTE: RAW vs LOG PRICES ===
# Dùng LOG PRICES cho cointegration testing (default choice).
# Lý do: (1) variance stabilization — loại heteroscedasticity inherent 
# trong raw prices; (2) log(P_A) - β·log(P_B) stationary ⟺ ratio 
# P_A/P_B^β mean-reverts — có ý nghĩa kinh tế; (3) standard trong 
# Vidyamurthy (2004), Avellaneda & Lee (2010), Liu et al. (2017).
#
# Khi nào dùng RAW: (a) tất cả cổ phiếu giá tương tự nhau, 
# (b) cần fixed-share portfolio không rebalance, (c) penny stocks 
# trong universe (log khuếch đại noise ở giá thấp).
#
# Hedge ratio từ log prices = market value ratio, KHÔNG phải share ratio.
# Spread = log(P_A) - β·log(P_B) ≈ percentage deviation.
# Đã filter penny stocks < $5 → dùng log prices an toàn.
```

---

## 4. Khi nào nên resample: 1-min vs 5-min vs 15-min vs hourly

Lựa chọn tần suất là trade-off giữa **information content** và **microstructure noise contamination**. Nghiên cứu cho thấy **5 phút là lựa chọn phổ biến nhất** trong literature intraday pairs trading, và là recommendation default.

**1-minute data** cho 390 obs/ngày (~98,000/năm), maximum information nhưng cũng maximum noise. Aït-Sahalia et al. (2005) chứng minh ngay cả S&P 500 constituents, "5-minute returns cannot be treated as noise-free" — 1-minute lại càng tệ hơn. Bid-ask bounce, stale quotes, và Epps effect đều strongest ở tần suất này. ADF test trở nên overpowered: sample size khổng lồ cho phép phát hiện stationary component nhỏ nhất (bao gồm noise), dẫn đến **spurious rejection rate cao**. Giữ nguyên 1-min chỉ hợp lý khi: (a) cả hai cổ phiếu đều extremely liquid (>$10M daily volume), (b) dùng mid-prices thay vì last trade, (c) mục đích là execution timing chứ không phải formation/testing.

**5-minute data** cho **78 obs/ngày** (~19,500/năm), giảm microstructure noise đáng kể. Andersen et al. (2001) recommend "5 or 15 min and not more frequently" là safe từ worst microstructure effects. Liu, Chang & Geman (2017) dùng 5-min cho intraday oil pairs trading, đạt Sharpe ratio 3.9. Hansen & Lunde (2006) xác nhận rằng noise có thể bị ignore ở tần suất 15–20 phút, nhưng 5 phút đã loại bỏ phần lớn bid-ask bounce. Đây là **sweet spot cho cointegration formation testing**.

**15-minute data** cho 26 obs/ngày, noise gần như minimal. Hansen & Lunde cho rằng giả định noise IID "seems reasonable" ở ~15 ticks. Miao (2014) dùng 15-min cho intraday pairs. Nhược điểm: chỉ ~130 obs/tuần — cần vài tuần data để đủ sample cho ADF. Có thể miss mean-reversion opportunities resolve trong vài phút.

**Hourly data** cho ~6.5 obs/ngày, minimal noise nhưng very limited observations. Bowen, Hutchinson & O'Sullivan (2010) dùng 60-min cho FTSE 100. Cần nhiều tuần đến tháng data cho reliable ADF test. Phù hợp nếu holding period dự kiến là vài giờ trở lên.

**Ảnh hưởng đến half-life estimation:** Ở 1-min, bid-ask bounce tạo ảo giác rapid mean-reversion → half-life ước lượng ngắn giả tạo. Ở hourly, short-lived mean-reversion bị averaged out → half-life ước lượng dài hơn hoặc undetectable. Holý & Tomanová (2018) đề xuất corrections cho bias trong OU parameter estimation trên ultra-high-frequency data. **Recommendation:** estimate half-life ở tần suất bạn dự định trade, cross-validate với tần suất lân cận.

**Lưu ý quan trọng về time span vs sample size:** Dave Giles nhấn mạnh rằng **span** quan trọng hơn **size** cho unit root tests. 500 observations 1-phút chỉ cover 1.3 ngày giao dịch — ADF có thể detect microstructure mean-reversion, không phải economic cointegration. Time span nên match intended holding period: ít nhất **10× expected half-life**. Với 5-min data, 3 tháng cho ~4,700 obs spanning ~63 ngày — đủ cho cả power lẫn span.

**Hayashi-Yoshida estimator** giải quyết asynchronous trading mà không cần resample, tính covariance trực tiếp trên dữ liệu không đồng bộ. Tuy nhiên, nó inconsistent khi có microstructure noise (Bibinger & Vetter, 2015) và chủ yếu dùng cho covariance estimation, không trực tiếp cho cointegration testing.

---

## 5. Align hai price series: forward fill, inner join, hay drop auctions?

**Timestamp alignment là bắt buộc** trước OLS regression — mỗi observation phải có cặp giá đồng thời. Ba phương pháp chính, mỗi cái có trade-off riêng.

**Forward fill (ffill)** đơn giản nhất (`df.ffill()`), giữ full sample size, phản ánh thực tế rằng nếu không có trade, giá cuối là best estimate. Nhưng tạo **artificial staleness** — Bandi et al. (2020, *Management Science*) chứng minh zero returns là "genuine economic phenomenon linked to liquidity." Ffill khuếch đại vấn đề này bằng cách tạo thêm zero returns ở bars không có trade. Với pairs trading, nếu Stock A stale trong khi Stock B biến động, OLS thấy noise trong X mà không có phản ứng tương ứng trong Y → **attenuation bias trong β̂** → hedge ratio kéo về 0 → spread variance inflated → ADF test mất power.

**Linear interpolation** tạo series mượt hơn, tránh zero-return inflation, nhưng **dùng thông tin tương lai** (giá tiếp theo) để nội suy — tạo **look-ahead bias** không chấp nhận được cho tradeable strategy. Ngoài ra, interpolated prices chưa bao giờ tồn tại trên market — tạo data giả. **Không recommend** cho pairs trading cointegration.

**Drop NaN / Inner join** chỉ giữ timestamps cả hai stocks đều có valid prices. Sạch nhất về mặt thống kê — mỗi obs đại diện giá thực, đồng thời. OLS beta estimate unbiased. Nhược điểm: giảm sample size (nếu một cổ phiếu illiquid, có thể mất đáng kể), tạo **uneven time spacing** ảnh hưởng lag structure trong ADF augmented component.

**Best practice — hybrid approach:**

1. Bắt đầu với inner join làm baseline
2. Nếu sample loss > 10–15%, áp dụng limited ffill (tối đa 5 phút) rồi inner join
3. Cổ phiếu nào inner join loại > 20% minutes → flag để loại khỏi universe
4. Track "alignment rate" = matched minutes / total expected minutes cho mỗi pair

**Xử lý opening auctions:** 5–15 phút đầu (9:30–9:45 ET) có volatility bất thường, spread rộng, giá chịu ảnh hưởng auction mechanism. NYSE data cho thấy 15 phút đầu chiếm 11–13% total daily volume với phần lớn từ opening auction print. **Recommendation: loại 5–10 phút đầu** (dùng data từ 9:35 hoặc 9:40).

**Xử lý closing auctions:** Closing auction là **single largest liquidity event** trong ngày (BMLL analysis). Giá convergence giữa indicative và continuous thường xảy ra 3:57–3:58 PM trên Nasdaq. Institutional rebalancing dominates, không phản ánh informed trading. **Recommendation: loại 5 phút cuối** (từ 15:56).

**Sau khi exclude auctions:** Usable minutes = ~370/ngày (9:35–15:55) thay vì 390, mất ~5% data nhưng quality tăng đáng kể.

**Tác động của misalignment lên β̂:** Đây chính xác là errors-in-variables / attenuation bias. Nếu X_t measured with error η_t, OLS β̂ → β × σ²_X/(σ²_X + σ²_η) — always biased toward zero. Stock & Watson (1993) đề xuất **Dynamic OLS (DOLS)** — thêm leads và lags của ΔX_t vào cointegrating regression — để correct endogeneity và tạo asymptotically efficient estimates. Đây là alternative tốt cho Engle-Granger step 1 khi nghi ngờ measurement error.

---

## 6. Data cleaning rules cho 500 pairs: thresholds cụ thể từ literature

Mục tiêu: loại bỏ dữ liệu không đáng tin cậy mà không over-filter. Dưới đây là các rules có precedent trong academic papers.

**Volume filter** — Gatev et al. (2006) yêu cầu "stocks traded every day" trong 12-month formation period, tương đương 100% daily completeness. Cho minute-level, thresholds phổ biến:

| Mức | Average daily dollar volume | Sử dụng |
|-----|---------------------------|---------|
| Conservative | > $5M | Large-cap only, safest |
| Moderate | > $1M | Standard cho academic study |
| Aggressive | > $500K | Bao gồm mid-cap, nhiều noise hơn |

**Recommendation: $1M–$5M daily dollar volume.** Cổ phiếu $1M/day trade ~$2,500/phút trung bình — đủ cho position sizing nhỏ trong research context.

**Price filter** — SEC định nghĩa penny stocks là < $5. Ở giá thấp, minimum tick $0.01 chiếm phần trăm lớn (1% cho $1 stock vs 0.01% cho $100 stock), tạo discrete jumps vi phạm continuous-price assumption. **Loại cổ phiếu < $5, prefer > $10** cho minute-level analysis. Do & Faff (2010, 2012) documented rằng illiquidity là primary constraint ăn mòn pairs trading profitability.

**Outlier removal trên returns** — Financial returns fat-tailed, nên threshold z-score phải đủ rộng:

- **|z| > 10σ**: Chỉ loại extreme errors (data feed glitches, erroneous prints). Safest threshold, gần như chắc chắn là lỗi dữ liệu
- **|z| > 5σ**: Moderate — bắt anomalous moves nhưng giữ fat-tail structure  
- **|z| > 3σ**: Quá aggressive cho financial data — loại ~0.3% dưới normal nhưng fat-tailed returns thực sẽ mất nhiều hơn

**Recommendation:** Primary filter |z| > 10σ (loại data errors), secondary treatment **winsorize** tại 0.1st/99.9th percentile thay vì remove (giữ nguyên sample size cho time series continuity trong ADF). **Không dùng IQR 1.5× cho financial data** — quá aggressive với fat tails.

**Minimum data completeness** — Cho 6 tháng (~49,140 expected minutes), yêu cầu **≥ 95% completeness** sau khi exclude opening/closing. Gatev et al. (2006) yêu cầu 100% ở daily level. Ở minute level, 100% unrealistic ngay cả cho liquid stocks (brief halts, no-trade minutes). Dưới 80% → loại khỏi universe.

**Corporate actions** — **Bắt buộc dùng split-adjusted và dividend-adjusted prices.** Unadjusted split tạo price discontinuity phá hủy cointegration relationship. Unadjusted dividends tạo drops giả trên ex-dates. M&A là structural break — loại pair ngay khi merger announced. Verify adjustment quality: occasionally splits bị delay trong databases. Gatev et al. (2006) dùng cum-dividend prices với reinvested dividends.

---

## 7. Năm dấu hiệu cần loại asset khỏi universe

Không phải cổ phiếu nào cũng phù hợp cho minute-level cointegration analysis. Dưới đây là criteria rõ ràng.

**Illiquid assets với bid-ask spread > 1% giá trung bình** nên loại ngay. Do & Faff (2012) ước lượng institutional commissions ~10 bps; nếu spread alone 100+ bps, round-trip cost ~200+ bps vượt xa typical pairs trading returns **30–90 bps/tháng**. Median daily volume < $500K cũng là strong exclusion signal.

**Assets có quá nhiều zero-return minutes** là metric staleness quan trọng nhất cho minute data. Bandi et al. (2020) chứng minh zero returns phản ánh "genuine economic phenomenon linked to trading volume and liquidity." Thresholds:

- **< 20% zero-return minutes**: Bình thường cho liquid stocks — include
- **20–40%**: Moderately illiquid — include với cảnh báo
- **40–50%**: Illiquid — cointegration test unreliable, xem xét downsample về 5-min
- **> 50%**: Quá illiquid cho minute-level analysis — **loại khỏi universe**

Khi > 50% returns bằng 0, ffill tạo "staircase" pattern → OLS beta bị extreme attenuation bias → ADF test gần như vô dụng.

**Assets vừa IPO (< 12 tháng history)** chưa đủ price discovery. IPO prices chịu ảnh hưởng underwriter stabilization, greenshoe option, lock-up expiration. Relationship với sector peers chưa ổn định. Cointegration đòi hỏi long-run equilibrium — **loại stocks < 12 tháng trading history**, minimum tuyệt đối 6 tháng.

**Assets gần delisted** — giá dưới $1 kéo dài (NYSE/Nasdaq gửi compliance notice sau 30 ngày liên tục dưới $1), volume suy giảm trend (3-month average < 50% 12-month average). Price dynamics bị dominated bởi speculation về survival, không phản ánh fundamental value — cointegration relationship với bất kỳ partner nào đều invalid.

**Assets có structural breaks rõ ràng** — Dùng **Gregory-Hansen test** (1996) cho cointegration với unknown structural break, vì standard Engle-Granger mất power nghiêm trọng khi có break. Recursive **CUSUM monitoring** trên cointegrating residual là fastest way phát hiện emerging breaks. Chow test khi biết breakpoint cụ thể (ví dụ ngày công bố earnings shock). Flag pair nào CUSUM cross 5% significance boundary → manual review hoặc automatic exclusion.

---

## 8. Cleaning nhanh vs over-cleaning: tìm điểm cân bằng

Đây là trade-off cốt lõi quyết định chất lượng nghiên cứu: clean quá ít → spurious cointegration; clean quá nhiều → miss genuine pairs và bias universe.

**Survivorship bias từ over-filtering** nghiêm trọng hơn nhiều người nghĩ. Aggressive filtering loại bỏ mid-cap, small-cap — chính xác là nơi market inefficiencies (và profitable pairs) likely tồn tại vì ít institutional coverage. QuantRocket analysis cho thấy nhìn lại 10+ năm, **hơn 40% stocks đang trade historically đã bị delist** — dùng current constituents để backtest inflates results đáng kể. Backtest trên survivorship-biased data cho 15% annual returns; include delisted companies drops actual returns xuống ~8%.

**Mất statistical power khi giảm search space** — Với N stocks, có N(N-1)/2 potential pairs. N=100 → 4,950 pairs. N=50 sau filter → chỉ 1,225 — giảm **75%** search space. Fewer pairs = fewer opportunities phát hiện genuinely cointegrated relationships. Miao (2014) bắt đầu 177 stocks (15,576 pairs), pre-filter Pearson ≥ 0.9 giảm còn ~1,378, chọn top 10 để trade — mỗi bước filter loại bỏ phần lớn candidates.

**False negative rate tăng khi over-filter** — Harlacher (2016, University of St. Gallen) chứng minh rõ ràng: **Bonferroni correction dẫn đến selection quá conservative và cản trở phát hiện even truly cointegrated combinations.** Một số pairs có data quality hơi kém (volume slightly lower) vẫn có valid cointegration — loại chúng tăng false negative rate.

**Multiple testing problem** là mặt ngược lại: test 500 pairs ở 5% significance → **~25 expected false positives** purely by chance. Ba giải pháp:

- **Bonferroni**: adjusted α = 0.05/500 = 0.0001. Quá conservative — Harlacher (2016) chứng minh nó loại cả true pairs
- **Benjamini-Hochberg FDR**: Control **proportion** of false discoveries thay vì probability of any false positive. Less conservative, retain more true discoveries. **Recommend cho pairs trading**
- **Practical two-step**: Pre-partition universe theo sector/industry (Do & Faff 2010 cho thấy within-industry pairs converge tốt hơn), hoặc dùng ML clustering (Sarmento & Horta 2020 — DBSCAN/OPTICS), rồi chỉ test cointegration trên surviving groups. Giảm số tests → giảm multiple testing burden

**Data snooping checks** — Lo & MacKinlay (1990) chứng minh "slight prior information from data has dramatic impact on test validity." White's Reality Check (2000) và Hansen's SPA test (2005) là bootstrap-based tests kiểm tra strategy performance có survive data-snooping hay không. Caldeira & Moura (2013) dùng White's RC thành công cho cointegration-based pairs trading. Sullivan, Timmermann & White (1999) áp dụng RC cho 7,846 technical rules và thấy profitability "largely disappears" sau correction.

**Điểm cân bằng recommended:** Apply moderate filters (không extreme), pre-partition by sector, dùng BH-FDR cho multiple testing, và validate out-of-sample. Formation 12 tháng + trading 6 tháng per Gatev et al. (2006). Walk-forward analysis với rolling windows.

---

## Recommended data-cleaning workflow cho Jupyter notebook

Thực hiện theo thứ tự — mỗi bước phụ thuộc bước trước.

**Bước 1 — Raw data ingestion & format check.** Load minute-level OHLCV data. Verify timezone consistency (tất cả ET). Parse timestamps, set DatetimeIndex. Confirm data source cung cấp **split-adjusted, dividend-adjusted prices.** Nếu dùng Yahoo Finance hoặc tương tự, dùng "Adj Close."

**Bước 2 — Filter trading hours & exclude auctions.** Chỉ giữ regular hours 9:30–16:00 ET. Loại pre-market và after-hours. Loại 5–10 phút đầu (từ 9:30) và 5 phút cuối (đến 16:00). Usable window: **9:35–15:55** (~370 minutes/ngày).

**Bước 3 — Universe-level screening (loại assets không phù hợp).** Áp dụng filters theo thứ tự:
1. Price filter: loại stocks có median price < $5 trong period
2. Volume filter: loại stocks có average daily dollar volume < $1M
3. History filter: loại stocks có < 12 tháng trading history (hoặc < formation period length)
4. Completeness filter: loại stocks có < 95% expected minutes với valid prices (sau bước 2)
5. Staleness filter: loại stocks có > 50% zero-return minutes (>40% nếu conservative)
6. Delisting/M&A filter: loại stocks có pending delisting notice hoặc announced M&A

**Bước 4 — Outlier treatment trên returns.** Tính minute returns cho mỗi stock. Remove returns |z| > 10σ (data errors). Winsorize remaining returns tại 0.1st/99.9th percentile. Replace removed/winsorized prices bằng recalculated levels từ cleaned returns.

**Bước 5 — Log transformation.** Áp dụng log() trên adjusted prices. Verify không có giá ≤ 0 (nếu có → đã bị loại ở bước 3).

**Bước 6 — Pairwise alignment.** Cho mỗi pair (i, j): inner join trên timestamp. Nếu sample loss > 15%, áp dụng limited ffill (max 5 bars = 5 phút) rồi inner join lại. Nếu vẫn loss > 20% → skip pair. Record alignment_rate cho mỗi pair.

**Bước 7 — Resample nếu cần.** Default: resample về **5-minute bars** (last price trong mỗi 5-min window). Giữ 1-min chỉ nếu cả hai stocks extremely liquid (>$10M daily volume) VÀ dùng mid-prices. Option: test ở cả 1-min và 5-min, compare kết quả.

**Bước 8 — Pre-partition universe.** Group stocks theo GICS sector/industry. Chỉ form pairs within same sector hoặc related sectors. Giảm đáng kể số pairs cần test → giảm multiple testing burden.

**Bước 9 — Engle-Granger cointegration test.** Step 1: OLS regression log(P_A) = α + β·log(P_B) + u. Step 2: ADF test trên residual u. Lưu ý: test cả hai chiều (A on B và B on A) vì Engle-Granger có asymmetry problem. Alternatively, dùng Johansen test để tránh vấn đề này.

**Bước 10 — Multiple testing correction.** Áp dụng Benjamini-Hochberg FDR trên tất cả p-values, target FDR = 5% hoặc 10%. Report cả raw p-values và BH-adjusted p-values.

**Bước 11 — Validation & robustness.** Cross-validate cointegration ở multiple frequencies (1-min, 5-min, 15-min). Pairs cointegrated ở nhiều frequencies → higher confidence. Chạy CUSUM trên residual để check structural stability. Estimate half-life và verify nó reasonable cho intended strategy.

---

## Pre-test checklist cho minute-level equity data

Dán checklist này vào đầu notebook trước khi chạy bất kỳ cointegration test nào.

```
# ============================================================
# PRE-TEST CHECKLIST — Minute-Level Cointegration Analysis
# ============================================================
# Tick each box [x] before proceeding to cointegration testing.
#
# DATA INTEGRITY
# [ ] Prices are split-adjusted AND dividend-adjusted
# [ ] Timezone consistent across all stocks (ET)
# [ ] No duplicate timestamps in any series
# [ ] Prices > 0 for all observations (required for log transform)
# [ ] Data source documented (vendor, API, date range)
#
# TEMPORAL FILTERING
# [ ] Only regular trading hours retained (9:30-16:00 ET)
# [ ] Pre-market and after-hours data excluded
# [ ] First 5-10 minutes excluded (opening auction effect)
# [ ] Last 5 minutes excluded (closing auction effect)
# [ ] Holidays and half-days handled correctly
#
# UNIVERSE SCREENING
# [ ] Median price >= $5 (prefer >= $10)
# [ ] Average daily dollar volume >= $1M
# [ ] Trading history >= 12 months (or >= formation period)
# [ ] Data completeness >= 95% of expected minutes
# [ ] Zero-return minute proportion < 50% (prefer < 30%)
# [ ] No pending delisting or announced M&A
# [ ] No recent IPO (< 6-12 months)
#
# OUTLIER TREATMENT
# [ ] Returns with |z| > 10σ identified and removed (data errors)
# [ ] Remaining returns winsorized at 0.1/99.9 percentile
# [ ] Prices recalculated from cleaned returns
# [ ] Outlier removal rate documented (should be < 0.1%)
#
# TRANSFORMATION
# [ ] Log prices computed (default) OR raw prices justified
# [ ] Decision note on raw vs log documented in notebook
#
# ALIGNMENT & RESAMPLING
# [ ] Pairwise inner join on timestamps performed
# [ ] Alignment rate recorded for each pair (target > 85%)
# [ ] Forward fill limited to max 5 bars if used
# [ ] Resampled to 5-min (default) OR 1-min justified
# [ ] Time span >= 10x expected half-life
#
# STATISTICAL SETUP  
# [ ] Engle-Granger tested in both directions (or Johansen used)
# [ ] ADF lag length selected by AIC/BIC (not arbitrary)
# [ ] Multiple testing correction planned (Benjamini-Hochberg FDR)
# [ ] Formation/trading period split defined (e.g., 12mo/6mo)
# [ ] Significance level chosen considering sample size inflation
#
# ROBUSTNESS
# [ ] Cross-frequency validation planned (1-min, 5-min, 15-min)
# [ ] Structural break test available (CUSUM or Gregory-Hansen)
# [ ] Half-life sanity check planned
# [ ] Out-of-sample validation designed
# ============================================================
```

---

## Tổng hợp: những điều không nên quên

Ba insight quan trọng nhất từ nghiên cứu này mà nhiều practitioners bỏ qua. Thứ nhất, **98,000 observations/năm không phải điều tốt** cho cointegration testing — nó cho ADF power phát hiện microstructure noise thay vì economic relationship. Giải pháp: resample về 5-min và cross-validate ở multiple frequencies; nếu pair chỉ cointegrated ở 1-min nhưng không ở 5-min hay 15-min, đó rất likely là artifact. Thứ hai, **Engle-Granger có asymmetry problem** — kết quả phụ thuộc cổ phiếu nào là dependent variable. Test cả hai chiều hoặc dùng Johansen; nếu kết quả khác nhau đáng kể giữa hai chiều, đó là warning sign. Thứ ba, **Benjamini-Hochberg FDR tốt hơn Bonferroni** cho pairs trading — Bonferroni quá conservative loại cả true pairs (Harlacher 2016), trong khi BH cho phép controlled proportion of false discoveries, phù hợp hơn với bài toán scan nhiều pairs. Kết hợp với pre-partition theo sector để giảm total tests, chiến lược này balance giữa discovery rate và false positive control hiệu quả nhất cho assignment context.