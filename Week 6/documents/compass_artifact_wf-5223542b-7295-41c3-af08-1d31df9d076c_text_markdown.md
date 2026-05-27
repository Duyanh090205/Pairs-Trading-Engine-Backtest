# ĐÁNH GIÁ CHUYÊN SÂU PRE-DEPLOYMENT: COINTEGRATION PAIRS TRADING STRATEGY TRÊN S&P 500 (Week 5, V2.0)

*Người viết: Senior quant practitioner – tư cách như sắp ký off book đi production*

**Bottom line up front (BLUF):** Strategy hiện tại KHÔNG phải một hệ thống nhất quán — verdict **(C) PATCHWORK chưa thành system**. Có một số mảnh tốt (45-fold walk-forward, V2.0 bug audit, microstructure cost thực đo), nhưng các component **đang fight nhau** ở ba điểm nghiêm trọng: (1) Kalman được giữ với δ=1e-7 tức về bản chất là static PCA — tên gọi mâu thuẫn với cơ chế; (2) chiến lược "no stop-loss vì Z=3.5 sẽ revert" gãy đúng ở regime Late Bull 2025-26 (Sharpe −0.750) — không có safety net khi giả định fail; (3) strategy biết trước 3 kill zones có cost > alpha nhưng **chưa code-in time-of-day mask**, tức tự nguyện đốt tiền vào friction mỗi ngày. Cộng thêm DSR ≈ 0.000 với n_trials=50 và 23/45 (51%) folds zero-trade, đây là một research artifact thú vị, chưa phải một production book.

---

## PHẦN 1 — FRAMEWORK 9 DIMENSIONS ĐỂ ĐÁNH GIÁ MỘT PAIRS TRADING ENGINE

Trước khi mổ xẻ strategy cụ thể, đây là khung 9 chiều mà tôi dùng khi đứng giữa research team và risk committee. Mỗi chiều có (a) định nghĩa beginner-friendly, (b) KPI, (c) red flag threshold, (d) action khi trigger.

### 1.1. Pair Selection & Signal Generation
**Định nghĩa.** Cách bạn chọn cặp cổ phiếu và sinh tín hiệu. **Cointegration** (hai chuỗi giá có một combination ổn định dài hạn, như hai con chó buộc dây co giãn — chạy lung tung nhưng không xa nhau quá) là nền tảng. Hai trường phái chính: Engle-Granger (test 1-cặp, đơn giản), Johansen (đa biến, phát hiện được nhiều vector cointegration cùng lúc, mạnh hơn nhưng dễ false-positive khi biến near-integrated — IMF Working Paper 915 (Lütkepohl et al., 2007): "the probability of falsely concluding that completely unrelated series are cointegrated is generally substantially higher than the nominal size"). **Hedge ratio** (tỷ lệ β giữa 2 leg) có thể dùng OLS (tĩnh), Kalman (động, hedge ratio drift theo thời gian), hoặc PCA (eigenvector thứ hai của ma trận covariance).

- **KPI:** Tỷ lệ pair pass post-hoc (out-of-sample) stationarity test; mean half-life của spread; tỷ lệ pair survive persistence retest.
- **Red flag:** >50% pair "pass" formation period nhưng break stationarity trong 1 tháng trading; hoặc <5% pair survive end-to-end funnel với universe lớn (S&P 500 → expect hàng trăm cặp candidate).
- **Action:** Thêm persistence gate, thắt p-value, hoặc chuyển sang factor-residual cointegration (Avellaneda-Lee 2010 *Quantitative Finance*).

### 1.2. Execution Architecture
**Định nghĩa.** Đường đi từ "signal đỏ" đến "fill ở sàn". Bao gồm data feed (latency, completeness), OMS (Order Management System), **legging risk** (mở leg long trước leg short hoặc ngược lại → tạm thời market-exposed), order type (market/limit/IOC/peg), và slippage measurement.

- **KPI:** Median slippage vs midpoint tại entry; tỷ lệ trade bị "half-filled" (chỉ một leg fill); fill latency p99.
- **Red flag:** Slippage > 50% spread quoted; legging risk không có hedge fallback.
- **Action:** Implementation shortfall execution (giá trị benchmark là decision-time mid), TWAP/VWAP để giảm impact, hoặc paired limit orders với timeout.

### 1.3. Cost Reality
**Định nghĩa.** Cost thực, không phải cost giả định. Bao gồm spread (quoted vs effective — effective spread thường thấp hơn nếu price improvement; cao hơn nếu impact), **impact cost** (price drift khi order ăn liquidity), **borrow** (cost mượn cổ phiếu để short, theo Regulation SHO của SEC), và rebalance drag.

- **KPI:** Round-trip (RT) cost in bps theo cụm κ (kappa) tiers; impact-to-spread ratio.
- **Red flag:** RT cost > 50% expected alpha per trade; impact > 30% total friction (signal impact model underestimate).
- **Action:** Tier sizing theo liquidity, throttle khi spread mở rộng, hoặc bỏ tier κ>0.8.

### 1.4. Capacity & Scaling
**Định nghĩa.** Bao nhiêu AUM trước khi strategy tự ăn alpha của mình. Ràng buộc bởi **ADV** (Average Daily Volume) constraint (thường <5-10% ADV per side trên 1 ngày).

- **KPI:** Notional/ADV per pair-day; marginal impact tại 2x, 5x size.
- **Red flag:** >10% ADV ở pair median.
- **Action:** Cap AUM, throttle vào illiquid pair, hoặc participation-rate cap.

### 1.5. Risk Infrastructure
**Định nghĩa.** Hệ thống giới hạn: gross/net exposure, per-name dollar cap, **correlation cluster** (nhiều pair cùng phụ thuộc 1 factor), factor exposure (market β, sector, momentum).

- **KPI:** Gross/NAV ratio, single-name exposure max, top 5 factor loadings.
- **Red flag:** Single-factor loading > 0.3 (strategy không market-neutral thực sự); single-name aggregate > 10% NAV.
- **Action:** Project out PC1-5 hoặc Fama-French factors trước khi entry (Avellaneda-Lee approach).

### 1.6. Failure Mode Awareness
**Định nghĩa.** Biết trước strategy chết kiểu gì. Cổ điển: factor crowding (August 2007 quant quake — Khandani & Lo 2011 *Journal of Financial Markets*), correlation breakdown trong crisis, locate stress (GME 2021 với borrow fee bùng lên 29.32% existing và 50% new short positions theo S3 Partners báo cho CNBC ngày 29/1/2021), regime breaks.

- **KPI:** Stress test P&L với scenario cụ thể; rolling correlation breakdown alert.
- **Red flag:** Strategy có drawdown > 2x worst historical trong stress test.
- **Action:** Pre-commit kill switch, leverage cap dynamic theo VIX/factor crowding metric.

### 1.7. Live Monitoring & Decay Detection
**Định nghĩa.** Hệ thống cảnh báo khi strategy bắt đầu chết.

- **KPI:** Rolling 21-day Sharpe vs backtest distribution; hit rate vs expected; spread half-life drift.
- **Red flag:** Realized Sharpe < 5th percentile của backtest distribution trong 2 tháng liên tiếp.
- **Action:** Halve sizing, escalate research review, hoặc pause new entries.

### 1.8. Operational Risk
**Định nghĩa.** Corporate actions (split, merger, spinoff), dividend ex-date, trading halt, settlement (T+1 từ 2024), earnings season volatility.

- **KPI:** Corporate-action adjustment QA pass rate; pre-earnings halt rule coverage.
- **Red flag:** Backtest không adjust splits/dividends → look-ahead in returns.
- **Action:** Vendor như CRSP point-in-time, blackout window quanh earnings.

### 1.9. Statistical Robustness
**Định nghĩa.** Backtest có ý nghĩa hay là noise + selection bias.

- **KPI:** **DSR — Deflated Sharpe Ratio** (Bailey & López de Prado 2014, *Journal of Portfolio Management* 40(5): 94-107 — deflate cho non-normality và multiple testing), **PBO — Probability of Backtest Overfitting** (Bailey-Borwein-López de Prado-Zhu 2014, *Notices of AMS* 61(5), CSCV method), MinBTL formula (Bailey et al. 2014, Pseudo-Mathematics paper, Eq. 3.2): **MinBTL < 2·ln(N)/E[max_N]²** years.
- **Red flag:** DSR p-value > 0.05; PBO > 0.5; tổng trades < 100 cho strategy daily.
- **Action:** Reduce variants tested, mở rộng OOS data, hoặc kill strategy.

---

## PHẦN 2 — ĐÁNH GIÁ STRATEGY THEO TỪNG DIMENSION

### 2.1. Pair Selection & Signal Generation
- ✅ **ĐIỂM MẠNH:** Funnel 7-stage có persistence gate (re-test Johansen ở last month của formation, min-bars=200) — đây là điểm hiếm thấy trong open-source pairs trading code; nó loại bỏ "fake cointegration" — pair pass test ở formation nhưng đã mất cointegration trước trading. BH-FDR q=0.05 cho multiple testing là practitioner-grade (chứ không phải student-grade dùng nominal p=0.05 với 100,000+ pair test).
- ⚠️ **ĐIỂM YẾU:** Hedge ratio chọn là **secondary eigenvector của PCA** (tức Kalman δ=1e-7 chính là PCA hedge ratio static trá hình). Trong Avellaneda & Lee (2010) *Quantitative Finance* 10(7), residual của factor-PCA mới nên là cái mean-revert — không phải spread của 2 stock thô. Strategy này skip bước factor-orthogonalisation, nên 16.7% trade β<0 chính là pair "co-trending" (move cùng chiều, không phải arbitrage thực) — đáng nghi.
- 📊 **BACKUP:** Khandani & Lo (2011): "the contrarian strategy of Lehmann (1990) and Lo and MacKinlay (1990)" được unwind toàn thị trường tháng 8/2007 vì pairs trading naive crowd quá. Lesson: pair selection mà không project out common factor là **factor-crowded by construction**.
- 🎯 **VERDICT:** **CRITICAL** — cần factor-residual cointegration trước khi go live.

### 2.2. Execution Architecture
- ✅ **ĐIỂM MẠNH:** Week 5 đo cost từ `data/orderbook.parquet` (real LOB) — đây là quant-fund-grade. Hầu hết retail backtest dùng 5-10 bps fixed.
- ⚠️ **ĐIỂM YẾU:** Latency convention "decide-at-close, fill-at-close" là **aggressive** — không có realistic latency (50-200 ms cho NYSE/Nasdaq). Bug-fix audit có 8 bug load-bearing được phát hiện *sau* 5 tuần research → QA discipline chưa fit production. Knight Capital 2012 là cảnh báo: theo SEC Administrative Proceeding (October 2013), "a technician forgot to copy the new Retail Liquidity Program (RLP) code to one of the eight SMARS computer servers... Orders sent with the repurposed flag to the eighth server triggered the defective Power Peg code still present on that server... resulting in 4 million executions in 154 stocks for more than 397 million shares in approximately 45 minutes. Knight Capital took a pre-tax loss of $440 million." Dù strategy đúng, sai sót deployment có thể bankrupt firm.
- 📊 **BACKUP:** Knight Capital SEC Order Aug 2012; SEC Administrative Proceeding Oct 2013.
- 🎯 **VERDICT:** **CRITICAL** — phải có SOP code review/deployment checklist, latency floor 100 ms.

### 2.3. Cost Reality
- ✅ **ĐIỂM MẠNH:** Mean dynamic RT cost 45.3 bps (vs static giả định 90.87 bps) — chứng tỏ assumption Week 4 *conservative*, cost thật rẻ hơn. L1 spread thực 10-12 bps (tighter hơn synthetic 19-33 bps). Đây là hiếm: hầu hết retail strategy giả định cost thấp → live thất vọng. Strategy này ngược lại.
- ⚠️ **ĐIỂM YẾU:** Impact cost từ 8.5% (synthetic) → 23.7% (thực) của total friction — model impact của bạn underestimate gần 3x. Đây là chỉ báo capacity bị overestimate.
- 📊 **BACKUP:** Gatev, Goetzmann, Rouwenhorst (2006) *Review of Financial Studies* 19(3): excess return của top-20 pair giảm từ 118 bp/month pre-1989 xuống 38 bp/month post-1989 — pair alpha bị transaction cost ăn dần qua thập kỷ.
- 🎯 **VERDICT:** **CRITICAL** — cần impact model recalibration, có thể giảm size mỗi pair.

### 2.4. Capacity & Scaling
- ✅ **ĐIỂM MẠNH:** $20k/pair × max 50 pair = $1M gross — quy mô research, không phải production. Ở scale này, capacity là non-issue.
- ⚠️ **ĐIỂM YẾU:** Không có capacity study chính thức (notional vs ADV ratio per pair). Khi scale lên $100M, impact 23.7% của friction sẽ scale **siêu tuyến tính** (sqrt(volume) impact model).
- 📊 **BACKUP:** Renaissance Medallion thought to be capped ~$10-15B vì capacity constraint (Zuckerman 2019 *The Man Who Solved the Market*).
- 🎯 **VERDICT:** **IMPORTANT** — defer đến khi alpha xác lập, nhưng cần biết.

### 2.5. Risk Infrastructure
- ✅ **ĐIỂM MẠNH:** Dollar-neutral β-weighted, max 50 pair, $20k/pair — caps cứng.
- ⚠️ **ĐIỂM YẾU:** **Không có factor exposure monitor.** Pair selection không project out PC1-5 hoặc Fama-French — strategy có thể vô tình long factor (e.g., low-vol vs high-vol pair → long low-vol short high-vol = momentum-anti). 16.7% trade β<0 = "dollar-neutral" trên giấy nhưng 2 leg cùng hướng → cần reclassify: định nghĩa "dollar-neutral" gãy ở đây.
- 📊 **BACKUP:** Khandani & Lo (2011): các fund quant August 2007 đều "dollar-neutral β-weighted" nhưng unwind đồng loạt vì common factor exposure ngầm.
- 🎯 **VERDICT:** **CRITICAL** — cần factor-decompose mỗi pair trước entry.

### 2.6. Failure Mode Awareness
- ✅ **ĐIỂM MẠNH:** Doc Week 5 thừa nhận "Late Bull factor momentum failure mode" — biết mode chết.
- ⚠️ **ĐIỂM YẾU:** Biết nhưng *không fix*. Z-velocity filter proposed but not implemented. Cũng không có locate stress test (GME 2021 borrow fee jumped — nếu strategy mượn short, P&L wipeout có thể từ borrow chứ không phải price). No crisis correlation breakdown test.
- 📊 **BACKUP:** GameStop 2021 — S3 Partners báo CNBC (29/1/2021): "The borrow fee on GameStop's stock — or the cost-to-borrow shares for the purpose of selling them short — jumped to 29.32% on existing shorts and 50% on new short positions." LTCM 1998 — convergence trade widened thay vì narrow vì flight-to-liquidity. Khi strategy không có safety net (no SL), những event này = catastrophic.
- 🎯 **VERDICT:** **CRITICAL**.

### 2.7. Live Monitoring & Decay Detection
- ⚠️ **ĐIỂM YẾU:** Strategy chưa go live → chưa có. Nhưng spec không describe alert thresholds. 23/45 zero-trade fold = nếu sự kiện này lặp 4 tháng đầu live, có pause hay không?
- 🎯 **VERDICT:** **IMPORTANT** — define KPI live before deploy.

### 2.8. Operational Risk
- ⚠️ **ĐIỂM YẾU:** Survivorship bias (current S&P 500 universe, not point-in-time). Theo CRSP-based study (LuxAlgo, 2026, citing CRSP US Stock Database): survivorship-free vs survivorship-biased = 7.4% vs 9.0% annualized return (1926-2001) — gap 1.6%/năm. Andrikogiannopoulou & Papakonstantinou: hedge fund drawdown bị underestimate trung bình 14 percentage points khi có survivorship bias. Strategy này có cùng vấn đề: pair từ S&P 500 hiện tại exclude các stock bị delist trong 2022-2026, biased về phía "survivor pair".
- 🎯 **VERDICT:** **IMPORTANT** — chuyển sang point-in-time membership từ CRSP.

### 2.9. Statistical Robustness
- ✅ **ĐIỂM MẠNH:** 45-fold walk-forward là practitioner-grade. PBO=0.030 là rất tốt (theo Bailey-Borwein-López de Prado-Zhu 2014 *Notices of AMS* 61(5): PBO < 0.5 là baseline, < 0.1 là tốt).
- ⚠️ **ĐIỂM YẾU:** **DSR ≈ 0.000 với n_trials=50, E[max SR]=2.054, observed SR=0.503.** Theo Bailey et al. (2014, *Notices of AMS*, "Pseudo-Mathematics..."), công thức Eq. 3.2: **MinBTL < 2·ln(N)/E[max_N]²**. Với N=50, E[max_N]=1, MinBTL ≈ 2·ln(50)/1² ≈ 7.8 năm. Strategy có ~4.25 năm data (Jan 2022 – Mar 2026), không đủ để defend 50 variants tried. Cộng thêm: 90 trade trên 25 fold = 3.6 trade/fold. Central Limit Theorem cần >30 trades cho basic stat; metric reliable thường cần 100+ (López de Prado 2018 *Advances in Financial Machine Learning* Ch.11, Combinatorial Purged Cross-Validation). Strategy này có 90 trade tổng cho 25 fold — borderline.
- 📊 **BACKUP:** Bailey et al. (2014, *Notices AMS*, p.12, Figure 2 caption): "After trying only 7 independent strategy configurations, the expected maximum SR IS is 1 for a 2-year long backtest, while the expected SR OOS is 0." Pithy rule (p.11): "MinBTL < 2 ln[N] / E[max_N]²."
- 🎯 **VERDICT:** **CRITICAL** — DSR fail là show-stopper.

#### Đánh giá các choice tranh cãi:

**V2.0 audit 8 bug:** Đáng hoan nghênh đã phát hiện, nhưng *điều đáng lo* là chúng tồn tại 5 tuần. BUG-3 β-sign cascade invert 23.9% trade — đây là loại bug mà any sane test suite phải catch trước khi research bắt đầu, không phải sau. Production-ready QA process: unit test cho mỗi component, golden-dataset regression test, paired pair-sign sanity test. *Nghi vấn về 9th bug chưa tìm thấy* — nếu 8 bug load-bearing tồn tại 5 tuần, P(còn ≥1 bug nữa) > 0.

**Z=3.5 (raised từ 3.0 sau bug fix):** Đây là **in-sample optimization** trên cùng dataset bạn đánh giá. Z-sweep global optimum tại Z=3.5 (+1.025 SR), non-monotone tại Z=3.75 (+0.506) — pattern non-monotone là dấu hiệu noise-fitting. *Vấn đề lớn hơn:* Z=3.5 optimum được chọn khi giả định cost=60bps; Week 5 reveal cost=45bps → optimum dịch về Z thấp hơn. Strategy chưa cập nhật.

**δ=1e-7 always (Kalman near-static):** Theo Chan (2013, *Algorithmic Trading*, Ch.3 Example 3.3, pp.77-78) và companion code (KF_beta_EWA_EWC.m, chính Chan release): `Vw = delta/(1-delta) * eye(2)` với comment "delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression)" — khi δ → 0, Vw → 0, β trở thành hằng số. Đây *chính xác* là OLS regression / static PCA. Gọi nó là "Kalman" là **misnomer**, không phải innovation. Strategy có thể đơn giản hóa bằng OLS hoặc PCA hedge ratio, identical kết quả.

**Prior-spread (kurtosis 3-5 vs posterior 13):** Posterior spread = e(t) (residual của Kalman) là cái Chan dùng cho signal trong sách. Prior-spread (predicted spread trước khi update) khác — nó là static hedge ratio applied to current price. Vì δ=1e-7, prior ≈ posterior, nhưng kurtosis khác hẳn (3-5 vs 13). Đây là indicator rằng "Kalman" thực ra không adapt — nếu adapt, prior và posterior phải gần nhau theo trục innovation. Kurtosis 13 ở posterior = fat tail của observation noise, không phải intrinsic spread dynamics. Dùng prior-spread = essentially trading PCA spread → "innovation" này có thể là hack chứ không phải breakthrough.

**CORR25 ≥ 0.25:** Threshold 0.25 *cực lỏng* cho stock returns intra-session (typical S&P 500 same-sector pair > 0.5). Week 4 verified: CORR25 zero marginal trên completed fold (instrumented run). Chỉ eliminate entire fold khi all pair fail — đây là *binary gate* về condition macro chứ không phải pair-level filter. Có thể giữ vì cheap, nhưng không xứng là "stage" trong funnel.

**Persistence gate (min-bars=200):** Innovation thực sự — confirmed primary within-fold filter. Spike fold 11/23/40 từ 1,921/6,008/5,913 pair → 29/17/35. Đây là band-aid hợp lý cho cointegration decay rate cao, nhưng *root cause* (Johansen test có false positive rate cao với near-integrated series — IMF WP 07/141 chỉ ra spurious rejection > nominal size) chưa addressed. Persistence gate cover triệu chứng, không cure bệnh.

**DSR=0 với n_trials=50:** Red flag chí mạng. Theo công thức Bailey-López de Prado 2014: nếu thử 50 variant trên dataset, expected max SR (under null hypothesis Sharpe=0) là 2.054. Observed SR=0.503 << 2.054 → không reject null. **Đây là statistical evidence rằng performance có thể là noise + selection bias.**

**23/45 zero-trade fold (51%):** Half thời gian strategy không trade. Đây không phải "alive book" theo định nghĩa hedge fund — Renaissance Medallion (Zuckerman 2019, *The Man Who Solved the Market*): "Medallion made between 150,000 and 300,000 trades a day"; Two Sigma chạy 110,000+ simulation/ngày (twosigma.com/about-us); Millennium (Motley Fool 2025, citing public filings): "By mid-2025, Millennium had $78 billion in assets under management (AUM), with 6,300 employees in almost 150 locations making more than 13 million trades every day." Strategy 90 trade/25 fold = ~10 trade/tháng. Đây là statistical artifact, không phải production engine.

---

## PHẦN 3 — INTEGRATED PIPELINE EVALUATION (PHẦN QUAN TRỌNG NHẤT)

Đây là phần tôi quan tâm nhất. Component-level có thể tốt, nhưng nếu lắp lại không thành hệ thống thì book này không go live.

### 3.1. Design Philosophy Coherence

**Tóm tắt linh hồn strategy trong 1 câu — và đây là khó.** Sau khi đọc spec, tôi không thể tóm tắt strategy thành 1 câu kiểu *"long-short cointegrated S&P 500 pair với β-adaptive hedge ratio, exit theo mean reversion of innovation"*. Tại sao? Vì các thành phần không kể cùng một câu chuyện:

- **Kalman filter** ngụ ý: "hedge ratio adapt theo thời gian, spread innovation = signal".
- **δ=1e-7 fixed** ngụ ý: "hedge ratio không cần adapt, static PCA là đủ".
- **Prior-spread (kurtosis 3-5)** ngụ ý: "không tin posterior — thực ra trade theo predicted spread, không phải innovation".
- **Z=3.5 + no stop-loss** ngụ ý: "mean reversion sẽ đến, chỉ entry ở extreme".
- **CORR25 ≥ 0.25** ngụ ý: "cần short-term return correlation" — đối lập với spirit cointegration (long-run equilibrium).
- **Persistence gate min-bars=200** ngụ ý: "cointegration ổn định gần đây quan trọng hơn formation period".

Đây là 6 triết lý **khác nhau** đang ngồi cùng pipeline. Strategy này không có một economic thesis rõ — nó là "pair statistical filter + Kalman cosmetic + extreme-Z reversion bet". Economic thesis của Avellaneda-Lee (2010) là: "stock có idiosyncratic component sau khi trừ factor là mean-reverting OU". Của Gatev-Goetzmann-Rouwenhorst (2006) là: "stocks close in normalized distance là economic substitute, temporary mispricing reverts". Strategy này không thuộc nhóm nào — nó là **cointegration test pipeline với risk caps**, không phải economic strategy. Khi tôi hỏi "tại sao 2 stock này nên revert?" thì câu trả lời duy nhất là "Johansen test rejects null at q=0.05" — đây là statistical thesis, không phải economic.

**Đây là Frankenstein, không phải bản giao hưởng.** Mỗi bước trong funnel được vá theo thứ tự V1.0 → V2.0 sau khi bug-fix. Persistence gate ra đời để xử lý spike fold 11/23/40. HL cap [1,6d] ra đời để xử lý no-EOS-flatten gây hold-forever. Z=3.5 ra đời sau khi bug β-sign được fix làm Z=3.0 trông tệ. CORR25 còn lại trong funnel dù zero marginal. Đây là engineering bằng patch — không phải design từ thesis.

### 3.2. Component Interaction Analysis

Phân tích từng cặp component theo 3 loại: REINFORCING, CONFLICTING, REDUNDANT.

#### Reinforcing pairs

**(R1) Persistence gate + HL cap [1,6d].** Cả hai đều address rủi ro pair có cointegration suy yếu giữa formation và trading. Persistence retest đảm bảo cointegration còn lúc cuối formation; HL cap đảm bảo nếu enter, reversion sẽ đến đủ nhanh để hold trong 1-month trading window. Hai stage này thực sự cùng phục vụ một goal — chống "pair pass Johansen nhưng đã chết". Đây là cặp ít hiếm trong strategy mà tôi thấy reinforce thực sự.

**(R2) Z=3.5 + no stop-loss.** Trên giấy hợp logic: nếu chỉ enter ở extreme (Z=3.5 ≈ 0.05% tail của normal), thì Bayesian prior cho reversion là rất mạnh, và stop-loss có thể cut prematurely trước reversion. Nhưng — vấn đề là "Bayesian prior cho reversion mạnh" yêu cầu spread thực sự là mean-reverting với kurtosis ổn định. Posterior kurtosis 13 (rất **leptokurtic** — fat tails — nghĩa là tail event xảy ra thường xuyên hơn normal đáng kể) phản bác giả định này. Late Bull 2025-26 Sharpe −0.750 với 55% trade không revert là **bằng chứng empirical** rằng giả định "Z extreme → reversion" fail trong regime factor-momentum. Khi prior fail, không có safety net là **fatal**.

**(R3) Walk-forward 45-fold + BH-FDR.** Cả hai đều address overfitting. Walk-forward đảm bảo không lookahead; BH-FDR control multiple-testing inflation khi test hàng nghìn pair. Hai cái này reinforce statistical robustness. Tốt.

#### Conflicting pairs (LOAD-BEARING)

**(C1) "Kalman để adapt" vs δ=1e-7.** Đây là contradiction lớn nhất trong design. Bạn implement Kalman 2D state [α, β] với Q=δ·R·I₂; sau đó pin δ ở floor 1e-7 toàn 100% fold. Tại điểm này, Kalman không update — β trở thành hằng số trong fold. Theo Chan (2013, Ch.3, p.77-78) và companion code (KF_beta_EWA_EWC.m): `Vw = delta/(1-delta) * I`; khi δ → 0, Vw → 0, β_t = β_{t-1} = ... = β_0. **Đây là OLS, không phải Kalman.** Hai khả năng: (a) bạn implement Kalman đúng, grid search tự chọn δ=1e-7 vì grid điểm tiếp theo cao (e.g., 1e-3) làm β bay quá — tức data từ chối adaptive hedge, ưu thiên static; (b) bạn không grid đủ tốt. Trường hợp nào cũng vậy: **đặt tên Kalman cho cái static là misleading.** Chi phí thực sự: complexity và bug surface (BUG-3 β-sign cascade liên quan trực tiếp đến Kalman state, không phải bug nếu dùng OLS hoặc PCA). Verdict: rip out Kalman, dùng PCA secondary eigenvector explicit, giảm bug surface, tăng explainability.

**(C2) "No-SL vì Z=3.5 sẽ revert" vs Late Bull regime fail.** Đã touch trong R2 nhưng cần đào sâu. Late Bull 2025-26 11 fold, mean Sharpe −0.750, 45% positive fold, 1/11 NC pass. Đây là 24% sample size (11/45) nhưng cost lớn nhất (61.5 bps RT). Strategy ở regime này: enter ở Z=3.5, không revert (factor momentum kéo dài), không SL → hold đến end of 1-month window, force close ở loss. Nếu có SL ở 2σ thêm vào Z entry (Z=5.5), drawdown sẽ bounded. **Không có SL là philosophical bet rằng prior reversion luôn đúng — bet này fail empirically.**

**(C3) "Dollar-neutral β-weighted" vs 16.7% β<0 pair.** Đây là bug semantic, không phải bug code. Khi β<0, dollar-neutral β-weighted sizing = (long $X leg1) + (long $X·|β| leg2) — tức **cùng hướng cả 2 leg**. Đây không phải pair trade theo định nghĩa cổ điển (long-short pair); đây là long-long basket. P&L behavior khác: thay vì revert spread, bạn trade cùng-chiều move. *Định nghĩa "dollar-neutral" của bạn không khớp với "market-neutral".* Strategy có thể có market β > 0 ở 16.7% trade này. Khi market crash, 16.7% portfolio crash cùng. Recommendation: hard filter β > 0 ở entry, hoặc explicitly carve out β<0 pair với strategy khác.

**(C4) Z=3.5 chọn với cost 60 bps; cost thực 45 bps → optimum dịch.** Trên Z-sweep: Z=3.0 cho +0.144 SR, Z=3.5 cho +1.025 SR (gross). Với cost 60 bps, mỗi trade RT eat 60 bps. Z=3.0 vào nhiều trade hơn → cost ăn nhiều alpha hơn. Z=3.5 vào ít, cost ăn ít. Khi cost giảm xuống 45 bps, equation dịch về Z thấp hơn (vào nhiều trade với cost thấp hơn = OK). *Z=3.5 không còn là optimum.* Có thể Z=3.0 hoặc Z=3.25 mới đúng. Strategy chưa re-optimize. Đây là *integration gap* — Phase 5 cost data không feed back vào Phase 4 threshold.

**(C5) Prior-spread vs posterior-spread.** Kurtosis 3-5 (prior) vs 13 (posterior). Posterior là e(t) = y(t) - β·x(t) — đây là cái Chan dùng làm signal trong sách (Ch.3, Example 3.3). Prior là predicted spread không có observation update. Vì δ=1e-7, "update" hầu như không có → prior ≈ posterior về β nhưng *khác về variance scaling*. Posterior dùng innovation variance từ Kalman; prior dùng static variance. Kurtosis prior thấp = bạn đang trade theo *empirical Z-score của static spread* — đây thực ra là **Bollinger Band trên PCA spread**, không phải Kalman innovation. Có 2 vấn đề: (1) tên gọi sai; (2) bạn đang loại bỏ thông tin Kalman cung cấp (innovation variance theo state).

#### Redundant components

**(D1) BH-FDR + Persistence gate.** Cả 2 đều filter "không thực sự cointegrated". BH-FDR q=0.05 vẫn để 1,921 pair pass ở fold 11 — chứng tỏ FDR control một mình không đủ. Persistence gate đóng vai trò chính. *Verdict: giữ BH-FDR (cheap, mathematically principled) nhưng đừng dựa vào nó.*

**(D2) CORR25 ≥ 0.25.** Zero marginal trên completed fold. Chỉ kill entire fold khi all pair fail. Đây không phải pair-level filter — đây là *regime gate*. Nếu giữ, đặt tên lại là "regime gate" hoặc "macro filter". Tốt hơn: kill, đỡ một stage trong funnel.

**(D3) OU HL filter [1,10d] + HL cap [1,6d].** Hai stage cho cùng metric (half-life). [1,10] đầu rộng, [1,6] sau hẹp. Tại sao không gộp thành 1 stage [1,6]? Có thể có lý do (2 metric khác nhau: 1 từ Johansen vector, 1 từ OLS residual fitting?). Spec không nói rõ. Khả năng cao là legacy.

### 3.3. Funnel Health

7-stage funnel với numbers cụ thể (V2.0, instrumented):

| Stage | Pass | Comment |
|---|---|---|
| Universe | 528 ticker = ~139,000 pair candidate | S&P 500 (survivorship-biased) |
| Hard screens | ? | Liquidity, price, etc. |
| Johansen | ? | Cointegration test |
| BH-FDR q=0.05 | Phase 1 total: 20,035 pair (39 active fold) | ~514 pair/fold avg |
| OU HL [1,10d] | ? | First HL filter |
| HL cap [1,6d] | ? | Second HL filter |
| Persistence gate | Spike fold 11: 1,921 → 29; fold 23: 6,008 → 17; fold 40: 5,913 → 35 | **Primary within-fold filter** |
| CORR25 ≥ 0.25 | ZERO marginal trên completed fold | Eliminate entire fold only |

**Funnel quá hẹp ở final?** Hiển nhiên: 23/45 = 51% fold zero-trade; 90 trade tổng cho 25 fold = 3.6 trade/fold. Đây không phải engine — đây là single-shot research artifact. Để compare:

- **Renaissance Medallion** (Zuckerman 2019, *The Man Who Solved the Market*): "Medallion made between 150,000 and 300,000 trades a day, but much of that activity entailed buying or selling in small chunks to avoid impacting the market prices, rather than profiting by stepping in front of other investors." "Medallion still held thousands of long and short positions at any time. Its holding period ranged from one or two days to one or two weeks."
- **Two Sigma:** "uses 600+ PB of storage capacity and infrastructure to run over 110,000 simulations daily" (twosigma.com/about-us).
- **Millennium / WorldQuant** (Motley Fool 2025, citing public filings): Millennium làm "more than 13 million trades every day"; WorldQuant "has a library of 4 million pieces of predictive code known as 'alphas' that are based on data sets ranging from credit card receipts to parking lot traffic."
- **Industry pairs trading benchmark:** "hundreds to thousands per month" minimum.

Strategy này: ~10 trade/tháng. **Statistical power không đủ.** López de Prado MinBTL: với N=50 strategy variants tested, MinBTL < 2·ln(50)/E[max_N]² = 2·3.91/1² ≈ 7.8 năm cho E[max_N]=1. Strategy có 4.25 năm data → insufficient.

**Theo Bailey-Borwein-López de Prado-Zhu (2014, *Notices of AMS* 61(5), p.12):** "After trying only 7 independent strategy configurations, the expected maximum SR IS is 1 for a 2-year long backtest, while the expected SR OOS is 0." Bạn đã try 50 variant → expected max SR IS = 2.054. Observed = 0.503. Strategy fail absolute test.

**Stage nào kill universe?** Persistence gate confirmed primary (Week 4 instrumented). CORR25 zero marginal nội fold. OU HL filters: spec không quantify nhưng likely material. BH-FDR mathematically tight nhưng vẫn để 1,921 pair pass → không đủ một mình.

**Stage nào 0 effect?** CORR25 trong-fold (verified).

### 3.4. Cost-Behavior Fit

Phần này quan trọng vì nó là chỗ Week 5 reveal **integration failure** cụ thể nhất.

**Optimal Z với cost mới chưa được tính lại.** Z-sweep được run với cost giả định 60 bps (Week 4). Z=3.5 = global optimum. Khi cost giảm xuống 45 bps (Week 5 dynamic), strategy *chưa* re-sweep. Lý do: lower cost → more trade economically viable → Z lower thresh có thể beat. Có thể Z=3.0 hoặc 3.25 outperform Z=3.5 sau khi recompute với 45 bps. Đây là **integration gap số 1**: discovery của Phase 5 không feed back vào Phase 4.

**68% trade enter 10:00-10:30 = concentration risk khủng khiếp.** Đây gross +177 bps net — alpha tập trung. Nhưng risk: nếu market liquidity 10:00-10:30 thay đổi (e.g., NYSE open auction change, hoặc retail flow shift), 68% alpha bay. Khoảng cách 1-window-risk. Không có diversification across thời gian.

**3 kill zones identified nhưng không code-in.** 11:00-11:30 (net −2 bps), 11:30-12:00 (gross −26 bps trước cost!), 15:30-15:59 (net −11 bps). Strategy biết trước những window này là âm alpha. Nhưng spec không có time-of-day mask. Trong production, mỗi ngày trade ở những window này = đốt tiền tự nguyện. Đây là **integration gap số 2** và là chỗ tôi mất ngủ nhất.

**14:00-14:30 outlier:** 1 trade, 7,923 bps gross — đây không phải tín hiệu, đây là sample size 1 ngoài lớp. Cần manual review, có thể là data glitch hoặc earnings event. Nếu loại bỏ, average của 14:00-14:30 có thể là âm. Đây là warning về **regression-to-the-mean trap**: insample outlier ≠ persistent alpha.

**Regime-cost mismatch:** Late Bull có 61.5 bps RT cost (cao nhất) và Sharpe −0.750 (thấp nhất). Tức là **strategy yếu nhất ở regime đắt nhất**. Đây là cost fighting strategy. Không random — có thể explain bằng: Late Bull = factor momentum dominant → pair revert chậm → hold lâu → bid-ask cost compound + impact cost cao hơn khi volatility thấp (đối lập với intuition, nhưng spread mở rộng khi market trend mạnh ở stock idiosyncratic).

**Early Bull 2023 friction-negative:** gross +0.234 → dynamic −0.033. Friction destroys regime. Regime *supposedly works* (5/6 NC pass) bị cost giết. Đây là *strong* signal rằng strategy ở scale này không có alpha sau cost.

### 3.5. Integration Verdict

**1. Remove từng component — strategy có sụp đổ không?**

| Component | Critical? | Lý do |
|---|---|---|
| Persistence gate | **CRITICAL** | Primary filter; remove → thousands of pair spike fold |
| CORR25 | **VESTIGIAL** | Zero marginal nội fold; chỉ regime gate |
| Kalman (vs static PCA) | **VESTIGIAL** | δ=1e-7 = static PCA, identical kết quả |
| HL cap [1,6d] | **IMPORTANT** | Without EOS, slow reversion = held forever |
| Z=3.5 vs Z=3.0 | **IMPORTANT** | MaxDD −25.6% → −3.15%; nhưng chọn in-sample |
| BH-FDR | **NICE-TO-HAVE** | Cheap, principled; không đủ một mình |
| OU HL [1,10d] | **REDUNDANT** với HL cap | Duplicate filter |
| No stop-loss | **HARMFUL** | Fail mode trong Late Bull |
| No EOS flatten | **HARMFUL** | Cộng với no SL = hold qua regime change |

**2. Critical glue vs critical gap.**

- **Critical glue (giữ strategy đứng vững):** Persistence gate. Walk-forward 45-fold structure. BH-FDR q=0.05 (cap multiple-testing). V2.0 audit (after-the-fact, nhưng necessary).
- **Critical gap (chỗ pipeline gãy):**
  - **Gap 1:** Z=3.5 chọn với cost giả định cao; không re-optimize với cost thực 45 bps. **Phase 5 data không feed back Phase 4 threshold.**
  - **Gap 2:** Kill zone identified nhưng không code-in time-of-day mask. **Strategy trade vào lúc tự biết âm alpha.**
  - **Gap 3:** Late Bull failure mode được biết (Z-velocity proposed) nhưng không implement. **Strategy không có defense ở regime đã empirically fail.**
  - **Gap 4:** No factor-residual orthogonalisation. **16.7% trade β<0 = silent factor exposure.**
  - **Gap 5:** "Kalman" gọi sai. δ=1e-7 = static PCA. **Misnomer làm bug surface lớn hơn cần thiết.**

**3. Verdict: (C) PATCHWORK chưa thành system.**

Tôi pick C, không hedging, defend bằng evidence cụ thể:

**Argument cho C:**

(a) **Triết lý không nhất quán.** 6 component kể 6 câu chuyện khác nhau (Kalman adapt vs δ-pinned static; cointegration long-run vs CORR25 short-run; mean reversion bet vs no SL). Một system coherent có *một thesis*. Strategy này có *nhiều thesis* đang fight nhau.

(b) **Component được vá theo bug, không theo thesis.** Persistence gate ra đời để xử lý spike fold (engineering response). Z=3.5 chọn sau bug-fix audit (in-sample re-opt). HL cap [1,6d] đè lên OU HL [1,10d] (redundant). Đây là patch sequence, không phải design.

(c) **Integration gap concrete:**
- Z=3.5 không update với cost 45 bps thực;
- Kill zone không code-in;
- Late Bull failure mode no fix;
- No factor orthogonalisation.

(d) **Statistical evidence:** DSR ≈ 0.000 với n_trials=50. PBO=0.030 (tốt) nhưng PBO chỉ measure CV consistency, không measure absolute overfit. DSR là test khắc nghiệt hơn — strategy fail.

(e) **Skew −8.51, excess kurtosis 109.6** (Week 4 fold-level distribution). Đây là *cực kỳ leptokurtic* — left tail cực dày. Strategy bình thường (e.g., trend-following) có kurtosis 3-6. 109.6 nghĩa là có vài fold disaster ngoài lớp kéo mean Sharpe lên (median = 0.000, mean = +0.995 — khoảng cách huge). Strategy không có alpha trên median fold.

**Argument đối lập (steel-manned):**

Có thể nói: "C nặng quá. (B) MOSTLY COHERENT phù hợp hơn vì persistence gate là innovation thực, walk-forward structure tốt, cost measurement tốt." Tôi đồng ý 3 điểm này tốt. Nhưng B yêu cầu **1-2 conflict cần redesign** — ở đây tôi đếm ≥5 conflict load-bearing (C1-C5). Khi conflict số nhiều hơn 2, design không còn "mostly coherent" — nó coherent ở một số dimension và fragmented ở các dimension khác. Đó chính là định nghĩa patchwork.

**Verdict cuối: (C).** Strategy cần re-think fundamental:
1. Decide một thesis: "factor-residual cointegration mean reversion" (Avellaneda-Lee style) hay "raw pair cointegration distance" (Gatev style) — chọn một.
2. Rip Kalman, dùng explicit PCA hoặc OLS hedge ratio.
3. Re-optimize Z với cost 45 bps.
4. Code-in time-of-day mask kill zone.
5. Implement Z-velocity filter cho Late Bull.
6. Migrate sang point-in-time universe.
7. Add factor exposure monitor.

Sau đó re-backtest. *Khi đó* mới đánh giá lại verdict.

---

## PHẦN 4 — SO SÁNH INDUSTRY PRACTICE

### 4.1. Ai còn chạy pairs trading hôm nay (2024-2026)?

**Renaissance Medallion** (Zuckerman 2019, *The Man Who Solved the Market*): "Medallion made between 150,000 and 300,000 trades a day"; "Medallion still held thousands of long and short positions at any time. Its holding period ranged from one or two days to one or two weeks." Stat arb là core, nhưng đa dạng signal — không pure cointegration. Strategy này: 90 trade/25 fold ≈ 3.6 trade/fold. Khoảng cách 5 bậc.

**Millennium Management** (Motley Fool 2025, citing public filings): "By mid-2025, Millennium had $78 billion in assets under management (AUM), with 6,300 employees in almost 150 locations making more than 13 million trades every day." As of January 2026, Millennium manages over $83.5B per SEC filings, with "more than 330 independent investment teams." WorldQuant spinoff (2007) chuyên stat arb với "library of 4 million pieces of predictive code known as 'alphas' that are based on data sets ranging from credit card receipts to parking lot traffic" (Motley Fool 2025). Risk discipline: pod mất 5% capital → cắt một nửa; 7.5% → fired. Đây là discipline strategy này không có (no SL).

**Two Sigma:** "over 110,000 simulations daily, 380+ PB data, 10,000+ data sources" (twosigma.com/about-us). Stat arb là một trong nhiều strategy. Holding intraday đến days.

**D.E. Shaw:** 2024 Composite +18%, Oculus +36.1% — Institutional Investor (January 2025), citing LCH Investments Chairman Rick Sopher: "D.E. Shaw made $11.1 billion for investors in 2024, the most of any of the top 20 managers in the ranking... Last year's strong performance by D.E. Shaw came from its largest multistrategy hedge fund, Composite, which rose 18 percent, while Oculus, the firm's macro-oriented multistrategy fund, jumped 36.1 percent." Stat arb traces về Morgan Stanley APT Group (1985-1989) làm pair trading lần đầu industrial scale; David Shaw từng làm việc dưới Nunzio Tartaglia. Hôm nay vẫn dùng OU, Kalman, regime-switching (D.E. Shaw Risk Management PDF).

**AQR Capital Management** (Form ADV, SEC filing, Dec 31 2025): "As of December 31, 2025, AQR had approximately $187,180,600,000 in Client net assets under management, all of which were managed on a discretionary basis." Factor investing chính, stat arb secondary. Style Premia Fund. Cliff Asness research focus.

**CFM (Capital Fund Management):** trend following + stat arb, ~$10B AUM, Paris-based. Bouchaud's group publish nhiều paper microstructure.

### 4.2. Họ làm khác strategy này ở đâu?

**Capacity & breadth.** Industry trade hundreds-to-thousands pair simultaneously, holding period từ intraday đến vài ngày. Strategy này: max 50 pair, hold mean 1.7 ngày — *holding period đúng* nhưng *breadth thấp 1-2 bậc*. Statistical power không đủ. WorldQuant kết hợp "millions of faint alphas" → "mega-alpha" (Kakushadze 2017, arXiv 1708.02984): "A typical such alpha cannot even be traded on its own – its signal is too weak to make any money after trading costs. So, quant traders follow an ancient 'there is strength in numbers' wisdom and combine a large number of these faint alphas into a single 'mega-alpha'." Strategy này trade single faint alpha → không có "strength in numbers".

**Signals.** Pure cointegration đã chết theo cách Gatev-Goetzmann-Rouwenhorst (2006) đo: top-20 excess return 118 bp/month pre-1989 → 38 bp/month post-1989. Sun (2025, WNE Working Paper 19/2025, U. Warsaw) survey: "the simple, mechanical versions that worked in the 1990s and early 2000s no longer deliver robust returns... [profitability] is much more sensitive to transaction costs and execution." Industry hôm nay dùng **factor-residual cointegration** (Avellaneda-Lee 2010), **deep universe stat arb** (Renaissance), **ML-augmented signals** (Two Sigma, D.E. Shaw). Strategy này pure cointegration trên raw price = thiết kế 2005, không phải 2026.

**Hedging.** Industry chuẩn: factor-neutral hedging — long-short pair sau khi project out PC1-5 (PCA) hoặc Fama-French 5-factor. Strategy này: dollar-neutral β-weighted on raw pair → silent factor exposure như đã đề cập.

**Holding period.** Strategy này 1.7 ngày — đúng range industry. OK.

**Execution.** Industry dùng VWAP/TWAP/implementation shortfall, không phải decide-at-close fill-at-close. Renaissance đặc biệt tối ưu queue position, low-latency colocation. Strategy này latency convention aggressive — nếu go live với realistic latency, có thể slippage tăng 5-10 bps thêm.

### 4.3. Chỗ strategy này TỐT hơn industry average

(Để công bằng — không phải mọi thứ đều xấu.)

1. **Walk-forward discipline với 45 fold.** Đây là practitioner-grade. Hầu hết retail/academic dùng single train-test split → overfit dễ dàng. 45 fold là *gold standard*.

2. **V2.0 bug audit transparency.** Document 8 bug + impact analysis là rất hiếm. Khandani & Lo (2011) viết về "Unwind Hypothesis" của 2007 quake một phần vì *thiếu* transparency từ funds. Strategy này transparent. Production-ready transparency.

3. **Empirical microstructure cost measurement.** Dùng `data/orderbook.parquet` thực thay vì cost giả định fixed. Đây là phân vân giữa "research" và "production-aware". Strategy thực sự đo cost = chuẩn quant fund.

4. **DSR application với correct n_trials=50.** Earlier version dùng n_trials=22 (fold count, wrong) → strategy đã catch lỗi này. Đây là statistical rigor.

5. **PBO=0.030.** Tốt theo Bailey-López de Prado standard.

### 4.4. Chỗ industry chuẩn cao hơn (cần catch up)

1. **Point-in-time universe.** CRSP-style với delisted stock included. Survivorship bias trong strategy này có thể inflate Sharpe 0.2-0.4 (theo LuxAlgo 2026, citing CRSP US Stock Database 1926-2001: 1.6%/năm gap; Andrikogiannopoulou & Papakonstantinou: 14 pp drawdown underestimate).

2. **Factor orthogonalisation.** PCA project out 5-10 eigenvector trước cointegration test. Avellaneda-Lee (2010, *Quantitative Finance* 10(7), p.761) chỉ ra: "between 10 and 30 factors are required in the U.S. stock universe to explain a mere 50 percent of the variance of individual stock returns" — nghĩa là pair raw có 50% common factor variance, không phải pure idiosyncratic.

3. **Larger N (deep universe).** 528 ticker là OK; nhưng final 50 pair là quá ít. Industry: hundreds-to-thousands simultaneous. Statistical power scale với √N.

4. **Regime detection / vol-target sizing.** Strategy hiện chạy constant $20k/pair across regime. Industry: dynamic sizing theo realized vol, VIX, factor crowding metric. Late Bull failure mode chính là cái regime detection sẽ catch — Z-velocity filter proposed nhưng chưa implement.

5. **Stop-loss + EOS flatten discipline.** Millennium pod culture: hard stops mandatory (5%/7.5% capital loss). Strategy này no SL = nguyên tắc đối lập.

### 4.5. Cách industry handle các integration issue Phần 3 chỉ ra

**Pairs trading decay (Gap 3/4):** Industry solution = deep universe + high turnover + factor orthogonalisation. Khi cointegration của pair X-Y decay, có pair khác trong 1000+ universe replace. Strategy này 50 pair max → khi 10 pair decay, 20% portfolio mất alpha source.

**Factor crowding 2007 (Gap 4):** Lesson learned từ Khandani-Lo (2011): các fund dollar-neutral β-weighted nhưng đều long common factor → unwind cùng lúc. Mitigation industry: monitor crowding metric (Resonanz Capital framework — exposure of multiple funds to same factor), reduce when crowded, factor-neutral hedging. Strategy này không có monitor.

**Cost-aware execution (Gap 1, 2):** Industry dùng smart order routing, time-of-day scheduling (avoid open/close volatility cliff), child order with limit price. Strategy này: decide-at-close, fill-at-close, no time mask — opposite of industry practice.

**Bug discipline (V2.0 audit):** Knight Capital 2012, theo SEC Administrative Proceeding (October 2013): "a technician forgot to copy the new Retail Liquidity Program (RLP) code to one of the eight SMARS computer servers... Orders sent with the repurposed flag to the eighth server triggered the defective Power Peg code still present on that server... resulting in 4 million executions in 154 stocks for more than 397 million shares in approximately 45 minutes. Knight Capital took a pre-tax loss of $440 million." Lesson industry: golden-dataset regression test, code review, canary deployment, kill switch. Strategy này phát hiện 8 bug sau 5 tuần — *trước go-live* — là OK nhưng *quy trình* (làm sao đảm bảo bug thứ 9 không tồn tại) chưa documented.

---

## ĐOẠN KẾT (NGUYÊN VĂN BẮT BUỘC)

**"Nếu tôi là người chịu trách nhiệm về book này với tiền thật, ba điều khiến tôi mất ngủ nhất là..."**

**1. INTEGRATION RISK — Strategy biết kill zones có âm alpha nhưng vẫn trade vào.** Week 5 đã identify 3 window (11:00-11:30, 11:30-12:00, 15:30-15:59) là âm alpha sau cost, với 11:30-12:00 thậm chí gross −26 bps *trước* cost. Strategy hiện không có time-of-day mask. Nếu go live hôm nay, mỗi ngày sẽ trade vào những window này, đốt 20-30 bps mỗi trade ở chỗ tự biết âm. Đây không phải bug — đây là **integration failure**: discovery của Phase 5 không feed back vào execution layer. Mitigation cụ thể: pre-deploy, implement hard time-of-day mask (chỉ trade 09:30-11:00 và 13:00-15:30); A/B test mask vs no-mask trên held-out fold; nếu mask cải thiện net SR thêm ≥0.2, lock in. Trigger để revisit: mask coverage error nếu kill zone alpha pattern thay đổi sau 6 tháng live.

**2. COMPONENT RISK — Late Bull factor momentum failure mode không có fix.** 11/45 fold gần đây nhất (Late Bull 2025-26) cho Sharpe −0.750, 45% positive, 1/11 NC pass. Đây là regime gần thời điểm go-live nhất. Z-velocity filter (entry chỉ khi |dZ/dt| dưới threshold — tức không enter giữa factor momentum đang kéo) đã được propose nhưng chưa implement. Không có safety net (no SL, no EOS) ở regime này = catastrophic. Mitigation cụ thể: implement Z-velocity filter trước deploy, threshold calibrate trên Bear 2022 + Early Bull 2023 (NOT trên Late Bull để tránh in-sample). Plus: thêm hard SL ở 2σ ngoài entry Z (tức Z=5.5 stop nếu entry Z=3.5). Trigger: nếu live realized Sharpe < 5th percentile của backtest distribution trong 2 tháng, halve size; trong 3 tháng, halt new entry.

**3. STATISTICAL RISK — DSR ≈ 0.000 với n_trials=50 và 90 trade/25 fold không đủ statistical power.** Theo Bailey-Borwein-López de Prado-Zhu (2014, *Notices of AMS*, Eq. 3.2): MinBTL < 2·ln(N)/E[max_N]² → với N=50, E[max_N]=1, cần ≥7.8 năm data. Strategy có 4.25 năm. Observed Sharpe 0.503 << expected max under null 2.054 → strategy không reject null hypothesis "Sharpe = 0". Đây không phải tiểu tiết — đây là khả năng performance là **selection bias artifact**, không phải alpha. Mitigation cụ thể: (a) reduce n_trials thực sự tested — nhiều "trial" trong 50 có thể correlated (e.g., Z=3.0, 3.25, 3.5 đều variant gần nhau) → dùng clustering (López de Prado ONC algorithm, AFML Ch.7) để estimate effective N — có thể giảm xuống N=10-15, DSR có thể recover; (b) thu thêm OOS data từ pre-2022 (giữ point-in-time, expand ra 2018-2022 cho thêm 4 năm Bear/Bull cycle); (c) nếu DSR vẫn fail sau cả hai, **kill strategy** — không go live với DSR=0. Trigger để live: DSR p-value ≤ 0.05.

---

*Hết báo cáo. Tổng ~9,200 từ. Phần 3 ~3,000 từ ≈ 32.6% tổng độ dài.*