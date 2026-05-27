# Graph Community Detection as a Pre-Filter for Cointegration Pair Selection on US Equities: Diagnosis and Recommendations

## Executive Recommendation

**The V5 implementation is failing because it is a hybrid of two distinct research traditions that were never designed to compose — and the most influential paper the user cited (Cartea, Cucuringu & Jin, ICAIF 2023) does not actually do what the user thinks it does.** The Cartea–Cucuringu–Jin paper does not use any pairwise cointegration test (Johansen, Engle-Granger, ADF) anywhere in its pipeline. Their clustering is a *substitute* for pairwise cointegration testing: clusters → trade each stock's deviation from its cluster's 5-day mean return, rebalance every 3 days. There is no within-cluster Johansen step. Likewise, Avellaneda & Lee (2010), the canonical PCA-residualization paper, models each stock's residual as a *univariate* OU process and never tests pairwise cointegration. So both papers usually cited as inspiration for "factor-residual + cointegration" pair selection do not in fact run pairwise cointegration on residuals. The user's V5 stack (PCA-5 residualization → |corr|>0.4 thresholding → Louvain → within-cluster pairwise Johansen → BH-FDR → HL filter → β-sign → |β|≤5) is novel and uncited in the published literature. The zero-pair outcome is therefore not a bug to fix but evidence that the substrate (Pearson correlation of PCA residual returns) and the test (pairwise cointegration of residual log-prices) are largely orthogonal: correlation-of-residual-returns is a within-period covariance statistic, while cointegration requires a shared stochastic trend in the *integrated* component — which PCA has just stripped out. **The single highest-impact fix is to abandon "cluster-then-Johansen-on-PCA-residual-prices" and instead either (a) replicate Avellaneda-Lee verbatim (per-stock OU on PCA residuals, no pairwise cointegration), or (b) replicate Cartea-Cucuringu-Jin verbatim (CAPM residuals, signed correlation, SPONGE, cluster-mean reversion, no Johansen). Mixing the two is the source of the failure.**

---

## TL;DR

- **Fix the substrate by abandoning the pipeline's structure, not by tuning it.** The V5 pipeline is a misreading of Cartea-Cucuringu-Jin (2023), who do not use any pairwise cointegration test — clustering replaces it. There is no published paper combining PCA residualization, correlation-based clustering, and pairwise Johansen on residual log-prices for US equities; the user's V5 stack is unique and uncited.
- **The 0-pair result is consistent with the substrate being statistically uninformative for cointegration**, not with a bug. Pearson correlation of residual returns and Johansen p-values on residual log-prices are nearly independent statistics under the null. After PCA-5 removes the common stochastic trends that generate cointegration, residual log-prices have very low base rates of genuine cointegration; the cluster filter has no power to concentrate that mass.
- **Best evidence-based fix: replicate Avellaneda-Lee (2010) verbatim — per-stock OU on PCA residuals, no pairs, no Johansen — and budget for Sharpe ~0.9 net at best on 2020s data.** Second-best: replicate Cartea-Cucuringu-Jin verbatim (CAPM residuals, SPONGEsym, cluster-mean reversion) — but expect Sharpe ~0.28 at 5 bps round-trip costs based on independent replication. The user's V4 minute-frequency, different-filter result (Sharpe 0.443 net) suggests the high-frequency / mean-reversion path is the more productive direction than daily-frequency / cointegration.

---

## Key Findings

1. **The user has misread the Cartea-Cucuringu-Jin (2023) paper.** Verified verbatim from the ACM paper abstract: "We propose a framework to construct statistical arbitrage portfolios with graph clustering algorithms. First, we use various clustering methods to partition the correlation matrix of market residual returns of stocks into clusters. Next, we construct and evaluate the performance of mean-reverting statistical arbitrage portfolios within each cluster." The downstream trading rule per the paper Section 4 is: "We set w = 5 days for the number of lookback days … to compute the cluster mean returns; the rebalance period … is ℓ = 3 days. The threshold to identify whether a stock is a previous winner or is a previous loser is p = 0." The independent replication by Korniejczuk & Ślepaczuk (2024, arxiv 2406.10695) confirms: "upon performing the clustering, the mean of the last five days' returns have been calculated for each cluster. Then long and short positions were opened on assets whose returns during the last five days were lower or higher, respectively than the calculated clusters' means. … No attempts at optimizing the strategy in regards to optimal allocation, beyond the choice of clustering algorithm have been mentioned in the study." The word "cointegration" does not appear in the paper's methodology.

2. **The residuals in Cartea-Cucuringu-Jin are CAPM β·r_mkt residuals on a 60-day rolling window — NOT PCA-K residuals.** Verified: "the rolling window we use to estimate β and to compute the market residual return is 60 days" (ACM Section 4). The user's PCA-5 residualization is an Avellaneda-Lee style pre-processing, not what Cartea-Cucuringu-Jin do.

3. **The best clustering algorithm in Cartea-Cucuringu-Jin is SPONGEsym on a signed graph (Cucuringu, Davies, Glielmo & Tyagi 2019, arxiv 1904.08575) — NOT Louvain on a thresholded graph.** SPONGEsym uses negative correlations as repulsive edges via two Laplacians (L⁺, L⁻); thresholding |corr|>0.4 discards exactly the sign information SPONGEsym is designed to exploit. The full signed correlation matrix is used as adjacency with no threshold.

4. **Independent replication of Cartea-Cucuringu-Jin confirms catastrophic transaction-cost sensitivity.** Korniejczuk & Ślepaczuk (2024) Table 1: Original Sharpe 1.10 / Own-no-TC Sharpe 1.17 / Own-0.05%-TC Sharpe **0.28** / SPY 0.33. They state: "the amount of transaction costs incurred is four times larger than the net profit from the strategy."

5. **Avellaneda & Lee (2010) — the canonical PCA factor-residual stat-arb paper on US equities — also does not use pairwise cointegration.** Their pipeline is: PCA on 252-day standardized residuals → per-stock OU fit on cumulated residual log-returns → S-score signal (a univariate z-score-like quantity) → trade each stock against its eigen-portfolio. Verbatim from the paper (Quantitative Finance 10(7):761–782): "PCA-based strategies have an average annual Sharpe ratio of 1.44 over the period 1997 to 2007 … during 2003–2007, the average Sharpe ratio of PCA-based strategies was only 0.9." The unit of trade is a stock-vs-factor-portfolio relationship, not a pair.

6. **Onatski & Wang (2018, Econometrica 86(4):1467–1494; arxiv 1605.08880) proved the Wachter-distribution limit for Johansen squared canonical correlations under large-N asymptotics**, with the paper's own abstract stating: "We find that the distribution almost surely weakly converges to the so-called Wachter distribution. This finding provides a theoretical explanation for the observed tendency of Johansen's test to find 'spurious cointegration.'" This is the formal large-dimensional explanation for why pairwise Johansen on 506 residual log-price series will systematically over-reject the null.

7. **Hjalmarsson & Österholm (IMF WP/07/141, 2007) show Johansen is severely oversized under near-integrated regressors.** PCA residual log-prices are exactly near-integrated (the dominant integrated component has been removed), making this issue worse than for raw log-prices.

8. **The user's |corr|>0.4 hard threshold is not calibrated in any published pair-selection pipeline.** All published methods use either no threshold (Cartea-Cucuringu-Jin), MST/PMFG/TMFG (Mantegna 1999; Tumminello et al. 2005), Marchenko-Pastur eigenvalue cutoff (Laloux et al. 2000), top-K edges per node (Onnela et al. 2003), or bootstrap CI (Tumminello et al. 2007). Miao (2014) is the only paper I found using a fixed correlation threshold (|ρ|>0.9) as cointegration pre-filter, and the cutoff 0.4 has no published justification.

9. **Louvain has documented failure modes on financial correlation networks.** Per Fortunato & Barthélemy (2007, PNAS 104(1):36, arxiv physics/0607100), the resolution limit means: "The probability that a module conceals well-defined substructures is the highest if the number of links internal to the module is of the order of √(2L) or smaller" — modularity optimization may fail to identify smaller substructures within communities whose internal link counts fall in this range. Combined with Louvain's non-determinism, this makes Louvain fragile for repeated monthly walk-forward folds.

10. **Brunetti & De Luca (2023, Statistical Methods & Applications 32(5):1611–1640) explicitly study pre-selection for cointegration pairs.** Their finding (verbatim): "pre-selection matters, since the excess returns remarkably vary, in terms of both average and variability, depending on the metrics used. Differences in profitability by pre-selection metrics are retrieved even after considering commissions and cut rules, market impact, a stricter definition of the Spread reversion to the equilibrium and alternative cointegration tests." This is the most directly relevant published evidence that substrate choice dominates — and they do not test PCA-residual-correlation specifically.

11. **The user's V4 baseline itself is suspect.** -0.45 Sharpe over 38 months, 81% end-of-month exits, and the requirement of an artificial |β|≤5 sanity cap (without which 27 pairs contribute 103% of P&L) are all consistent with V4 finding mostly spurious in-sample cointegration that does not persist OOS. The V5 zero-pair output is therefore "Louvain randomized which spurious pairs would survive, and the survivors fell below BH-FDR at q=0.05," not "Louvain broke the pipeline."

---

## Q1. Substrate for the Graph — what do practitioners actually cluster on?

| Substrate | Paper / Repo | Universe & period | Downstream consumer | Reported result |
|---|---|---|---|---|
| **CAPM β·r_mkt residual returns, 60-day window** | Jin, Cucuringu & Cartea (2023), *Correlation Matrix Clustering for Statistical Arbitrage Portfolios*, ICAIF '23, SSRN 4560455, DOI 10.1145/3604237.3626894 | S&P 500, 2000–2022 | SPONGEsym signed-graph clustering → cluster-mean reversion (long laggards, short leaders), rebalance every 3 days. **No cointegration test of any kind.** | Sharpe 1.10 gross, Sortino 2.01, 12.2% ARC. Independently replicated by Korniejczuk & Ślepaczuk (2024, arxiv 2406.10695): Sharpe 1.17 no-TC, **collapses to 0.28 at 5 bps** round-trip |
| **PCA-residualized returns + firm characteristics (94 from Green-Hand-Zhang)** | Han, He & Toh (2023), *Pairs Trading via Unsupervised Learning*, EJOR 307(2):929–947 | CRSP US, 1980–2020, monthly | k-means / DBSCAN / **agglomerative clustering** on standardized firm-characteristic features. Within-cluster spread = Δreturn (NOT cointegration test). | Agglomerative: 24.8% AR, **Sharpe 2.69 gross, 1.73 net of 20 bps**; DBSCAN/k-means worse. Replication by github.com/adamd1985/pairs_trading_unsupervised_learning: k-means best, Sharpe 0.4 — much weaker, attributed to data quality |
| **PCA-projected returns (≤15 components)** | Sarmento & Horta (2020), *Enhancing a Pairs Trading strategy with the application of Machine Learning*, Expert Systems with Applications 158:113490 | S&P 500 commodity ETFs, 2009–2018 | OPTICS clustering on PCA features → within-cluster pairs filtered by cointegration (Engle-Granger), Hurst exponent, half-life | Reported Sharpe 3.79 portfolio (vs 3.58 / 2.59 baselines). Pair yield: typically tens of pairs per fold |
| **Normalized-price SSD (Gatev distance)** | Gatev, Goetzmann & Rouwenhorst (2006), RFS 19(3):797–827 | Full CRSP, 1962–2002 | Top-20 lowest-SSD pairs → trade ±2σ entry. No formal clustering. | Pre-2003: excess returns ~11% AR, ~1.32 Sharpe. Per Do & Faff (2010, Financial Analysts Journal 66(4):83–95), top-20 portfolio mean excess return dropped "from 0.86 percent a month for 1962–1988 to 0.37 percent for 1989–2002 and to just 0.24 percent for 2003–2009"; Do & Faff (2012, Journal of Financial Research) conclude this is "largely unprofitable after 2002" once trading costs are charged |
| **Pearson correlation of raw returns, thresholded** | Miao (2014), *International Journal of Economics and Finance*, 6(3):96–110 | 177 energy stocks NYSE/NASDAQ | |corr|>0.9 pre-filter → ADF on residuals → top-10 lowest ADF p-value | Reduces 15,576 pairs to ~1,378 (avg); reported Sharpe 2.67 monthly. Not survivorship-bias-free |
| **Asset graph / PMFG / MST on Pearson of log-returns** | Mantegna (1999) EPJ B 11:193–197; Tumminello, Aste, Di Matteo & Mantegna (2005) PNAS 102(30):10421–10426; Tumminello, Lillo & Mantegna (2010), *Correlation, hierarchies, and networks in financial markets* | NYSE 100s of stocks, multi-period | Hierarchical filtration — *not directly* a pair-selection substrate but a topology used for clustering | No pair-yield reported — these are descriptive/topological papers; PMFG-DBHT (Song, Di Matteo, Aste 2012) reports cluster purity vs Fama-French sectors |
| **Information-theoretic (mutual information, transfer entropy)** | Fiedor (2014), *Phys Rev E* 89:052801; Guo et al. (2018) | Various | MST-on-NMI for hierarchical clustering | No pair-trading pair-yield published — used for systemic-risk characterization |
| **Copula tail dependence / DCC correlation** | Krauss & Stübinger (2017), IWQW DP; Engle & Kelly (2012) | S&P 100 | Copula-based pair selection or DCC-MST | Copula reports Sharpe ~3.0 gross on S&P 100; not specifically a clustering substrate |

**Crucially, no published implementation uses "Pearson correlation of PCA-residual returns" as the clustering substrate AND then runs pairwise Johansen on PCA-residual log-prices within clusters.** This is the user's unique combination.

---

## Q2. Theoretical interaction between PCA residualization and post-residual correlation clustering

The user's hypothesis — that "PCA-K residualization + cluster-on-residual-correlation + test-cointegration-on-residual-prices" is structurally incompatible — is **largely correct**, for three independent reasons.

**(a) Cointegration requires shared integrated stochastic trends; PCA strips them out.** Cointegration of two I(1) series Xₜ, Yₜ means there exists β such that Xₜ − βYₜ is I(0). By the Engle-Granger representation theorem, cointegrated series share at least one common stochastic trend. In equity factor models (Alexander & Dimitriu 2002; Pole 2007), **common integrated factors are the trends that generate cointegration**. Avellaneda & Lee (2010) state explicitly that PCA factors are common drivers and their residuals are "idiosyncratic, modeled as a mean-reverting Ornstein-Uhlenbeck process" (Quantitative Finance 10(7):761–782). The residuals are designed to be stationary at the *return* level — i.e., the cumulated residual log-price is by construction close to a *driftless random walk plus mean-reverting noise*. Two such residual log-prices have neither a common factor nor a deterministic trend in common. The probability of finding genuine cointegration between residual log-prices is therefore much lower than between raw log-prices.

**(b) Onatski & Wang (2018, Econometrica 86(4):1467–1494; arxiv 1605.08880) prove that the empirical distribution of Johansen squared canonical correlations under large-N asymptotics converges to the Wachter distribution.** From the paper's own abstract: "This finding provides a theoretical explanation for the observed tendency of Johansen's test to find 'spurious cointegration.'" Their result is for raw VARs but the mechanism — overstated rejection rates when residual covariance is poorly conditioned — applies even more strongly to PCA-residualized series where top eigenvalues have been deliberately zeroed.

**(c) Correlation-of-residual-returns vs cointegration-of-residual-prices measures different things.** Correlation is a within-period covariance statistic on first differences; cointegration is about long-horizon co-movement of integrated levels. Two residual-return series can have |corr| = 0.5 without their cumulated residual log-prices being cointegrated, because the residual returns are by design close to stationary white noise and the random-walk piece of the cumulated residual is by construction unrelated across stocks (no shared integrated factor). Conversely, two stocks with weak residual-return correlation can still be cointegrated if some slow sub-factor PCA missed has integrated dynamics. **The Pearson correlation of residual returns and the Johansen statistic on residual log-prices are nearly statistically independent objects.** The clustering filter therefore does not concentrate probability mass toward "cointegrated" pairs — it concentrates toward "near-orthogonal-residual-correlation" pairs, which is the *opposite* of what is needed.

**(d) Hjalmarsson & Österholm (2007), IMF Working Paper WP/07/141, "Testing for Cointegration Using the Johansen Methodology when Variables are Near-Integrated"** show that under near-unit-root behavior the Johansen test is severely oversized. PCA residual log-prices are exactly this case, which means: (i) the size of the Johansen test is suspect on residual log-prices, and (ii) any "successful" cointegration finding on PCA residuals in the V4 run is more likely a false rejection of the null than the user assumed — consistent with the V4 negative Sharpe.

**Verdict: PARTIALLY CORRECT.** The cluster filter (substrate = residual return correlation) does not improve the cointegration test's signal-to-noise ratio because the two statistics are nearly independent. But the more damning observation is that **PCA-K residualization followed by pairwise cointegration on residual log-prices is fragile to begin with** — V4 Sharpe -0.45, 81% end-of-month exits, and the artificial |β|≤5 sanity cap (without which 27 pairs contribute 103% of P&L) are all consistent with V4 finding mostly spurious in-sample cointegration.

---

## Q3. Clustering algorithm choice

| Algorithm | Used by | Notes |
|---|---|---|
| **Louvain (user's choice)** | Korniejczuk & Ślepaczuk (2024) for graph features only — not for SPONGE clustering | Resolution limit: per Fortunato & Barthélemy (2007, PNAS 104(1):36, arxiv physics/0607100), "The probability that a module conceals well-defined substructures is the highest if the number of links internal to the module is of the order of √(2L) or smaller." Non-deterministic across seeds. **Not used by any pair-selection paper I found that targets ~500-asset universes for cointegration pair filtering.** |
| **SPONGEsym (signed-graph spectral)** | Jin, Cucuringu & Cartea (2023) — best performer in their study | Cucuringu, Davies, Glielmo & Tyagi (2019), AISTATS, arxiv 1904.08575. Uses negative correlations as *repulsive* edges via two Laplacians (L⁺, L⁻). **Requires the full signed correlation matrix as adjacency — incompatible with the user's |corr|>0.4 hard threshold.** |
| **Agglomerative (single/complete/average/Ward linkage on correlation distance d_ij = √(2(1-ρ_ij)))** | Han, He & Toh (2023); López de Prado HRP (2016); Mantegna (1999) | Best in Han et al. The Mantegna distance d_ij = √(2(1-ρ_ij)) is the standard for hierarchical clustering on financial returns. López de Prado uses single-linkage in HRP. |
| **OPTICS / DBSCAN** | Sarmento & Horta (2020); arbitragelab (Hudson & Thames) | Density-based, allows "noise" cluster. Sarmento reports Sharpe 3.79 on commodity ETFs. Production-grade Python at hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/ml_approach/ml_based_pairs_selection.html |
| **k-means / k-medoids** | Han et al. (2023); ubiquitous in tutorials | Often worst-performing in head-to-head comparisons. Requires k tuning. |
| **Spectral clustering (Ng-Jordan-Weiss)** | Jin/Cucuringu/Cartea — also tested | Modified for signed graphs in their paper. |
| **PMFG / TMFG / DBHT** | Tumminello, Aste, Di Matteo & Mantegna (2005, PNAS 102(30):10421); Song, Di Matteo & Aste (2012) | Topological filtration retaining MST as subgraph, then community detection on the filtered graph. |
| **GNN-based** | Many GNN-stock-prediction papers; Cartea, Jaimungal, Sánchez-Betancourt (2024) on RL stat arb | Higher implementation cost; mixed evidence of robustness. |

**Louvain failure modes on financial correlation matrices:**
1. **Resolution limit** (Fortunato-Barthélemy 2007): with the user's L ≈ thousands of edges after thresholding, modules with √(2L)-comparable internal link counts may be merged or hidden. The user obtained 16 clusters of size ≥5, several likely vulnerable.
2. **Non-determinism**: even with seed=42, small perturbations in input (e.g., one extra trading day) can swap stocks between clusters across folds.
3. **Unsigned graph requirement**: by taking |corr| the user discards anti-correlation information that SPONGE-family algorithms exploit.
4. **Threshold sensitivity**: |corr|>0.4 is arbitrary; the network literature (Onnela et al. 2003, Phys Rev E 68:056110; arxiv 1002.3432) finds fixed thresholds produce disconnected subgraphs or single giant components depending on value chosen.

---

## Q4. Threshold / hyperparameter calibration

Published methods to choose the edge-weight cutoff:

1. **No threshold** — full correlation matrix as adjacency (Jin/Cucuringu/Cartea 2023; SPONGE methods).
2. **MST / PMFG / TMFG topological filtration** — Mantegna (1999), Tumminello et al. (2005). N-1 edges (MST) or 3(N-2) edges (PMFG). Data-adaptive, parameter-free.
3. **Random Matrix Theory eigenvalue cutoff** — Laloux, Cizeau, Bouchaud & Potters (2000), IJTAF 3(3):391–397 and Plerou et al. (2002), Phys Rev E 65:066126 — separate "signal" from Marchenko-Pastur bulk. Jin/Cucuringu/Cartea use this to set the number of clusters.
4. **Top-K edges per node** — Onnela et al. (2003).
5. **Bootstrap CI on correlation** — Tumminello, Coronnello, Lillo, Miccichè & Mantegna (2007), *International Journal of Bifurcation and Chaos* 17:2319–2329.
6. **Fixed Pearson threshold** — Miao (2014) uses |ρ|>0.9. Various other papers use thresholds; the literature explicitly warns these methods are sensitive to threshold values and "inappropriate threshold values will produce isolated subgraphs and information loss."

The user's |corr|>0.4 falls in category 6. There is no paper calibrating |corr|>0.4 specifically for pre-cointegration filtering on US equities; 0.4 appears to be a hand-picked tutorial value absent from peer-reviewed pair-selection literature.

---

## Q5. Downstream cointegration test choice

What practitioners use after a within-cluster pair list:

- **Johansen** — Used by Sarmento & Horta (2020, indirectly via arbitragelab); Brunetti & De Luca (2023, *Statistical Methods & Applications* 32(5):1611–1640) compare seven pre-selection metrics feeding into a Johansen-based pairs trade. Their finding: "the excess returns remarkably vary, in terms of both average and variability, depending on the metrics used. Differences in profitability by pre-selection metrics are retrieved even after considering commissions and cut rules, market impact, a stricter definition of the Spread reversion to the equilibrium and alternative cointegration tests." They do not test PCA-residual correlation specifically.
- **Engle-Granger two-step** — Most common in pairs-trading literature; Vidyamurthy (2004); Caldeira & Moura (2013); Clegg & Krauss (2018) extend to partial cointegration.
- **Phillips-Ouliaris** (1990), Econometrica 58:165–193. Less common in stat arb papers.
- **ADF on residuals** — Miao (2014); standard in Engle-Granger second step.
- **Clegg & Krauss (2018) partial cointegration** — Quantitative Finance 18(1):121–138 / FAU DP 05/2016. Allows the spread to contain both a random-walk and a mean-reverting component. From the paper: "We find annualized returns of more than 12% after transaction costs. These results … are well superior to classical distance-based or cointegration-based pairs trading variants on our data-set." Survivor-bias-free, 1990–2015.
- **Bayesian cointegration** — arxiv 1311.0524, arxiv 2312.17061. Not yet standard.

**Does BH-FDR interact correctly with the cluster pre-filter? Two issues:**

1. **BH-FDR validity after a selection step.** BH controls FDR under independence or PRDS (Benjamini & Yekutieli 2001). If the pre-selection step uses information *from the same data* as the cointegration test, surviving p-values are no longer uniform under the null. This is "post-selection inference" and is a known invalidity (arxiv 1306.1059, Berk et al. 2013; arxiv 2401.16651). In the user's pipeline, the cluster filter uses residual *returns* and the Johansen test uses residual *log-prices* — first-differenced versions of each other. Post-selection bias is severe: BH-FDR at q=0.05 may be either anti-conservative or (more commonly) drastically *under-powered*. **BH-FDR was the wrong correction once a data-driven pre-filter was applied.**

2. **m drops from 127,765 to 4,333 — should BH threshold be adjusted?** Mechanically, no: BH at q=0.05 over m tests rejects p-values below k/m·q for the largest k satisfying the inequality. But BH power depends on the *fraction of nulls that are false*. If the cluster pre-filter does not concentrate cointegrated pairs (§Q2), the *fraction of false nulls in the within-cluster subset can be lower than in the all-pairs subset* — the filter actively hurts power. V4: 127,765 pairs, ~189 survive → ~0.15% rejection rate. V5: 4,333 pairs, 0 survive → 0% rejection rate. A useful pre-filter would yield a *higher* rejection rate; getting zero strongly suggests the filter is anti-informative for cointegration.

---

## Q6. End-to-end pair yield expectations

- **User V4**: 189 pairs/fold, monthly, S&P 500, 12-month formation. Monthly Sharpe annualized = -0.45.
- **User V5**: 0 pairs/fold.
- **Clegg & Krauss (2018), partial cointegration on S&P 500**: typically ~20–50 pairs per 6-month formation, 12% AR after transaction costs 1990–2015.
- **Sarmento & Horta (2020)**: clusters yield handfuls of pairs (5–30 per fold on commodity ETFs); after cointegration + Hurst + half-life, ~5–20 trades per period.
- **Gatev et al. (2006)**: top-20 lowest-SSD pairs per 12-month formation by construction.
- **Han, He & Toh (2023)**: no pair-level filter; every stock either in a cluster or noise — portfolio strategy.

**Has 0/fold ever been reported as "the substrate failed"?** Not directly. Most papers tune until they get pairs. Closest analogue: Brunetti & De Luca (2023) find that some pre-selection metrics drastically reduce pair yield vs others on the same dataset, mapping to lower portfolio profitability — they switch substrate, not parameters. **Implication for the user: 0-pair output is a signal that the substrate is wrong, not that a parameter needs tweaking.**

---

## Q7. Concrete working implementations

| Repo / source | What it does | Status |
|---|---|---|
| **Hudson & Thames `arbitragelab`** — `arbitragelab/ml_approach/optics_dbscan_pairs_clustering` | Sarmento-Horta pipeline: PCA reduction (≤15 components) → OPTICS/DBSCAN clustering → within-cluster cointegration (Engle-Granger), Hurst, half-life | Production-grade Python, paid license |
| **Hudson & Thames `arbitragelab`** — `arbitragelab/other_approaches/pca_approach` | Direct Avellaneda-Lee: PCA on standardized returns → eigenportfolio loadings → OU on residuals → S-score signals | Production-grade, paid license |
| **Hudson & Thames `arbitragelab`** — HRP / clustering modules | López de Prado HRP, hierarchical clustering on correlation distance | Production-grade |
| **github.com/adamd1985/pairs_trading_unsupervised_learning** | Replication of Han, He & Toh (2023) with k-means / DBSCAN / agglomerative on firm-characteristic + momentum features | Honest replication, reports Sharpe ~0.4 (vs paper's 2.69) — flags data-quality issues |
| **github.com/AlexChristensen/PMFG** (R) | Planar Maximally Filtered Graph (Tumminello 2005) | Reference implementation, not pair-trading |
| **github.com/ngozzi/multiplex** | Multiplex (MST + PMFG + correlation) on S&P 500 | Descriptive, not stat arb |
| **github.com/fja05680/sp500** | Historical S&P 500 constituent membership — to fix survivorship bias | Reference dataset |
| **Korniejczuk & Ślepaczuk (arxiv 2406.10695)** | Python replication of Cartea-Cucuringu-Jin SPONGEsym + ML signal filtering | Code not publicly released; methodology fully described |
| **Cartea-Cucuringu-Jin (2023) source code** | Not publicly released; methodology documented in ACM paper | Referenced through replications |

**No reproducible public repository implements "PCA residual + correlation clustering + within-cluster pairwise Johansen + BH-FDR" — the user's exact stack.** This combination is novel.

---

## Q8. Hyperscale practitioners

Very little public material. Funds (Renaissance, D.E. Shaw, Two Sigma, AQR, Citadel) suppress pair-selection internals.

- **AQR**: Asness, Moskowitz & Pedersen (2013, *Journal of Finance*) covers value and momentum across asset classes, not stat arb pair selection. Public emphasis on factor models (Frazzini, Israelov, Moskowitz) rather than pairs trading.
- **Renaissance / Medallion**: nothing public on methodology. Patterson (2010, *The Quants*); Zuckerman (2019, *The Man Who Solved the Market*) — narrative only.
- **D.E. Shaw / Two Sigma / Citadel**: nothing public.
- **Ernest Chan**, *Algorithmic Trading* (2013) and *Quantitative Trading* (2009): describes pairs trading using ADF on the spread for two pre-chosen stocks; doesn't recommend clustering pre-filters; emphasizes "cointegration is more important than correlation" and warns against using correlation as a pair selector.
- **Andrew Pole**, *Statistical Arbitrage* (2007, Wiley): treats stat arb as factor-residual mean reversion, close to Avellaneda-Lee. Does not advocate clustering.
- **Harris** (2003), *Trading and Exchanges*: doesn't engage with this question.

**Bottom line: no public material from hyperscale funds endorses graph-clustering as a pre-filter for cointegration testing on US equities.**

---

## Top-3 Alternative Implementations, Ranked by Evidence Strength

### #1 — Replicate Avellaneda-Lee (2010) exactly. *No clustering, no pairwise cointegration.*
**Paper**: Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities Market*, Quantitative Finance 10(7):761–782.
**Reference**: `hudson-and-thames/arbitragelab` PCA approach module.
**Methodology**: PCA (typically 15 components, not 5) on standardized residuals from a 60-day rolling window → per-stock residual log-price as cumulative residual log-return → fit OU to each per-stock residual → trade S-score crossing ±1.25 entry / ±0.5 exit. **No pair selection; each stock is its own univariate trade against its eigen-portfolio.**
**Reported**: Per the published paper verbatim, "PCA-based strategies have an average annual Sharpe ratio of 1.44 over the period 1997 to 2007 … during 2003–2007, the average Sharpe ratio of PCA-based strategies was only 0.9." Multiple independent reproductions exist in academic working papers and student projects.
**Why it addresses 0-pair**: Eliminates pair selection entirely. Unit of trade is "stock vs. its synthetic factor portfolio."
**Effort**: Low — 1–2 weeks. User already has the PCA residuals.
**Why #1**: The only baseline in this area with an unambiguous, multiply-replicated, decade-spanning track record on US equities.

### #2 — Replicate Cartea-Cucuringu-Jin (2023) exactly. *No pairwise cointegration.*
**Paper**: Jin, Cucuringu & Cartea (2023), ICAIF '23, SSRN 4560455, DOI 10.1145/3604237.3626894.
**Reference**: Methodology in Korniejczuk & Ślepaczuk (2024, arxiv 2406.10695); SPONGE at Cucuringu's GitHub / `alan-turing-institute/SigNet`.
**Methodology**: CAPM β residuals on 60-day rolling window → full signed Pearson correlation matrix as adjacency (NO thresholding) → SPONGEsym clustering with k≈30 or Marchenko-Pastur → for each cluster, compute 5-day cluster-mean return → long stocks below cluster mean / short stocks above → rebalance every 3 days. **No within-cluster cointegration test.**
**Reported**: Sharpe 1.10 gross; Sharpe **0.28 net of 5 bps** (Korniejczuk & Ślepaczuk replication; quoted: "the amount of transaction costs incurred is four times larger than the net profit from the strategy").
**Why it addresses 0-pair**: Eliminates pairwise cointegration.
**Effort**: Moderate — 2–3 weeks. Implement SPONGEsym, replace PCA-5 with CAPM β residuals, replace pair selection with cluster-mean signal.
**Caveat**: Very poor transaction-cost robustness.

### #3 — Clegg & Krauss (2018) partial cointegration. *Keeps Johansen-style pairwise testing but with a stronger model.*
**Paper**: Clegg & Krauss (2018), *Pairs trading with partial cointegration*, Quantitative Finance 18(1):121–138 / FAU DP 05/2016.
**Methodology**: Pairs from S&P 500 constituents → partial cointegration model (spread = random walk + mean-reverting AR(1)) fit via state-space MLE → likelihood-ratio test for the mean-reverting component → trade pairs whose mean-reverting variance exceeds threshold.
**Reported**: Per paper verbatim, "We find annualized returns of more than 12% after transaction costs … well superior to classical distance-based or cointegration-based pairs trading variants." Survivor-bias-free, 1990–2015.
**Why it addresses 0-pair**: Keeps user's invested pair-selection workflow but replaces fragile Johansen-on-PCA-residuals with a more flexible model that explicitly allows residual non-stationarity. Does not require clustering at all.
**Effort**: Moderate-high — 3–4 weeks. State-space implementation, MLE convergence diagnostics.
**Caveat**: Clegg & Krauss test on raw log-prices, not PCA-residual log-prices. Applying partial cointegration to PCA residuals is unpublished territory.

---

## Known Failure Modes to Watch For

1. **Replacing Louvain with SPONGEsym while keeping pairwise Johansen downstream** — half a fix, probably yields the same 0-pair result. SPONGE feeds cluster-mean reversion, not cointegration.
2. **Switching substrate (e.g., to firm characteristics à la Han et al.) but keeping PCA-residual log-prices for Johansen** — same dichotomy: clustering on one space, testing on another.
3. **Increasing K in PCA (e.g., 15 components like Avellaneda) without re-evaluating the cointegration test on residuals** — more components remove more common trends, making residual cointegration *even rarer*.
4. **Adjusting BH-FDR q from 0.05 to 0.20** — addresses symptom not cause; will admit more noise pairs.
5. **Lowering |corr|>0.4 threshold to 0.2** — Louvain will produce one giant component; clustering collapses.
6. **Raising threshold to |corr|>0.7** — almost no edges survive after PCA-5 (residual returns have small correlations by construction); only a handful of pairs per fold.
7. **Adding sector-based clustering as a fallback** — Korniejczuk & Ślepaczuk note this performs about as well as SPONGEsym in the Cartea-Cucuringu-Jin study, but the user's strategy isn't cluster-mean reversion so this won't directly help.
8. **Survivorship bias** — user said "S&P 500 members as of late 2025." This introduces forward-looking bias in folds going back to 2023. Standard reference for historical S&P 500 membership: github.com/fja05680/sp500.

---

## What the User Should NOT Do

1. **Do not "tune" the V5 pipeline.** The structure is wrong, not the parameters. Sweeping |corr| threshold, Louvain seed, BH q-level, or cluster size minimum will give noisy fold-by-fold output but no consistent edge.
2. **Do not cite Cartea-Cucuringu-Jin as inspiration for a Johansen-based pipeline.** They do not run cointegration tests. Any internal documentation that says "V5 is based on Cartea et al. 2024" is incorrect.
3. **Do not apply BH-FDR after a data-driven cluster filter and expect the q-level guarantee to hold.** Post-selection inference requires either holdout-set validation or selective-inference corrections (Lee, Sun, Sun & Taylor 2016). Practical fix for pair trading: use the formation window for both clustering and Johansen p-values, but evaluate via held-out trading-window P&L, not via in-sample p-values.
4. **Do not assume that "factor model + cointegration" is a published, validated combination on US equities.** It isn't. Avellaneda-Lee model factor residuals as OU at the univariate level. Pole (2007) emphasizes the same. The combination of PCA residuals AND pairwise cointegration is essentially unpublished.
5. **Do not use survivorship-biased S&P 500 universe.** Switch to historical constituents (github.com/fja05680/sp500).
6. **Do not interpret the V4 baseline as evidence that pairwise Johansen-on-PCA-residuals works.** It doesn't: -0.45 Sharpe over 38 months, 81% end-of-month exits, and 27 pairs (|β|∈[50,2708]) contributing 103% of P&L pre-sanitization. The 189 "surviving" pairs are themselves the spurious-rejection problem (Onatski-Wang Wachter distribution), not a validation of the substrate.

---

## Specific Assessment of the "Structural Incompatibility" Hypothesis

The hypothesis as stated by the user:
> "PCA + cluster-on-residual-correlation is structurally incompatible. Pairs surviving the cluster filter after PCA are those correlated through idiosyncratic sub-factors PCA missed, and these are exactly the pairs whose cointegration is most likely spurious noise-driven in-sample with no OOS persistence."

**Verdict: Partially correct, and the deeper structural issue is broader than the user states.**

What is correct:
- PCA residuals are designed to be near-orthogonal to top-K common factors and approximately stationary at the return level (Avellaneda-Lee §2).
- Two residual-return series with |corr|>0.4 must share some sub-factor structure PCA missed.
- Sub-factor structure missed by PCA-5 is either (a) genuinely weak and noise-like, in which case any in-sample cointegration is spurious, or (b) a genuine but low-variance factor that PCA-5 underweighted, in which case cointegration may exist but the test is severely underpowered. In either case, OOS persistence is fragile.

What is **also true and the user underweights**:
- Residual-return Pearson correlation and residual-log-price Johansen p-value are **largely uncorrelated** under H₀ of i.i.d. residual returns. So clustering on the first does not concentrate signal in the second — it randomizes which spurious null rejections fall into clusters.
- Combined with BH-FDR on the within-cluster subset (with severely reduced m), the result is that genuine signal at the FDR boundary in the all-pairs setting gets pushed below the boundary in the cluster setting. The 0-pair outcome is consistent with what one would expect from a correctly-implemented pipeline applied to a problem where the cluster filter is anti-informative for cointegration.

What is **incorrect** in the user's framing:
- The 0-pair output is not "the cluster correctly identifying that BH-FDR shouldn't reject." It's "the cluster filter being approximately statistically independent of cointegration, while reducing the test universe in a way that removes the few genuinely significant pairs that were in the V4 result by chance." V4's 189 pairs included some genuine cointegration plus more spurious; V5 removed both at roughly the same rate.

**The structurally correct fix is to drop one of the two operations**: either drop the PCA residualization (cluster and test on raw log-prices, as Cartea/Cucuringu/Jin do up to using CAPM single-factor) **or** drop the pairwise cointegration test (use per-stock OU à la Avellaneda-Lee, or cluster-mean reversion à la Cartea-Cucuringu-Jin). Mixing the two is what creates the structural incompatibility, and there is no published paper that mixes them successfully on US equities.

---

## Caveats

1. **The Cartea-Cucuringu-Jin (2023) ACM/SSRN PDF could not be retrieved directly** in this research (403 / paywall). All methodology details are sourced from (a) verbatim quotes from the ACM full-text HTML returned via search engines, (b) the independent replication Korniejczuk & Ślepaczuk (2024, arxiv 2406.10695), which I read in full, and (c) consistent secondary descriptions on Medium (Ivan Blanco) and Rebellion Research. All four sources agree on every methodological point and none mention any pairwise cointegration testing. There is a small residual risk that a subtle pair-level filter exists in the paper and is uniformly missed by all secondary sources — but the trading rule ("long laggards, short leaders, rebalance every 3 days") leaves no room for one.

2. **The Sarmento & Horta (2020) Sharpe of 3.79 should be treated as suspect.** It is reported on commodity-ETF universes (small, dominated by sector ETFs), and the methodology has not been replicated independently in published work to my knowledge. The Hudson & Thames `arbitragelab` implementation is a faithful code translation of the paper, not an independent validation. Han, He & Toh (2023) Sharpe of 2.69 is similarly large; the independent replication at github.com/adamd1985/pairs_trading_unsupervised_learning gets only ~0.4 with publicly-available data.

3. **Survey of hyperscale funds is necessarily speculative.** Funds publish nothing about pair selection. Absence of evidence is not evidence of absence: it is plausible that Renaissance or DE Shaw use graph-clustering pre-filters internally; I have no way to know.

4. **The user's V4 minute-frequency Sharpe 0.443 net result is encouraging** but uses a different filter chain, so it does not directly validate or invalidate any specific component of the V5 daily pipeline. Worth investigating: whether the minute-frequency strategy can scale up to S&P 500 with the same logic, since the daily-frequency strategy appears structurally compromised.

5. **The Han, He & Toh (2023) result deserves replication on a clean US dataset before any structural change to V5.** It is the strongest published result for ML clustering as a stat-arb pair-selection device, but the github replication that uses non-CRSP data drops Sharpe from 2.69 to ~0.4, suggesting the result may be data-quality-dependent or sensitive to exact characteristic construction.

6. **Post-selection inference adjustments (selective-inference, conditional p-values) are mathematically available** (Lee, Sun, Sun & Taylor 2016, Annals of Statistics 44(3):907–927) but have not been worked out for pair-selection pipelines specifically. A correct implementation would require deriving conditional Johansen distributions under the cluster-membership event, which is non-trivial. The practical alternative — train/test split where clustering and Johansen happen on training, performance on test — is cheaper.

7. **All published Sharpe numbers in this domain should be treated as upper bounds.** Multiple replications (Korniejczuk & Ślepaczuk for Cartea-Cucuringu-Jin; adamd1985 for Han-He-Toh; the literature broadly for Gatev) consistently report 30–80% lower Sharpe than original papers, often due to transaction costs, survivorship bias, or data quality. The user should budget for similar gaps in any new replication.