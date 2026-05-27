# Deep-research prompt — graph clustering pre-filter for cointegration pair selection in factor-residual stat-arb

**Purpose:** I have an implementation that should work in theory but produces 0 surviving pairs on real data. Before pivoting away, I need to know what *practitioners* (academic and industry) actually do, with sources, so I can decide whether (a) my substrate is wrong, (b) my algorithm choice is wrong, (c) the pre-filter idea itself is wrong, or (d) my downstream filter chain is incompatible with clustering output. I want concrete recommendations grounded in published code/papers, not speculation.

---

## Hand the entire section below to the researcher

You are researching the engineering of **pair-selection pre-filters using graph community detection** for a factor-residual cointegration pairs-trading strategy on US equities. Your goal is to recommend a robust, evidence-based implementation, citing specific papers, code repositories, and empirical results.

### My current setup (background — do not assume this is right)

**Universe:** S&P 500 members (as of late 2025), ~506 tickers passing liquidity/price/completeness screens, daily close-to-close log returns.

**Walk-forward:** 12 months formation + 1 month trading, 39 monthly folds covering 2023-01 through 2026-03.

**Factor model:** Sklearn PCA, 5 components, fit on standardized daily log-returns over the 12-month formation window. Cumulative variance explained ~0.40–0.62 across folds.

**Residual:** Each ticker's "residual log-price" is computed by integrating its residual log-returns after PCA projection: `residual_returns = returns - W @ W.T @ returns`; `residual_log_prices = cumsum(residual_returns) + initial_log_price`. These residuals are by construction near-orthogonal to the top 5 PCs.

**Cointegration test:** Pairwise Johansen on each pair of `residual_log_prices`, p-value via chi-squared LR statistic. Hedge ratio (`beta_pca`) and intercept (`alpha_pca`) from the cointegrating vector.

**Pair filtering pipeline (in order):**
1. Hard screens: min liquidity, min price, ≥80% bar completeness — applied before PCA. ~506 survive.
2. Pairwise Johansen on residual log-prices → raw p-values for C(506,2) = 127,765 pairs.
3. Benjamini–Hochberg FDR control at q=0.05.
4. Ornstein-Uhlenbeck half-life filter: keep pairs with `HL ∈ [5, 30]` trading days.
5. β > 0 filter (drop pairs that move same direction).
6. |β| ≤ 5 sanity cap (PCA artifact protection — empirically calibrated, 27 pairs with |β| ∈ [50, 2708] contributed 103% of P&L in unsanitized run).

**Performance baseline (V4, no clustering, all 127k pairs tested):** Monthly Sharpe annualized = -0.45 over 38 traded months, sum return -6.58%. The strategy as-is has no alpha after honest cost modeling (Week 5 spread + impact + commission). 81% of trades exit at end-of-month (open-at-EOM), suggesting many "cointegrated" pairs in formation break in trading — i.e., spurious cointegration.

### The proposed pre-filter (V5, currently failing)

Inspired by arxiv:2406.10695 (Cartea et al., 2024 — reports Sharpe 1.23 vs 0.48 baseline on US equities). The idea: restrict the Johansen test universe from "all pairs" to "pairs in the same economic cluster."

**Implementation:**
1. Compute pairwise Pearson correlation matrix on residual *returns* (first-differenced residual log-prices) over the formation window.
2. Build an undirected graph: nodes = tickers, edges where `|corr| > 0.4`, weight = `|corr|`.
3. Run Louvain community detection on the weighted graph (`networkx.algorithms.community.louvain_communities`, seed=42).
4. Drop clusters with fewer than 5 members.
5. Enumerate within-cluster pairs only. Run Johansen on those (~3–8k tests).
6. Continue with the existing pipeline (BH-FDR → HL → β>0 → |β|≤5).

### Symptom

On fold 1 (formation 2022-01 to 2022-12, trade 2023-01):
- 506 survivors after hard screens
- Louvain finds 153 communities total, 16 with size ≥ 5
- **4,333 within-cluster pairs** (96.6% reduction from 127,765 all-pairs — as expected/predicted)
- Pairwise Johansen runs on all 4,333 pairs
- **BH-FDR at q=0.05 rejects 0 of 4,333 nulls**
- 0 pairs survive → 0 trades → fold abandoned

V4 baseline on the *same fold*, with the same factor model and same hard-screens, finds **189 pairs** after the same downstream filters. So Johansen IS finding cointegration in this dataset — just not in the within-cluster subset.

### What I need from you

Answer the following with specific sources (papers, GitHub repos, working code, empirical results), and explicitly flag where there is disagreement in the literature or where evidence is thin.

**Q1. Substrate for the graph.** What do practitioners actually cluster on for cointegration pair selection on US equities?
- Raw log-returns?
- Post-factor-model residual log-returns (like mine)?
- Distance metrics derived from prices (e.g., normalized-price distance, sum-of-squared-deviations)?
- Information-theoretic measures (mutual information, transfer entropy)?
- Copula-based dependence measures?
- Embeddings (e.g., from a learned representation)?

For each substrate found in the literature, report (1) which paper or repo uses it, (2) what dataset it was validated on, (3) the reported pair-yield (how many pairs typically survive end-to-end), (4) whether the substrate is *consistent with* a subsequent factor-residual cointegration test or *replaces* it.

**Q2. Theoretical interaction between PCA residualization and post-residual correlation clustering.** If PCA projects out the top-K common factors, what is the expected within-cluster cointegration yield when clustering on the residuals vs the raw returns? Is there a known result (paper or empirical observation) that residual-return-correlated pairs are *less* likely to cointegrate than randomly-paired pairs? If so, what is the mechanism?

**Q3. Clustering algorithm choice.** For ~500-ticker US-equity pair-selection, what algorithm do practitioners use and why?
- Louvain (what I used)
- Hierarchical agglomerative clustering (single/complete/average linkage)
- k-means / k-medoids
- DBSCAN / HDBSCAN
- Spectral clustering
- Affinity propagation
- Newer GNN-based methods

Specifically: are there known failure modes of Louvain on financial correlation matrices (e.g., resolution limits, instability across formation windows)?

**Q4. Threshold / hyperparameter calibration.** How is the edge-weight threshold (mine: `|corr| > 0.4`) calibrated in published implementations? Is it a fixed value, percentile-based (e.g., top-K edges per node), data-adaptive (e.g., MST + planar graph filtration), or statistically validated (e.g., bootstrap CI on correlation)?

**Q5. Downstream cointegration test choice.** Given a within-cluster pair list, do practitioners use:
- Johansen (like me)
- Engle-Granger two-step
- Phillips-Ouliaris
- KPSS on the residual
- VECM with explicit common-trend constraint
- Bayesian alternatives

Is there evidence that any of these *interacts* better with clustering output than others? Specifically: does the BH-FDR threshold need adjustment when m drops from ~127k to ~5k?

**Q6. End-to-end pair yield expectations.** For US-equity factor-residual stat-arb with monthly walk-forward folds:
- What is the typical pair count surviving end-to-end in published implementations? (Mine V4: ~190/fold, V5: 0/fold.)
- Is 0/fold ever reported as "the substrate failed"? What was the proposed fix?

**Q7. Concrete working implementations.** Point me to specific GitHub repositories (preferably with reproducible results) that implement graph-clustering pair selection on liquid US equities at daily frequency. Mention any quantopian/quantconnect/etc. notebooks if relevant. Distinguish "demo on synthetic data" from "validated on real data with positive OOS performance."

**Q8. Hyperscale practitioners.** Do funds (Renaissance, D.E. Shaw, Two Sigma) publish anything about how they do pair selection? If clustering is mentioned in any public-facing material (interviews, papers, conference talks), summarize their stated approach.

### Output format

Give me a structured report:
1. **Executive recommendation** (one paragraph): for my exact setup, what is the single most likely fix grounded in the literature?
2. **Q1–Q8 answers** with citations.
3. **Top-3 alternative implementations** to consider, ranked by evidence strength, each with: paper/repo link, what they do differently, why it might address my 0-pair symptom, what effort is needed to swap in.
4. **Known failure modes** to watch for if I do swap in an alternative.
5. **What I should NOT do** (common mistakes the literature warns against).

If the literature is thin on any question, **say so explicitly** rather than guessing. I would rather have "no published evidence found" than confident speculation.

---

## Notes for me (the user, not the researcher)

- This prompt is self-contained — the researcher needs no other context from the codebase.
- The factual claims about my setup (V4 -0.45 Sharpe, V5 0 pairs at fold 1, etc.) are auditable in `documents/log.md` and `results/v4/final_dynamic_cost/fold_metrics.csv`.
- After the research returns: update `documents/log.md` with the literature-grounded path forward as a **new pre-commitment** before any further coding. Do not selectively pick recommendations that match prior assumptions.
- The pre-committed V5 decision rule (Sharpe ≥ +0.5 ship competitive / ≥ +0.2 ship modest / ≥ 0 doc only / < 0 pivot to V6) still applies to whatever the substrate-change run produces. The substrate change itself is a *design decision* requiring a new pre-commitment; the *performance threshold* for shipping is unchanged.
