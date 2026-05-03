# Spurious Correlation Rejection Rules

To prevent false positives, every candidate pair must pass two separate filter layers before final approval.

## Layer 1: Statistical Filter
1. **Stationary Residuals (Core Requirement):** The ADF test must run *on the residual* of the pair's regression (Engle-Granger method), not on the raw price. The residual must be structurally stationary (p-value < threshold).
2. **Plausibly I(1) Legs (Pragmatic Approach):** Cointegration theory assumes non-stationary legs (typically I(1)). While we mathematically require this, *do not over-engineer a massive testing bottleneck for every single asset upfront*. Use level-vs-difference checks or individual ADF tests as a diagnostic *only for the shortlisted assets* that pass the initial residual check.
3. **Hedge Ratio Sanity:** The regression hedge beta must make mathematical sense and not be extreme or inverted relative to the asset class.
4. **Half-Life Rules (Optional but Recommended):** The residual spread must mean-revert frequently enough to be tradable, excluding pairs with exceedingly long or purely Brownian half-lives.

## Layer 2: Economic Logic Filter
To prevent approving statistical anomalies, pairs must be structurally explainable. Prioritize in this exact order:
1. **Tier 1 (Ideal):** Same sector or same specific industry (e.g., two regional banks, two gold miners).
2. **Tier 2 (Acceptable):** Substitute products or explicitly shared macro exposure (e.g., crude oil producer vs. airline, or corn vs. ethanol).
3. **The 1-Sentence Rejection Rule:** If the pair's relationship cannot be explained by a clear, one-sentence economic rationale, **reject it unconditionally**, regardless of how statistically perfect its p-value is.
