**1\. What is a spread in pair trading?**A spread is a synthetic financial instrument created by mathematically combining two or more underlying assets that share a common stochastic trend 1\. Because the assets are historically cointegrated, their combined price action cancels out broader market movements, resulting in a stationary, mean-reverting series that allows you to trade relative mispricings regardless of market direction 1, 2\.  
**2\. What are the main ways to define spread:**

* **Raw price difference ($S\_t \= P\_{A,t} \- P\_{B,t}$):** The simplest method, which assumes a fixed 1:1 hedge ratio 3\. However, it lacks robustness if the assets trade at drastically different nominal price levels (e.g., $500 vs $50), as it fails to maintain true market neutrality 4\.  
* **Price ratio ($Ratio\_t \= P\_{A,t} / P\_{B,t}$):** This divides one price by the other, providing a scale-invariant metric that protects against parallel price surges 4, 5\. However, ratios of normally distributed prices are not themselves perfectly normal, which complicates the accurate interpretation of Z-score standard deviations 5\.  
* **Log price ratio ($Spread\_t \= \\ln(P\_{A,t}) \- \\ln(P\_{B,t})$):** Applying a natural logarithm normalizes percentage-based changes instead of absolute currency changes 6\. It stabilizes variance, directly mimics continuously compounded returns, and usually exhibits the symmetry needed for accurate Z-score modeling 6\.  
* **Regression residual / hedge-ratio spread ($S\_t \= P\_{A,t} \- (\\beta \\cdot P\_{B,t} \+ \\alpha)$):** The most mathematically sound approach. It uses Ordinary Least Squares (OLS) to find an optimal hedge ratio ($\\beta$), which represents the exact percentage weight to short asset B for every share of A bought, strictly minimizing the spread's variance 7\.

**3\. Which spread definition is most appropriate for a robust but simple Week 2 implementation?**The **Log Price Ratio** is the optimal choice for an early-stage implementation 8\. It bypasses the complexity of rolling regressions and cointegration tests required for the regression residual method, while still handling different price scales gracefully and providing variance stability 6, 8\.  
**4\. How is rolling Z-score computed from a spread series?**The Z-score standardizes the spread, converting absolute price differences into a dimensionless metric representing standard deviations away from the historical mean 9\. It is computed using a rolling window (e.g., 60 periods) to continuously adapt to recent market behavior without look-ahead bias 10, 11\. The basic formula is:$$Z\_t \= \\frac{S\_t \- \\mu\_t}{\\sigma\_t}$$Where $S\_t$ is the current spread, $\\mu\_t$ is the rolling mean, and $\\sigma\_t$ is the rolling standard deviation 12\. For optimal computational speed and numerical stability in high-frequency engines, developers use **Welford’s Online Algorithm** to update these rolling metrics in $O(1)$ constant time 11, 13\.  
**5\. What does a Z-score of \+2 or \-2 mean in trading terms?**In a normal distribution, \~95.45% of all occurrences fall within two standard deviations 14\. Thus, reaching $\\pm2$ means the price relationship has diverged to an extent that occurs less than 5% of the time by chance 14\.

* **A Z-score $\> \+2$:** The spread is exceptionally wide. Asset A is currently overpriced relative to Asset B. The trading action is to **short the spread** (sell Asset A, buy Asset B) 15\.  
* **A Z-score $\< \-2$:** The spread is exceptionally narrow. Asset A is underpriced relative to Asset B. The trading action is to **long the spread** (buy Asset A, sell Asset B) 15\.

**6\. What technical problems can make Z-score misleading?**

* **Non-stationary spread:** If the fundamental relationship between the assets breaks ("pair divorce"), the spread will drift permanently 16\. A Z-score might stick near $+2$ as the assets continuously diverge, tricking you into a trade that never reverts and accumulates massive losses 17\.  
* **Unstable variance (Heteroscedasticity):** In quiet markets with low variance, a tiny, meaningless price tick can cause the Z-score to artificially explode to $\\pm 10$. Conversely, during a macro market shock, massive variance spikes can suppress the Z-score, masking genuine historical mispricings 18\.  
* **Zero standard deviation:** If an asset doesn't trade for a few minutes (illiquidity), standard deviation drops to exactly zero. This causes a division-by-zero error that will yield a NaN or Infinity Z-score, crashing the algorithm 19\.  
* **Missing timestamps:** If data feeds are asynchronous, using "forward-filled" missing data artificially compresses volatility 20\. When a new tick finally arrives, the sudden change against an artificially tight variance triggers a massive, false Z-score spike 20\.  
* **Intraday noise:** On high-resolution data, micro-fluctuations can cause the Z-score to jitter back and forth across the $\\pm2.0$ boundary in seconds 21\. This triggers rapid-fire open/close orders ("repainting"), which will bleed your account dry via transaction fees 21\.

**7\. Minimum mathematically correct version to implement first:**For a realistic, homework-style implementation, your code should do the following in sequence:

1. **Data Alignment:** Use Adjusted Close prices and perform an "inner join" on the timestamps so your engine only calculates metrics on periods where *both* assets actively traded 22, 23\.  
2. **Transformation:** Calculate the Log Price Ratio spread: $S\_t \= \\ln(P\_{A,t}) \- \\ln(P\_{B,t})$ 23, 24\.  
3. **Rolling Calculations:** Use a 60-period rolling window (e.g., rolling(window=60) in Pandas) to find the moving average and standard deviation 24\.  
4. **Z-Score Guarding:** Implement a volatility floor (epsilon, e.g., $1e-10$) in your standard deviation denominator to mathematically prevent division-by-zero crashes 24\.  
5. **State Machine:** Generate $+2 / \-2$ entry signals and $0$ exit signals, but strictly implement a state tracker (holding vs. flat) so you do not generate duplicate buy orders while a position is already open 24, 25\.

### Recommended Default Formula Set for Week 2

* **Spread Definition:** $S\_t \= \\ln(P\_{A,t}) \- \\ln(P\_{B,t})$ 24  
* **Rolling Mean:** $\\mu\_t \= \\frac{1}{n} \\sum\_{i=t-n+1}^{t} S\_i$ (where $n \= 60$) 12, 24  
* **Rolling Std Dev:** $\\sigma\_t \= \\sqrt{\\frac{1}{n-1} \\sum\_{i=t-n+1}^{t} (S\_i \- \\mu\_t)^2}$ 12  
* **Guarded Z-Score:** $Z\_t \= \\frac{S\_t \- \\mu\_t}{\\max(\\sigma\_t, \\epsilon)}$ (where $\\epsilon \= 1e-10$) 19, 24  
* **Trading Logic:**  
* **Enter Short Spread:** IF $Z\_t \> 2.0$ AND State \== Flat 15, 25  
* **Enter Long Spread:** IF $Z\_t \< \-2.0$ AND State \== Flat 15, 25  
* **Exit Position:** IF (Position \== Short AND $Z\_t \\le 0$) OR (Position \== Long AND $Z\_t \\ge 0$) 24, 26

