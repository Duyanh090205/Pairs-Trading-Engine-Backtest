Here is a practical, implementation-oriented breakdown of your Dev Role task for Week 2, strictly focusing on the vectorization and generation of the Z-Score signals.  
**1\. What exactly is the dev role task for Week 2?**Your core task is to build the algorithmic engine that calculates the spread between two assets, normalizes that spread into a rolling Z-score, and generates entry and exit trading signals 1-3. Because "speed matters," your implementation must rely entirely on Pandas' built-in vectorized functions (like .rolling()) to process the entire time-series dataset at once, avoiding slow, iterative for loops 3\.  
**2\. What does “calculate when the pair spreads too far apart” mean in practical quantitative terms?**In quantitative terms, this means the relationship between the two assets has deviated from its historical norm by a statistically significant margin 4, 5\. Practically, your code will measure this by checking when the **Z-score crosses a predefined threshold**, which is typically set at **\+2.0 or \-2.0 standard deviations** from the rolling mean 4, 5\. A Z-score above \+2.0 means the spread is exceptionally wide, while below \-2.0 means it is exceptionally narrow 4, 6\.  
**3\. What is the role of spread, rolling mean, rolling standard deviation, and Z-score in this task?**

* **Spread:** The baseline input of your system. It is the mathematical difference between the two asset prices (often calculated using natural logarithms to stabilize variance and normalize price scales) 7\.  
* **Rolling Mean ($\\mu\_t$):** The dynamic equilibrium line. By calculating the moving average of the spread over a specific lookback window (e.g., 60 periods), your system establishes what the "normal" spread looks like at any given time 1\.  
* **Rolling Standard Deviation ($\\sigma\_t$):** The dynamic measure of volatility. This tells your system how wildly the spread is fluctuating around the rolling mean over that same lookback window 1\.  
* **Z-Score:** The standardized trading signal. Calculated as (Current Spread \- Rolling Mean) / Rolling Standard Deviation, the Z-score translates absolute price differences into a standardized, dimensionless metric that dictates exactly when to trigger a trade 1\.

**4\. What outputs should a correct dev implementation produce?**A correct implementation should produce a time-aligned Pandas DataFrame containing the following columns for every timestamp:

* The raw and/or log prices of both assets 7, 8\.  
* The calculated **Spread** 9\.  
* The **Rolling Mean** and **Rolling Standard Deviation** 1\.  
* The normalized **Z-Score** 1\.  
* The **Trading Signals**, typically represented as integers (+1 for going long the spread, \-1 for shorting the spread, and 0 for closing positions/staying flat) 10, 11\.  
* *Crucially*, it should include a **State Machine** representation that tracks open vs. closed positions, ensuring you don't generate continuous entry signals while a trade is already active 12, 13\.

**5\. Based on the uploaded materials, what assumptions are safe to make and what assumptions would be risky?**

* **Safe Assumptions:**  
* It is safe to assume you should use **Adjusted Close prices** to automatically account for stock splits and dividends, which would otherwise create false divergence signals 8\.  
* It is safe (and highly recommended) to **add a tiny constant (epsilon)** to your rolling standard deviation calculation to serve as a volatility floor. This prevents division-by-zero errors during periods of zero trading activity 3, 14\.  
* It is safe to use the **Log Price Ratio** to calculate your spread for a "Week 2" assignment, as it is simple to vectorize and robust across assets with different nominal price scales 7, 15\.  
* **Risky Assumptions:**  
* It is incredibly risky to assume your datasets are perfectly synchronized. If you don't explicitly align your timestamps (e.g., using an inner join) or handle missing data properly, your system will calculate erroneous spreads 2, 16\.  
* It is risky to allow **look-ahead bias**. If your rolling window accidentally includes the current row's closing price before the signal is generated, your backtest will be invalid 2, 17\.

**6\. How does Sam’s instruction about using prior 6–12 months and then evaluating the next 6–12 months change the technical implementation?**This instruction dictates a strict **Formation Period (training) vs. Trading Period (out-of-sample testing)** split 18, 19\. Technically, it means you cannot calculate your core relationship parameters (like the hedge ratio or determining if the assets are cointegrated) over the entire dataset 15, 20.In Pandas, you must slice your DataFrame. You will use the first 6–12 months *only* to calculate the fixed parameters (like the hedge ratio multiplier) 15, 20\. Then, you apply those fixed parameters to the second 6–12 month slice to calculate the spread, using .rolling() windows strictly on the out-of-sample data to generate your dynamic Z-scores and signals 20\.

### Dev Role Checklist

To successfully code the vectorization task, your script must:

*  Import and inner-join the two asset datasets to align timestamps 2\.  
*  Convert prices to natural logarithms (if using the Log Price Ratio method) 2\.  
*  Split the DataFrame into a Formation (training) period and Trading (testing) period 18, 19\.  
*  Calculate the constant hedge ratio using only the Formation period data 15, 20\.  
*  Calculate the Spread on the Trading period using Pandas vectorization 3\.  
*  Use df.rolling(window=n) to calculate the Rolling Mean and Rolling Standard Deviation 3\.  
*  Apply a max(std, epsilon) floor to the standard deviation to prevent division-by-zero errors 3, 14\.  
*  Calculate the Z-score without look-ahead bias 2, 17\.  
*  Generate binary entry/exit signals (+1, \-1, 0\) using vectorized conditional logic (e.g., np.where) when the Z-score breaches $\\pm2.0$ 21, 22\.  
*  Implement state-tracking logic to manage open positions vs. new signals 12\.

