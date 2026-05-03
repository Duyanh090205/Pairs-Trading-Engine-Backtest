# 1987 Crash Strategist Memo Outline

## Thesis
During the 1987 crash, portfolio insurance algorithms sold into falling markets not because they misread value, but because they were built to mechanically reduce risk through dynamic hedging. When many institutions followed the same rule in a rapidly deteriorating liquidity environment, the strategy became a self-reinforcing feedback loop that amplified the crash.

## 1. Executive Summary
- Portfolio insurance aimed to limit downside by reducing equity exposure as prices fell.
- In the 1987 crash, this logic became destabilizing because many institutions sold simultaneously.
- The core failure was structural: mechanical selling rules collided with collapsing liquidity.
- The strategy was individually rational but systemically dangerous.

## 2. What the Data Shows
- The minute-level data shows an accelerated and disorderly decline, not a smooth correction.
- Extreme negative moves cluster in a short period, alongside rising volatility and drawdown.
- The chart should identify an approximate break point where market stress became self-reinforcing.
- This evidence supports the claim that normal market functioning deteriorated.

## 3. How Portfolio Insurance Worked
- Portfolio insurance sought synthetic downside protection without buying actual puts.
- It used dynamic hedging, typically by selling futures as prices fell.
- The strategy assumed continuous liquidity and orderly execution.
- Those assumptions failed during the crash.

## 4. Why the Algorithms Sold
- The models were designed to reduce risk, not buy undervalued assets.
- Falling prices triggered more selling, especially in futures markets.
- Similar rules across institutions created synchronized selling and a feedback loop.
- The problem was not a coding error, but a procyclical strategy in a stressed market.

## 5. Risk-Control Proposal
- Stop-loss and hedging systems should not rely on price triggers alone.
- They should include liquidity checks, volatility filters, and execution throttles.
- Human override or circuit-breaker logic should activate during disorderly conditions.
- Risk systems must account for market impact and herd behavior, not just portfolio protection.

## Logical Flow Between Sections
- **1 → 2:** Start with the conclusion, then ground it in the market evidence from the chart and metrics.
- **2 → 3:** After showing that the market became disorderly, step back and explain the strategy operating inside that environment.
- **3 → 4:** Once the reader understands how portfolio insurance worked, explain why its built-in logic forced selling and why that logic became dangerous at scale.
- **4 → 5:** End by converting the diagnosis into a forward-looking recommendation: if the failure came from mechanical selling under stressed liquidity, the solution is liquidity-aware and impact-aware risk control.

## Recommendation for the Stop-Loss / Risk-Control Proposal
Large-scale hedging and stop-loss systems should be liquidity-aware, volatility-aware, and capable of slowing or suspending execution when their own trades risk amplifying market stress.

Suggested safeguards:
- **Liquidity filter:** Do not force sales when spreads widen sharply or market depth collapses.
- **Volatility filter:** Reduce or pause automatic execution during extreme short-term volatility.
- **Execution throttle:** Spread trades over time instead of dumping exposure all at once.
- **Human or circuit-breaker override:** Require supervisory review when trading conditions become disorderly.