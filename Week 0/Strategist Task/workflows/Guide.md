# Week 0 Strategist Project Guide

This guide outlines the methodology for dissecting the 1987 market crash and explaining how Portfolio Insurance caused the algorithms to sell when they should have bought.

## Core Objective
Deliver a 650-850 word professional memo, a chart of the market break, and a modern risk-control proposal.

## Approach
1. **Data Visualization (Deliverable 1)**
   Load your provided `1987_crash_market_data.csv` via Jupyter Notebook or your preferred Python script environment. Plot the index values around October 19, 1987. Annotate the exact point the market "broke" due to the algorithmic selling cascade. Save to `outputs/crash_plot.png`.

2. **Source Synthesis & Note Taking**
   Read the provided Portfolio Insurance Report. Document in `notes/portfolio_insurance_notes.md` how the strategy involved shorting S&P futures as portfolio values dipped, driving cash prices down and triggering further mechanical selling. 

3. **Drafting (Deliverable 2 & 3)**
   In `outputs/strategist_memo.md`, compose the memo. Structure it cleanly:
   - **Executive Summary**
   - **The Portfolio Insurance Mechanicism** (why it failed)
   - **The Market Break** (reference your chart)
   - **Risk-Control Proposal** (provide actionable constraints like dynamic hedging limits/circuit breakers).

4. **Review & Refine**
   Check your drafted memo. Ensure it reads like a professional memo, the word count is strictly 650-850 words, and the English is clear and concise.
