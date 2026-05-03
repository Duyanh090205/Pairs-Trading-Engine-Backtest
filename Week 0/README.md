# 1987 Black Monday Backtest & Market Break Detection

This repository contains the deliverables for the Quantitative Finance Program - Week 0 project. The primary focus of this project is the analysis of the 1987 Black Monday market crash and the development of a robust, generalized Market Break Detection system.

## Project Scope
The objective of this project is to build an institutional-grade, multivariate changepoint detection model capable of identifying structural market breaks in real-time or historical data.

Key areas of scope include:
- **Multivariate Feature Engineering**: Analyzing market microstructure using Autocorrelation, Price Staleness, and Jump-Share to identify early signs of market dysfunction.
- **Changepoint Detection**: Implementing the PELT (Pruned Exact Linear Time) algorithm to find statistically significant regime shifts across multiple features simultaneously.
- **Generalization**: Refactoring the detection logic so it can easily adapt to other historical crises (e.g., the 2008 Financial Crisis, the 2020 COVID Crash) using standard market data.
- **Strategic Evaluation**: Producing automated, professional-grade strategic memos and autopsy reports on market behavior during extreme volatility.

## Deliverables & Directory Structure

- **`week0_submission.ipynb`**: The primary Jupyter Notebook containing the end-to-end analysis, methodology, and visualizations for the Week 0 submission.
- **`Dev Task/`**: Contains the core Python scripts for market break detection, including `analyze_crash.py` for multivariate PELT analysis, `test_multiple_crashes.py` for yfinance data ingestion, and extensive documentation on parameter tuning.
- **`Strategist Task/`**: Includes code to generate strategic analysis and output PDF memos detailing market regime shifts and insights.
- **`Vibe Coding Task/`**: Contains experimental analysis, visualizations, and automated post-mortem autopsy reports of the crash scenarios.
- **`data/`**: Historical market data CSV files (e.g., 1987 market data, 2008, 2020, etc.) used for testing and backtesting the detection models.
- **`outputs/`**: Generated charts, feature convergence plots, and diagnostic logs (e.g., `market_break_changepoint.png`, `crash_autopsy.png`).

## Key Findings (1987 Black Monday)
- The multivariate PELT model successfully detects the onset of the October 19, 1987 crash at **09:42 AM** (8 minutes into the crash), providing a more robust signal than univariate indicators.
- The convergence of negative autocorrelation (bid-ask friction), high staleness, and discontinuous jumps proved to be a highly reliable signature of a severe market break.
