# Black Litterman and Portfolio Optimization Analysis

## Summary

This project focuses on financial portfolio optimization using historical stock data. It encompasses data acquisition, preprocessing, performance analysis, and optimization techniques including the Efficient Frontier and Black-Litterman model. The analysis primarily targets tech stocks, aiming to construct an optimized investment portfolio that balances risk and return.

## Features

- **Data Acquisition**: Fetches historical stock returns and prices using libraries like `yfinance` and `pandas_datareader`.
- **Preprocessing**: Cleans and prepares data for analysis, aligning timestamps and filtering relevant date ranges.
- **Initial Portfolio Construction**: Builds a simple equal-weighted portfolio of selected tech stocks to serve as a baseline.
- **Performance Analysis**: Utilizes `quantstats` to generate comprehensive performance reports comparing the portfolio against benchmarks like the S&P 500 index.
- **Optimization**: Implements the Efficient Frontier to find the optimal weights that maximize the Sharpe ratio. Further, it explores the Black-Litterman model to incorporate investor views into the optimization process.

## How It Works

1. **Data Acquisition**: The script starts by downloading historical return data for selected stocks and a benchmark index for comparison.
2. **Preprocessing**: Adjusts data for time zone differences and filters it based on the specified date range.
3. **Portfolio Analysis**: Analyzes the performance of an initial equal-weighted portfolio against the benchmark.
4. **Optimization**:
    - **Efficient Frontier**: Calculates expected returns and covariance, then optimizes the portfolio to maximize the Sharpe ratio.
    - **Black-Litterman Model**: Incorporates investor views and confidence levels to adjust the expected returns, followed by re-optimization.
5. **Performance Reporting**: Generates detailed reports comparing the optimized portfolios with the initial portfolio and benchmark.

## Installation

To run this project, you will need to install the required Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly scipy yfinance pandas_datareader quantstats ta scikit-learn
```

## Usage
Run the script in a Jupyter notebook or any Python environment. Ensure you have an internet connection for data download. The script is structured in sections for clarity and ease of navigation.
