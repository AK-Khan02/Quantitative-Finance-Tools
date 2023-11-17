"""
This Program Gets a list of 10 stocks, and creates a portfolio of those 10 stocks, each with its own respective weightage

The program then uses a simple moving average algorithm and compares the portfolio we made above to a portfolio where each stock
is given equal weightage

The program graphs the two returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yfinance
from datetime import datetime, timedelta

def simulate_portfolio(stock_symbols, weights, start_date, end_date, forecast_days):
    # Download stock data
    stock_data = yfinance.download(stock_symbols, start=start_date, end=end_date)['Adj Close']
    
    # Predict future prices using a simple moving average
    predicted_prices = stock_data.rolling(window=forecast_days).mean().shift(-forecast_days)
    
    # Initial investment and number of stocks
    initial_investment = 100000  # e.g., $100,000
    number_of_stocks = len(stock_symbols)

    # Simulating the custom weighted portfolio
    custom_portfolio = (predicted_prices / predicted_prices.iloc[0]) * weights * initial_investment
    custom_portfolio_return = custom_portfolio.sum(axis=1)

    # Simulating an equally weighted portfolio
    equal_weights = np.full(number_of_stocks, 1 / number_of_stocks)
    equal_weighted_portfolio = (predicted_prices / predicted_prices.iloc[0]) * equal_weights * initial_investment
    equal_weighted_return = equal_weighted_portfolio.sum(axis=1)

    return custom_portfolio_return, equal_weighted_return

# Configuration
stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'BRK-B', 'JPM', 'V', 'NVDA']
custom_weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1]  # Example custom weights
start_date = datetime.now() - timedelta(days=365 * 2)
end_date = datetime.now()
forecast_days = 5

# Run simulation
custom_return, equal_return = simulate_portfolio(stock_symbols, custom_weights, start_date, end_date, forecast_days)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(custom_return, label='Custom Weighted Portfolio')
plt.plot(equal_return, label='Equally Weighted Portfolio')
plt.title('Portfolio Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()
