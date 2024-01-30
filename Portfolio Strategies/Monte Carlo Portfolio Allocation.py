import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stocks we're interested in
stocks = ['AAPL', 'MSFT', 'GOOG', 'FB']

# Fetch historical data
start_date = '2020-01-01'
end_date = '2020-12-31'
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Number of portfolios to simulate
num_portfolios = 10000

# Set up array to hold results
results = np.zeros((4, num_portfolios))
for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    # Calculate portfolio return and volatility
    portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualize returns
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualize volatility

    # Calculate Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility

    # Store results in results array
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio
    results[3, i] = max_sharpe_idx = results[2].argmax()

# Extract the portfolio with the highest Sharpe ratio
max_sharpe_ratio = results[2, max_sharpe_idx]
optimal_return = results[0, max_sharpe_idx]
optimal_volatility = results[1, max_sharpe_idx]

print(f"Optimal Portfolio Return: {optimal_return}")
print(f"Optimal Portfolio Volatility: {optimal_volatility}")
print(f"Maximum Sharpe Ratio: {max_sharpe_ratio}")

# Plot the simulated portfolios
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(optimal_volatility, optimal_return, color='red', s=50)  # Mark the optimal portfolio
plt.show()
