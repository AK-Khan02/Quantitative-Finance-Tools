# Mean Variance Optimization
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Define assets and download historical data
assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
data = yf.download(assets, start='2020-01-01', end='2021-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Mean returns and covariance
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Optimization function
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

# Objective function to minimize (Sharpe Ratio)
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(assets)))

# Initial guess and optimization
init_guess = len(assets) * [1. / len(assets),]
opt_result = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
opt_weights = opt_result.x
