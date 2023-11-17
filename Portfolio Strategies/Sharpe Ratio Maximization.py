# Necessary Library Imports
import numpy as np
import pandas as pd
import yfinance as finance_api
from scipy.optimize import minimize
import datetime

# Configuration for Enhanced Data Download Speed
finance_api.pdr_override()

# Input Parameters Initialization
ticker_symbols = ['FB', 'AAPL', 'MRK']
time_span = 365 * 7
start_period = datetime.datetime.now() - datetime.timedelta(days=time_span)
end_period = datetime.datetime.now()
base_rate = 0.003

def fetch_stock_data(tickers, start, end):
    """Fetch historical stock data."""
    return finance_api.download(tickers, start, end)['Adj Close']

def compute_logarithmic_returns(data):
    """Calculate daily logarithmic returns."""
    return np.log(data) - np.log(data.shift(1))

def annualize_returns(log_returns):
    """Convert daily log returns to annual returns."""
    annual_returns = np.exp(log_returns.groupby(log_returns.index.year).sum()) - 1
    return annual_returns

def get_covariance_matrix(returns):
    """Compute covariance matrix of returns."""
    return np.cov(returns.T)

def calculate_portfolio_risk(cov_matrix, weight_vector):
    """Calculate portfolio risk."""
    return np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))

def calculate_return(returns, weights):
    """Calculate portfolio return."""
    return np.dot(returns.mean(), weights)

def calculate_sharpe_ratio(returns, weights, risk_free_rate):
    """Compute Sharpe ratio of a portfolio."""
    portfolio_return = calculate_return(returns, weights)
    portfolio_risk = np.sqrt(calculate_portfolio_risk(get_covariance_matrix(returns), weights))
    return (portfolio_return - risk_free_rate) / portfolio_risk

def optimize_portfolio(returns, risk_free_rate):
    """Find the optimal portfolio weights to maximize Sharpe Ratio."""
    init_guess = np.full(returns.shape[1] - 1, 1 / returns.shape[1])

    def objective_function(weights):
        complete_weights = np.append(weights, 1 - np.sum(weights))
        return -calculate_sharpe_ratio(returns, complete_weights, risk_free_rate)

    optimal_weights = minimize(objective_function, init_guess).x
    complete_optimal_weights = np.append(optimal_weights, 1 - np.sum(optimal_weights))
    return complete_optimal_weights

# Data Preparation and Processing
historical_data = fetch_stock_data(ticker_symbols, start_period, end_period)
log_returns = compute_logarithmic_returns(historical_data)
annualized_returns = annualize_returns(log_returns)

# Portfolio Analysis
uniform_weights = np.full(len(ticker_symbols), 1 / len(ticker_symbols))
uniform_sharpe_ratio = calculate_sharpe_ratio(annualized_returns, uniform_weights, base_rate)

optimal_weights = optimize_portfolio(annualized_returns, base_rate)
optimal_sharpe_ratio = calculate_sharpe_ratio(annualized_returns, optimal_weights, base_rate)

# Output Results
print('Ticker Symbols: ', ticker_symbols)
print('Uniform Portfolio: Weights:', uniform_weights, 'Sharpe Ratio:', uniform_sharpe_ratio)
print('Optimal Portfolio: Weights:', optimal_weights, 'Sharpe Ratio:', optimal_sharpe_ratio)

