# Importing necessary libraries for data manipulation, financial analysis, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import datetime as dt
import warnings

# Importing libraries for financial data acquisition and analysis
import pandas_datareader
from pandas_datareader import data
import quantstats as qs
import yfinance as yf
import ta

# Importing libraries for portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, black_litterman, BlackLittermanModel

# Importing machine learning model for potential financial modeling
from sklearn.linear_model import LinearRegression

# Enabling Plotly offline mode for interactive plotting in notebooks
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Setting up environment configurations
warnings.filterwarnings("ignore")  # Hiding warnings for cleaner notebook presentation

"""# --- Data Acquisition and Preprocessing ---

"""

# Benchmark data: Downloading historical returns for the S&P 500 index
benchmark_returns = qs.utils.download_returns('^GSPC')
benchmark_returns = benchmark_returns.loc['2018-01-01':'2022-12-31']
benchmark_returns.index = benchmark_returns.index.tz_localize(None) if benchmark_returns.index.tz is not None else benchmark_returns.index

# Selecting new tech stock symbols for analysis
tech_stocks = ['AAPL', "TSLA"]
tech_returns = {}

# Fetching and preparing daily returns for the chosen tech stocks
for symbol in tech_stocks:
    stock_return = qs.utils.download_returns(symbol)
    stock_return = stock_return.loc['2018-01-01':'2022-12-31']
    stock_return.index = stock_return.index.tz_localize(None) if stock_return.index.tz is not None else stock_return.index
    tech_returns[symbol] = stock_return

# Equal weight allocation for a simple initial portfolio
initial_weights = [0.5] * len(tech_stocks)

# Assembling the initial portfolio by aggregating weighted returns
initial_portfolio = sum(tech_returns[symbol] * weight for symbol, weight in zip(tech_stocks, initial_weights))

# Performance report for the initial portfolio
qs.reports.full(initial_portfolio, benchmark=benchmark_returns)

# Fetching adjusted close prices for optimization and further analysis
tech_prices = {}
for symbol in tech_stocks:
    tech_prices[symbol] = yf.download(symbol, start='2018-01-01', end='2022-12-31')['Adj Close']

# Aggregating adjusted close prices into a single DataFrame
prices_df = pd.concat(tech_prices.values(), axis=1)
prices_df.columns = tech_stocks

"""# --- Portfolio Optimization ---

"""

# Calculating expected returns and sample covariance for the optimization
expected_mu = expected_returns.mean_historical_return(prices_df)
sample_S = risk_models.sample_cov(prices_df)

# Efficient Frontier optimization for maximal Sharpe ratio
efficient_frontier = EfficientFrontier(expected_mu, sample_S)
sharpe_weights = efficient_frontier.max_sharpe()
optimized_weights = efficient_frontier.clean_weights()

# Displaying the optimized portfolio weights
print("Optimized Portfolio Weights:", optimized_weights)

# Constructing the optimized portfolio with updated weights
optimal_weights_list = [0.75393, 0.24607]
optimized_portfolio = sum(tech_returns[symbol] * weight for symbol, weight in zip(tech_stocks, optimal_weights_list))

# Evaluating the optimized portfolio against the initial one
qs.reports.full(optimized_portfolio, benchmark=initial_portfolio)

"""# --- Advanced Optimization with Black-Litterman Model ---"""

# Market capitalization data for Black-Litterman model input
market_caps_bl = {symbol: yf.Ticker(symbol).info["marketCap"] for symbol in tech_stocks}

# Market-implied risk aversion using S&P 500 index prices
sp500_prices = yf.download("^GSPC", start='2018-01-01', end='2022-12-31')['Adj Close']
delta_market = black_litterman.market_implied_risk_aversion(sp500_prices)

# Investor's views and confidences for Black-Litterman model
views_Q = np.array([0.06, 0.04])  # Hypothetical views on returns
views_P = np.eye(len(tech_stocks))  # Direct mapping of views to tech stocks
views_confidence = [0.6, 0.5]  # Confidence levels in the views

# Initializing Black-Litterman model with market data and investor's views
black_litterman_model = BlackLittermanModel(sample_S, pi=expected_mu, Q=views_Q, P=views_P, omega='idzorek', view_confidences=views_confidence)

# Expected returns from the Black-Litterman model
bl_expected_returns = black_litterman_model.bl_returns()

# Portfolio optimization based on Black-Litterman model returns
efficient_frontier_bl = EfficientFrontier(bl_expected_returns, sample_S)
efficient_frontier_bl.max_sharpe()
bl_weights = efficient_frontier_bl.clean_weights()
bl_weights

# Constructing the Black-Litterman optimized portfolio
# Note: Update 'bl_optimal_weights' based on 'bl_weights' output for actual use
bl_optimal_weights = [0.79112, 0.20888]  # Example weights, should be replaced with actual BL weights
bl_optimized_portfolio = sum(tech_returns[symbol] * weight for symbol, weight in zip(tech_stocks, bl_optimal_weights))

# Performance evaluation of the Black-Litterman optimized portfolio against the initial one
qs.reports.full(bl_optimized_portfolio, benchmark=initial_portfolio)

