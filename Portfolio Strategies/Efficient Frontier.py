import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Sample data: Asset names, their expected returns, volatilities, and a covariance matrix
assets = ['Asset 1', 'Asset 2', 'Asset 3']
expected_returns = np.array([0.10, 0.12, 0.14])  # Expected returns
volatilities = np.array([0.15, 0.20, 0.25])      # Asset volatilities
corr_matrix = np.array([[1.0, 0.2, 0.3],
                        [0.2, 1.0, 0.4],
                        [0.3, 0.4, 1.0]])         # Correlation matrix

# Calculate the covariance matrix from volatilities and correlation matrix
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# Function to calculate portfolio return
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

# Function to calculate portfolio volatility
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function to minimize (for finding minimum volatility for a given return)
def min_func_variance(weights, returns, cov_matrix, target_return):
    return portfolio_volatility(weights, cov_matrix) if portfolio_return(weights, returns) >= target_return else float('inf')

# Number of portfolios to simulate
num_portfolios = 10000

# Store returns, volatility, and weights of all the portfolios
all_weights = np.zeros((num_portfolios, len(assets)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)
    
    all_weights[i, :] = weights
    ret_arr[i] = portfolio_return(weights, expected_returns)
    vol_arr[i] = portfolio_volatility(weights, cov_matrix)

# Plotting the portfolios
plt.scatter(vol_arr, ret_arr, c=ret_arr/vol_arr, marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Variables to store the efficient frontier
eff_returns = np.linspace(min(ret_arr), max(ret_arr), 200)
eff_volatilities = []

# Constraints (weights sum to 1)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Boundaries for weights
bounds = tuple((0, 1) for asset in assets)

# Calculate efficient frontier
for ret in eff_returns:
    cons = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, expected_returns) - ret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(portfolio_volatility, num_assets*[1./num_assets,], args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=cons)
    eff_volatilities.append(result['fun'])

# Plotting the efficient frontier
plt.plot(eff_volatilities, eff_returns, 'r--', linewidth=3)
plt.title('Efficient Frontier')
plt.show()
