import numpy as np
from scipy.optimize import minimize

# Sample data: Asset names, expected returns, covariance matrix, and market cap weights
assets = ['Asset 1', 'Asset 2', 'Asset 3']
expected_returns = np.array([0.05, 0.06, 0.07])  # Equilibrium returns
cov_matrix = np.array([[0.01, 0.0012, 0.0018],
                       [0.0012, 0.014, 0.0024],
                       [0.0018, 0.0024, 0.019]])  # Covariance matrix
market_weights = np.array([0.6, 0.3, 0.1])  # Market cap weights

# Investor's views (P: Pick matrix, Q: Expected returns for the views)
P = np.array([[1, -1, 0], [0, 1, -1]])
Q = np.array([0.02, 0.03])

# Black-Litterman model parameters
tau = 0.05  # Scalar indicating the uncertainty of the equilibrium return estimate
omega = np.dot(np.dot(P, cov_matrix), P.T) * np.eye(Q.shape[0])  # Uncertainty of views

# Compute the Black-Litterman expected returns
inverse = np.linalg.inv(np.dot(tau, cov_matrix))
M = np.dot(np.dot(np.dot(inverse, P.T), np.linalg.inv(np.dot(np.dot(P, inverse), P.T) + omega)), (Q - np.dot(P, expected_returns)))
bl_returns = expected_returns + np.dot(np.dot(tau, cov_matrix), M)

# Mean-variance optimization using the Black-Litterman expected returns
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def objective(weights):
    return portfolio_variance(weights, cov_matrix)

# Constraints: sum of weights = 1, and no short selling
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in assets)

# Optimization
initial_weights = np.array([1./len(assets)]*len(assets))
optimized = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

print(f"Optimal Portfolio Weights: {optimized.x}")
