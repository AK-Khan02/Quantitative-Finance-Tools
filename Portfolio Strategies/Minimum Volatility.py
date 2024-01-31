import numpy as np
from scipy.optimize import minimize

# Example data
mean_returns = np.array([0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])

# Define the objective function for minimum volatility optimization
def min_volatility_objective(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_volatility

# Initial guess for portfolio weights
initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

# Define the constraint that weights should sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Set bounds for portfolio weights (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(len(mean_returns)))

# Minimize the minimum volatility objective function
result = minimize(min_volatility_objective, initial_weights, args=(cov_matrix,),
                  method='SLSQP', constraints=constraints, bounds=bounds)

# Extract the optimized portfolio weights
optimal_weights = result.x
print("Optimal Minimum Volatility Portfolio Weights:", optimal_weights)
