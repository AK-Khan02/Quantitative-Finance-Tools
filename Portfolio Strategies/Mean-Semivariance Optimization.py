import numpy as np
from scipy.optimize import minimize

# Define the objective function for mean-semivariance optimization
def mean_semivariance(weights, mean_returns, cov_matrix, target_return):
    portfolio_return = np.sum(weights * mean_returns)
    lower_returns = mean_returns[mean_returns < target_return]
    semivariance = np.mean((lower_returns - target_return) ** 2)
    return semivariance

# Example data
mean_returns = np.array([0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])

target_return = 0.14  # Sample Target Return

# Initial guess for portfolio weights
initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

# Define the constraint that weights should sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Set bounds for portfolio weights (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(len(mean_returns)))

# Minimize the semivariance objective function
result = minimize(mean_semivariance, initial_weights, args=(mean_returns, cov_matrix, target_return),
                  method='SLSQP', constraints=constraints, bounds=bounds)

# Extract the optimized portfolio weights
optimal_weights = result.x
print("Optimal Portfolio Weights:", optimal_weights)
