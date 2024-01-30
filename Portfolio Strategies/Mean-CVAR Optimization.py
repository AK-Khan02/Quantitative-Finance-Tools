import numpy as np
from scipy.optimize import minimize

# Define the objective function for mean-CVaR optimization
def mean_cvar(weights, mean_returns, cov_matrix, alpha):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate CVaR at alpha level
    sorted_returns = np.sort(mean_returns)
    threshold_index = int(alpha * len(sorted_returns))
    cvar = -np.mean(sorted_returns[:threshold_index])
    
    return cvar

# Example data
mean_returns = np.array([0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])
alpha = 0.05  # Confidence level for CVaR optimization, adjust as needed

# Initial guess for portfolio weights
initial_weights = np.ones(len(mean_returns)) / len(mean_returns)

# Define the constraint that weights should sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Set bounds for portfolio weights (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(len(mean_returns)))

# Minimize the CVaR objective function
result = minimize(mean_cvar, initial_weights, args=(mean_returns, cov_matrix, alpha),
                  method='SLSQP', constraints=constraints, bounds=bounds)

# Extract the optimized portfolio weights
optimal_weights = result.x
print("Optimal Portfolio Weights:", optimal_weights)
