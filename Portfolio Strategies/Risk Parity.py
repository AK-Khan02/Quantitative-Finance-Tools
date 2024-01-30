import numpy as np
from scipy.optimize import minimize

# Example data
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])

# Function to calculate portfolio risk for given weights and covariance matrix
def portfolio_risk(weights, cov_matrix):
    port_variance = weights.T @ cov_matrix @ weights
    return np.sqrt(port_variance)

# Function for risk parity objective
def risk_parity_objective(weights, cov_matrix):
    portfolio_risks = np.array([portfolio_risk(weights, cov_matrix)])
    risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_risks
    target_risk = 1 / len(weights)  # Equal risk contribution for each asset
    return np.sum((risk_contributions - target_risk) ** 2)

# Initial guess for portfolio weights
n_assets = cov_matrix.shape[0]
initial_weights = np.ones(n_assets) / n_assets

# Set bounds for portfolio weights (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(n_assets))

# Minimize risk parity objective
result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),
                  method='SLSQP', bounds=bounds)

# Extract the optimized portfolio weights
optimal_weights = result.x
print("Optimal Portfolio Weights (Risk Parity):", optimal_weights)
