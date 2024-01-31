import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Example data: Historical returns for multiple assets
data = {
    'Asset1': [0.02, 0.03, 0.02, 0.01, 0.02],
    'Asset2': [0.03, 0.02, 0.01, 0.03, 0.04],
    'Asset3': [0.04, 0.05, 0.04, 0.05, 0.06],
    'Asset4': [0.02, 0.01, 0.03, 0.02, 0.03]
}

returns_df = pd.DataFrame(data)

# Rolling window parameters
window_size = 3  # Adjust the window size as needed
n = len(returns_df)
optimal_weights_list = []

for i in range(n - window_size + 1):
    # Get historical data for the rolling window
    historical_returns = returns_df.iloc[i:i+window_size]
    
    # Calculate mean returns and covariance matrix for the window
    mean_returns = historical_returns.mean()
    cov_matrix = historical_returns.cov()
    
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

    # Minimize the minimum volatility objective function for the window
    result = minimize(min_volatility_objective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints, bounds=bounds)

    # Extract the optimized portfolio weights for the window
    optimal_weights = result.x
    optimal_weights_list.append(optimal_weights)

# The 'optimal_weights_list' contains portfolio weights for each rolling window
