import numpy as np
import statsmodels.api as sm

# Example data
asset_returns = np.array([0.12, 0.15, 0.18, 0.10])  # Asset returns
factor_data = np.array([[0.05, 0.04],
                        [0.03, 0.02],
                        [0.07, 0.05],
                        [0.02, 0.03]])  # Factor returns (e.g., market and size factors)

# Perform factor model regression
X = sm.add_constant(factor_data)
model = sm.OLS(asset_returns, X)
results = model.fit()
factor_loadings = results.params[1:]  # Estimated factor loadings
alpha = results.params[0]  # Alpha (asset-specific return)

# Define target factor exposures (e.g., equal exposure to each factor)
target_factor_exposures = np.ones(len(factor_loadings)) / len(factor_loadings)

# Define the objective function for factor model portfolio optimization
def factor_model_objective(weights, factor_loadings, alpha, target_factor_exposures):
    expected_asset_returns = alpha + np.dot(weights, factor_loadings)
    factor_exposure_diff = weights @ factor_loadings - target_factor_exposures
    return np.var(expected_asset_returns) + np.sum(factor_exposure_diff ** 2)

# Initial guess for portfolio weights
initial_weights = np.ones(len(factor_loadings)) / len(factor_loadings)

# Define the constraint that weights should sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Set bounds for portfolio weights (0 <= weight <= 1)
bounds = tuple((0, 1) for asset in range(len(factor_loadings)))

# Minimize the factor model objective function
result = minimize(factor_model_objective, initial_weights, args=(factor_loadings, alpha, target_factor_exposures),
                  method='SLSQP', constraints=constraints, bounds=bounds)

# Extract the optimized portfolio weights
optimal_weights = result.x
print("Optimal Factor Model Portfolio Weights:", optimal_weights)
