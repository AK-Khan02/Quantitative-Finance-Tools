import numpy as np
import cvxpy as cp

# Example data
mean_returns = np.array([0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])

# Define the decision variables
weights = cp.Variable(len(mean_returns))

# Define the objective function (minimize portfolio risk)
portfolio_risk = cp.quad_form(weights, cov_matrix)
objective = cp.Minimize(portfolio_risk)

# Define the constraints
constraints = [
    cp.sum(weights) == 1,  # Weight sum constraint
    weights >= 0  # Non-negative weights
]

# Create and solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract the optimized portfolio weights
optimal_weights = weights.value
print("Optimal Robust Portfolio Weights:", optimal_weights)
