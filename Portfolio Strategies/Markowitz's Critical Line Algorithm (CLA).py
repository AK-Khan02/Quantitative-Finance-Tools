import numpy as np
import cvxpy as cp

# Example data
mean_returns = np.array([0.12, 0.15, 0.18, 0.10])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.03],
                       [0.02, 0.06, 0.03, 0.04],
                       [0.01, 0.03, 0.05, 0.02],
                       [0.03, 0.04, 0.02, 0.07]])

# Number of assets
n = len(mean_returns)

# Define variables for portfolio weights
weights = cp.Variable(n)

# Define the target return
target_return = cp.Parameter()

# Portfolio returns and volatility
portfolio_return = mean_returns @ weights
portfolio_volatility = cp.norm(weights.T @ cov_matrix @ weights, 'fro')

# Define the objective function (minimize volatility)
objective = cp.Minimize(portfolio_volatility)

# Define the constraints
constraints = [
    cp.sum(weights) == 1,  # Weight sum constraint
    portfolio_return == target_return,  # Target return constraint
    weights >= 0  # Non-negative weights
]

# Create and solve the optimization problem
problem = cp.Problem(objective, constraints)

# CLA: Find efficient frontier by varying target return
num_points = 50  # Number of points on the efficient frontier
efficient_frontier = []

for i in range(num_points):
    target_return.value = i / num_points * max(mean_returns)
    problem.solve()
    
    efficient_frontier.append((target_return.value, portfolio_volatility.value))

efficient_frontier = np.array(efficient_frontier)

# Print the efficient frontier points (return, volatility)
print("Efficient Frontier:")
print(efficient_frontier)
