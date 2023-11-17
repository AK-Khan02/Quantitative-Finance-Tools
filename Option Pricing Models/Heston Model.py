import numpy as np

def heston_model(S0, K, T, r, V0, kappa, theta, xi, rho, simulations, time_steps):
    """
    Monte Carlo simulation using the Heston model for option pricing.

    Parameters:
    S0 (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    V0 (float): Initial variance
    kappa (float): Rate of reversion
    theta (float): Long-run variance
    xi (float): Volatility of the volatility
    rho (float): Correlation coefficient between asset and variance
    simulations (int): Number of simulations
    time_steps (int): Number of time steps

    Returns:
    float: Estimated option price
    """
    dt = T / time_steps
    prices = np.zeros((time_steps + 1, simulations))
    variances = np.zeros((time_steps + 1, simulations))

    prices[0] = S0
    variances[0] = V0

    for t in range(1, time_steps + 1):
        Z1 = np.random.standard_normal(simulations)
        Z2 = np.random.standard_normal(simulations)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        variances[t] = np.maximum(variances[t - 1] + kappa * (theta - np.maximum(variances[t - 1], 0)) * dt + xi * np.sqrt(np.maximum(variances[t - 1], 0)) * np.sqrt(dt) * Z2, 0)
        prices[t] = prices[t - 1] * np.exp((r - 0.5 * variances[t]) * dt + np.sqrt(variances[t]) * np.sqrt(dt) * Z1)

    payoffs = np.maximum(prices[-1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

# Example usage
S0 = 100  # Current stock price
K = 100   # Strike price
T = 1     # Time to maturity (1 year)
r = 0.05  # Risk-free rate
V0 = 0.04 # Initial variance
kappa = 3.0 # Rate of reversion
theta = 0.04 # Long-run variance
xi = 0.1  # Volatility of volatility
rho = -0.7 # Correlation between asset and variance
simulations = 10000 # Number of simulations
time_steps = 100 # Number of time steps

option_price = heston_model(S0, K, T, r, V0, kappa, theta, xi, rho, simulations, time_steps)
print(f"Option Price: {option_price}")
