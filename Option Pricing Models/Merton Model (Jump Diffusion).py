import numpy as np
import scipy.stats as si

def merton_jump_diffusion(S, K, T, r, sigma, lambda_, mu_j, sigma_j, simulations):
    """
    Monte Carlo simulation using the Merton Jump Diffusion Model for option pricing.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock
    lambda_ (float): Average number of jumps per year
    mu_j (float): Mean jump size
    sigma_j (float): Standard deviation of jump size
    simulations (int): Number of simulations

    Returns:
    float: Estimated Call Option Price
    """
    np.random.seed(0)  # For reproducible results
    dt = T / simulations
    jump_num = np.random.poisson(lambda_ * dt, simulations)
    jump_size = np.random.normal(mu_j, sigma_j, simulations)

    # Simulate stock price paths with jumps
    prices = np.zeros(simulations)
    prices[0] = S

    for i in range(1, simulations):
        Z = np.random.standard_normal()
        prices[i] = prices[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        # Apply jumps
        for j in range(jump_num[i]):
            prices[i] = prices[i] * np.exp(jump_size[j])

    payoff = np.maximum(prices - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)
lambda_ = 0.1 # Average number of jumps per year
mu_j = -0.05 # Mean jump size
sigma_j = 0.1 # Standard deviation of jump size
simulations = 100000 # Number of simulations

call_option_price = merton_jump_diffusion(S, K, T, r, sigma, lambda_, mu_j, sigma_j, simulations)
print(f"Call Option Price: {call_option_price}")
