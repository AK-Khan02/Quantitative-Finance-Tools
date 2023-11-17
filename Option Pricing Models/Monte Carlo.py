import numpy as np

def monte_carlo_call_option(S, K, T, r, sigma, simulations):
    """
    Monte Carlo simulation for European Call Option Pricing.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock
    simulations (int): Number of simulations

    Returns:
    float: Estimated Call Option Price
    """
    np.random.seed(0)  # For reproducible results
    Z = np.random.standard_normal(simulations)  # Random standard normals
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # Stock price at maturity
    payoff = np.maximum(ST - K, 0)  # Payoff for Call Option
    option_price = np.exp(-r * T) * np.mean(payoff)  # Discounted average payoff
    return option_price

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)
simulations = 1000000  # Number of simulations

call_price = monte_carlo_call_option(S, K, T, r, sigma, simulations)
print(f"Call Option Price: {call_price}")
