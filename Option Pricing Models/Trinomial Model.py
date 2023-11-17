import numpy as np

def trinomial_tree_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculate the option price using the Trinomial Tree Model.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock
    N (int): Number of time steps
    option_type (str): Type of the option - 'call' or 'put'

    Returns:
    float: Option price
    """
    # Time step
    dt = T / N
    # Up, down, and unchanged factors
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    m = 1

    # Risk-neutral probabilities
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) / 
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) / 
          (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    pm = 1 - pu - pd

    # Initialize asset prices at maturity
    asset_prices = np.zeros((2 * N + 1, N + 1))
    asset_prices[0, 0] = S

    for j in range(1, N + 1):
        for i in range(2 * j + 1):
            if i < j:
                asset_prices[i, j] = asset_prices[i, j - 1] * u
            elif i == j:
                asset_prices[i, j] = asset_prices[i, j - 1]
            else:
                asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

    # Initialize option values at maturity
    option_values = np.zeros((2 * N + 1, N + 1))
    for i in range(2 * N + 1):
        if option_type == "call":
            option_values[i, N] = max(asset_prices[i, N] - K, 0)
        elif option_type == "put":
            option_values[i, N] = max(K - asset_prices[i, N], 0)

    # Backward induction for option pricing
    for j in range(N - 1, -1, -1):
        for i in range(2 * j + 1):
            option_values[i, j] = (pu * option_values[i, j + 1] +
                                   pm * option_values[i + 1, j + 1] +
                                   pd * option_values[i + 2, j + 1]) * np.exp(-r * dt)

    return option_values[0, 0]

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)
N = 50   # Number of time steps

call_price = trinomial_tree_option_pricing(S, K, T, r, sigma, N, 'call')
put_price = trinomial_tree_option_pricing(S, K, T, r, sigma, N, 'put')

print(f"Call Option Price: {call_price}")
print(f"Put Option Price: {put_price}")
