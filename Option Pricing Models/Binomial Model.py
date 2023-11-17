import math

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculate the option price using the Binomial Option Pricing Model.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock
    N (int): Number of steps in the binomial tree
    option_type (str): Type of the option - 'call' or 'put'
    
    Returns:
    float: Option price
    """
    # Time interval
    dt = T / N

    # Up and down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    p = (math.exp(r * dt) - d) / (u - d)

    # Initializing asset prices at maturity
    asset_prices = [0] * (N + 1)
    for i in range(N + 1):
        asset_prices[i] = S * (u ** (N - i)) * (d ** i)

    # Initializing option values at maturity
    option_values = [0] * (N + 1)
    for i in range(N + 1):
        if option_type == "call":
            option_values[i] = max(asset_prices[i] - K, 0)
        elif option_type == "put":
            option_values[i] = max(K - asset_prices[i], 0)

    # Recursive step for the binomial tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = (p * option_values[j] + (1 - p) * option_values[j + 1]) * math.exp(-r * dt)

    return option_values[0]

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)
N = 50   # Number of steps in the binomial tree

call_price = binomial_option_pricing(S, K, T, r, sigma, N, 'call')
put_price = binomial_option_pricing(S, K, T, r, sigma, N, 'put')

print(f"Call Option Price: ", call_price)
print(f"Put Option Price: ", put_price)
