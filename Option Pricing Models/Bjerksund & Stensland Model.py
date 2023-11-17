import numpy as np
import scipy.stats as si

def bjerksund_stensland_1993(S, K, T, r, sigma):
    """
    Bjerksund and Stensland (1993) approximation for American call options.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock

    Returns:
    float: American Call Option Price
    """
    if S >= K:
        return S - K

    alpha = (0.5 - r / sigma**2) + np.sqrt((r / sigma**2 - 0.5)**2 + 2 * r / sigma**2)
    beta = (np.exp(r * T) - 1) / (alpha * (np.exp(alpha * r * T) - 1))
    BInfinity = beta * K
    B0 = max(K, r / (r - 1) * K)
    h1 = -(r * T + 2 * sigma * np.sqrt(T)) * B0 / (BInfinity - B0)
    I = B0 + (BInfinity - B0) * (1 - np.exp(h1))

    if S < I:
        return B0 - S + (S - B0)**2 / (BInfinity - B0) * np.exp(h1)
    else:
        return S - K

# Example usage
S = 90    # Current stock price
K = 100   # Strike price
T = 1     # Time to maturity (1 year)
r = 0.05  # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

american_call_price = bjerksund_stensland_1993(S, K, T, r, sigma)
print(f"American Call Option Price: {american_call_price}")
