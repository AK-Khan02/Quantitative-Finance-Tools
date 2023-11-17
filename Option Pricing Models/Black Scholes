import math
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price for a call or put option.
    
    Parameters:
    S (float): Spot price of the underlying asset
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    option_type (str): Type of the option - 'call' or 'put'
    
    Returns:
    float: Black-Scholes option price
    """
    # Calculating d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Calculate call or put option price
    if option_type == 'call':
        option_price = (S * norm.cdf(d1, 0, 1) - K * math.exp(-r * T) * norm.cdf(d2, 0, 1))
    elif option_type == 'put':
        option_price = (K * math.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

# Example usage
S = 100  # Spot price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)

call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

print(f"Call Option Price: ", {call_price})
print(f"Put Option Price: ",{put_price})
