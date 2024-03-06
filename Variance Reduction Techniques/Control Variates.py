import numpy as np
import scipy.stats as stats

def black_scholes_call_price(S, K, T, r, sigma):
    """Calculate the Black-Scholes theoretical price for a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

def european_call_with_control_variates(S0, K, T, r, sigma, n, K_cv):
    """Price a European call option using control variates."""
    # Time parameters
    dt = T
    discount_factor = np.exp(-r * T)

    # Simulate end-of-period prices
    Z = np.random.normal(0, 1, n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Compute payoffs
    payoff_option = np.maximum(ST - K, 0)
    payoff_cv = np.maximum(ST - K_cv, 0)

    # Theoretical price of the control variate option
    cv_theoretical_price = black_scholes_call_price(S0, K_cv, T, r, sigma)

    # Adjust payoffs using the control variate
    adjusted_payoff = payoff_option + cv_theoretical_price - payoff_cv

    # Estimate the option price and reduce variance
    option_price_estimate = discount_factor * np.mean(adjusted_payoff)

    return option_price_estimate

# Parameters
S0 = 100  # Initial stock price
K = 105  # Strike price of the option of interest
K_cv = 110  # Strike price of the control variate option
T = 1  # Time to expiration in years
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
n = 10000  # Number of simulations

# Price the option using control variates
option_price = european_call_with_control_variates(S0, K, T, r, sigma, n, K_cv)
print("Estimated European Call Option Price with Control Variates:", option_price)

