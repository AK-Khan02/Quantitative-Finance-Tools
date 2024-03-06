import numpy as np
from scipy.stats import norm

# Helper functions for Black-Scholes parameters
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Theta of a call option
def theta_call(S, K, T, r, sigma):
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return (-S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T) * norm.cdf(d2_val))

# Portfolio of options
class ThetaHedgedPortfolio:
    def __init__(self, options):
        self.options = options

    def portfolio_theta(self, S, r):
        total_theta = sum([theta_call(S, opt['K'], opt['T'], r, opt['sigma']) for opt in self.options])
        return total_theta

    def adjust_for_theta(self, S, r):
        current_theta = self.portfolio_theta(S, r)
        print(f"Current Portfolio Theta: {current_theta}")
        # Implement theta hedging logic here, possibly by adjusting option positions or adding time-sensitive instruments

# Example usage
options = [
    {'K': 100, 'T': 1, 'sigma': 0.2},  # Option 1
    {'K': 105, 'T': 0.5, 'sigma': 0.25}  # Option 2
]
portfolio = ThetaHedgedPortfolio(options)
S = 100  # Current stock price
r = 0.05  # Risk-free rate

portfolio.adjust_for_theta(S, r)
