import numpy as np
from scipy.stats import norm

# Helper functions for Black-Scholes parameters
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Rho of a call option
def rho_call(S, K, T, r, sigma):
    d2_val = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2_val)

# Portfolio of options
class RhoHedgedPortfolio:
    def __init__(self, options):
        self.options = options

    def portfolio_rho(self, S, r):
        total_rho = sum([rho_call(S, opt['K'], opt['T'], r, opt['sigma']) for opt in self.options])
        return total_rho

    def adjust_for_rho(self, S, r):
        current_rho = self.portfolio_rho(S, r)
        print(f"Current Portfolio Rho: {current_rho}")
        # Implement rho hedging logic here, possibly by adjusting option positions or adding interest rate sensitive instruments

# Example usage
options = [
    {'K': 100, 'T': 1, 'sigma': 0.2},  # Option 1
    {'K': 105, 'T': 0.5, 'sigma': 0.25}  # Option 2
]
portfolio = RhoHedgedPortfolio(options)
S = 100  # Current stock price
r = 0.05  # Risk-free rate

portfolio.adjust_for_rho(S, r)
