import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Updated Black-Scholes functions
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T)

# Gamma of a call option
def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))

# Portfolio of options
class OptionPortfolio:
    def __init__(self, options):
        self.options = options
    
    def portfolio_vega(self, S, r):
        return sum([vega(S, opt['K'], opt['T'], r, opt['sigma']) for opt in self.options])
    
    def portfolio_gamma(self, S, r):
        return sum([gamma(S, opt['K'], opt['T'], r, opt['sigma']) for opt in self.options])
    
    def rebalance_hedges(self, S, r, target_vega, target_gamma):
        # Placeholder for a more complex rebalancing strategy
        current_vega = self.portfolio_vega(S, r)
        current_gamma = self.portfolio_gamma(S, r)
        print(f"Current Portfolio Vega: {current_vega}, Target Vega: {target_vega}")
        print(f"Current Portfolio Gamma: {current_gamma}, Target Gamma: {target_gamma}")
        # Implement rebalancing logic here

# Example usage
options = [
    {'K': 100, 'T': 1, 'sigma': 0.2},
    {'K': 110, 'T': 1, 'sigma': 0.25},
    {'K': 90, 'T': 1, 'sigma': 0.22}
]
portfolio = OptionPortfolio(options)
S = 100  # Current stock price
r = 0.05  # Risk-free rate

# Rebalance to target vega and gamma
target_vega = 20  # Arbitrary target vega
target_gamma = 0.1  # Arbitrary target gamma
portfolio.rebalance_hedges(S, r, target_vega, target_gamma)
