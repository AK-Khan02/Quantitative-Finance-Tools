import numpy as np
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt

# Black-Scholes formula for call option price
def bsformula(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Vega of a call option
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

# Simulate changes in volatility and adjust vega hedge
def simulate_vega_hedge(S, K, T, r, initial_sigma, delta_sigma, steps):
    sigmas = np.linspace(initial_sigma - delta_sigma, initial_sigma + delta_sigma, steps)
    vegas = [vega(S, K, T, r, sigma) for sigma in sigmas]
    option_prices = [bsformula(S, K, T, r, sigma) for sigma in sigmas]

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(sigmas, vegas, label='Vega')
    plt.title('Vega vs. Volatility')
    plt.xlabel('Volatility (sigma)')
    plt.ylabel('Vega')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sigmas, option_prices, label='Option Price')
    plt.title('Option Price vs. Volatility')
    plt.xlabel('Volatility (sigma)')
    plt.ylabel('Option Price')
    plt.grid(True)

    plt.show()

# Parameters
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity in years
r = 0.05  # Risk-free rate
initial_sigma = 0.2  # Initial volatility
delta_sigma = 0.1  # Range of volatility change to simulate
steps = 50  # Number of steps in the simulation

simulate_vega_hedge(S, K, T, r, initial_sigma, delta_sigma, steps)
