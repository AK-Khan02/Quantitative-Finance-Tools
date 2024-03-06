import numpy as np
from scipy.stats import norm

# Defining necessary functions

def black_scholes(S, K, T, r, sigma):
    """Black-Scholes formula for European call option price."""
    if T <= 0:  # At or after maturity
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def delta(S, K, T, r, sigma):
    """Delta of a European call option."""
    if T <= 0:  # At or after maturity
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    """Gamma of a European call option."""
    if T <= 0:  # At or after maturity
        return 0
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def simulate_portfolio_path(S, K, T, r, sigma, mu, N):
    dt = T / N
    portfolio_values = np.zeros(N+1)
    S_t = S  # Initial stock price
    
    # Initial delta and gamma hedging
    curr_delta = delta(S_t, K, T, r, sigma)
    curr_gamma = gamma(S_t, K, T, r, sigma)
    curr_cash = black_scholes(S_t, K, T, r, sigma) - curr_delta * S_t - 0.5 * curr_gamma * S_t**2
    portfolio_values[0] = curr_cash + curr_delta * S_t + 0.5 * curr_gamma * S_t**2
    
    for j in range(1, N):
        # Simulate stock price movement
        Z = np.random.normal()
        S_next = S_t * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Update delta and gamma
        next_delta = delta(S_next, K, T - j*dt, r, sigma)
        next_gamma = gamma(S_next, K, T - j*dt, r, sigma)
        
        # Update cash position considering delta and gamma adjustments
        curr_cash = (curr_cash + (curr_delta - next_delta) * S_next +
                     0.5 * (curr_gamma - next_gamma) * S_next**2) * np.exp(r * dt)
        
        # Update portfolio value
        portfolio_values[j] = curr_cash + next_delta * S_next + 0.5 * next_gamma * S_next**2 - \
                              black_scholes(S_next, K, T - j*dt, r, sigma)
        
        # Update current values for the next iteration
        S_t = S_next
        curr_delta = next_delta
        curr_gamma = next_gamma

    # Final portfolio value at maturity
    portfolio_values[-1] = curr_cash + curr_delta * S_t + 0.5 * curr_gamma * S_t**2 - max(S_t - K, 0)
    return portfolio_values

# Parameters
S = 100  # Initial stock price
K = 105  # Strike price
T = 1    # Time to maturity in years
r = 0.05 # Risk-free rate
sigma = 0.2  # Volatility
mu = 0.15   # Expected return
N = 10      # Number of steps

# Simulate one portfolio path
portfolio_path = simulate_portfolio_path(S, K, T, r, sigma, mu, N)

portfolio_path
