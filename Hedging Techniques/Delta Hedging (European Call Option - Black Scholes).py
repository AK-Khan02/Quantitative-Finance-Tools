import numpy as np
import matplotlib.pyplot as plt

# Black-Scholes formula for call option price
def bsformula(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)

# Delta of a call option
def deltaformula(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + sig**2 / 2) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1)

# Generate one path of portfolio values
def gen_one_valuepath(r, mu, sig, K, S0, T, N):
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    delta = np.zeros(N + 1)
    delta[0] = deltaformula(S0, K, T, r, sig)
    cash = np.zeros(N + 1)
    cash[0] = bsformula(S0, K, T, r, sig) - delta[0] * S0

    for j in range(1, N + 1):
        S[j] = S[j - 1] * np.exp((mu - sig**2 / 2) * dt + sig * np.sqrt(dt) * np.random.normal())
        delta[j] = deltaformula(S[j], K, T - j * dt, r, sig)
        cash[j] = cash[j - 1] * np.exp(r * dt) + (delta[j - 1] - delta[j]) * S[j]
    
    # Portfolio value excluding the final option payoff
    portfolval = S * delta + cash
    
    # Adjust the final value for the option payoff
    portfolval[-1] -= max(S[-1] - K, 0)
    
    return portfolval

# Parameters
np.random.seed(1)
n = 10000
r = 0.05
mu = 0.15
sig = 0.1
K = 105
S0 = 100
T = 1
N = 10  # Trading times

# Run simulations
out = np.array([gen_one_valuepath(r, mu, sig, K, S0, T, N) for _ in range(n)])

# Plot 25 paths
for i in range(min(25, n)):
    plt.plot(out[i], label=f'Path {i+1}')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.title('Simulated Portfolio Value Paths')
plt.legend()
plt.show()

# Histogram of final P&L
plt.hist(out[:, -1], bins=30)
plt.axvline(out[:, -1].mean(), color='r', linestyle='dashed', linewidth=1)
plt.title('Histogram of P&L')
plt.xlabel('P&L')
plt.ylabel('Frequency')
plt.show()

# Risk measures
sortedval = np.sort(out[:, -1])
VaR = sortedval[int(0.05 * n)]  # Value-at-Risk at 95%
ES = np.mean(sortedval[:
