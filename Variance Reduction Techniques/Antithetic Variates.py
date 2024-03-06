import numpy as np

def european_call_price_antithetic_variates(S0, K, T, r, sigma, n):
    # Time-adjusted parameters
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    # Step 1: Generate n/2 standard normal random variables
    Z = np.random.normal(0, 1, int(n / 2))

    # Step 2: Calculate S_T for each Z
    ST = S0 * np.exp(drift + vol * Z)

    # Step 3: Calculate antithetic S_T using -Z
    ST_antithetic = S0 * np.exp(drift - vol * Z)

    # Step 4: Compute the payoff for each path
    payoff = np.maximum(ST - K, 0)
    payoff_antithetic = np.maximum(ST_antithetic - K, 0)

    # Combine the original and antithetic payoffs
    all_payoffs = np.concatenate((payoff, payoff_antithetic))

    # Step 5: Average the payoffs and discount back to present value
    option_price = np.exp(-r * T) * np.mean(all_payoffs)

    return option_price

# Parameters
S0 = 100  # Initial stock price
K = 100  # Strike price
T = 1  # Time to expiration in years
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
n = 10000  # Number of simulations

# Price the option
option_price = european_call_price_antithetic_variates(S0, K, T, r, sigma, n)
print("Estimated European Call Option Price:", option_price)


