monte_carlo_call_option <- function(S, K, T, r, sigma, simulations) {
  set.seed(0)  # For reproducible results
  Z <- rnorm(simulations)  # Random standard normals
  ST <- S * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)  # Stock price at maturity
  payoff <- pmax(ST - K, 0)  # Payoff for Call Option
  option_price <- exp(-r * T) * mean(payoff)  # Discounted average payoff
  return(option_price)
}

# Example usage
S <- 100  # Current stock price
K <- 100  # Strike price
T <- 1    # Time to maturity (1 year)
r <- 0.05 # Risk-free rate (5%)
sigma <- 0.2 # Volatility (20%)
simulations <- 1000000  # Number of simulations

call_price <- monte_carlo_call_option(S, K, T, r, sigma, simulations)
cat("Call Option Price:", call_price, "\n")
