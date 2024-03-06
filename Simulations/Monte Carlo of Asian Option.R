asian_option_MC <- function(s0 = 100, K = 100, r = 0.02, t_i = c(15, 30, 45, 60),
                                      vol = 0.2, callOrPut = "call", n = 10^5) {
  # Convert time points to annualized form (assuming 252 trading days per year)
  t_i <- t_i / 252
  
  # Pre-calculate constants for efficiency
  dt <- c(t_i[1], diff(t_i))
  drift <- (r - vol^2 / 2) * dt
  diffusion <- vol * sqrt(dt)
  
  # Initialize stock price paths
  sT <- numeric(n)
  sum_sT <- numeric(n)
  
  # Simulate price paths and accumulate
  for (i in seq_along(dt)) {
    sT <- if (i == 1) s0 * exp(drift[i] + diffusion[i] * rnorm(n)) else sT * exp(drift[i] + diffusion[i] * rnorm(n))
    sum_sT <- sum_sT + sT
  }
  
  # Calculate average price and option payoff
  avg_sT <- sum_sT / length(t_i)
  payoff <- if (callOrPut == "call") pmax(avg_sT - K, 0) else pmax(K - avg_sT, 0)
  
  # Discount payoff back to present value
  discounted_payoff <- payoff * exp(-r * t_i[length(t_i)])
  
  # Calculate statistics
  mean_price <- mean(discounted_payoff)
  SE_price <- sd(discounted_payoff) / sqrt(n)
  CI_width <- 1.96 * SE_price
  lower_price <- mean_price - CI_width
  upper_price <- mean_price + CI_width
  
  return(c(Price = mean_price, SE = SE_price, Lower = lower_price, Upper = upper_price))
}

# Test the optimized function
asian_option_MC()
