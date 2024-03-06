asian_option_MC_cvST <- function(s0 = 100, K = 100, r = 0.02, t_i = c(15, 30, 45, 60),
                                           vol = 0.2, callOrPut = "call", n = 10^5) {
  # Convert time points to annualized form (assuming 252 trading days per year)
  t_i <- t_i / 252
  
  # Pre-calculate constants for efficiency
  dt <- c(t_i[1], diff(t_i))
  drift <- (r - vol^2 / 2) * dt
  diffusion <- vol * sqrt(dt)
  
  # Initialize and simulate stock price paths
  sT <- s0 * exp(drift[1] + diffusion[1] * rnorm(n))
  sum_sT <- sT
  
  # Accumulate price paths and sum for Asian option average
  for (i in 2:length(dt)) {
    sT <- sT * exp(drift[i] + diffusion[i] * rnorm(n))
    sum_sT <- sum_sT + sT
  }
  
  # Calculate option payoff based on the average price
  avg_sT <- sum_sT / length(t_i)
  payoff <- if (callOrPut == "call") pmax(avg_sT - K, 0) else pmax(K - avg_sT, 0)
  discounted_payoff <- payoff * exp(-r * t_i[length(t_i)])
  
  # Apply control variate technique
  theta_star <- cov(discounted_payoff, sT) / var(sT)
  corrected_payoff <- discounted_payoff - theta_star * (sT - s0 * exp(r * t_i[length(t_i)]))
  
  # Compute statistics
  mean_price <- mean(corrected_payoff)
  SE_price <- sd(corrected_payoff) / sqrt(n)
  CI_width <- 1.96 * SE_price
  lower_price <- mean_price - CI_width
  upper_price <- mean_price + CI_width
  
  return(c(Price = mean_price, SE = SE_price, Lower = lower_price, Upper = upper_price))
}

# Test the optimized function
asian_option_MC_cvST()
