calculate_var_es <- function(returns, confidence_level=0.95) {
  mean_return <- mean(returns)
  std_return <- sd(returns)

  var <- qnorm(1 - confidence_level, mean_return, std_return)
  es <- mean_return - std_return * dnorm(qnorm(1 - confidence_level)) / (1 - confidence_level)

  return(c(var, es))
}

# Example usage
set.seed(123)  # For reproducibility
historical_returns <- rnorm(1000, 0, 0.01)  # Simulated daily returns
confidence_level <- 0.95

var_es <- calculate_var_es(historical_returns, confidence_level)
cat("Value at Risk (VaR) at", confidence_level * 100, "% confidence level:", var_es[1], "\n")
cat("Expected Shortfall (ES) at", confidence_level * 100, "% confidence level:", var_es[2], "\n")
