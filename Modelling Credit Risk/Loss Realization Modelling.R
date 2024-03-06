simulate_portfolio_loss <- function(num_obligors, prob_default, exposure_at_default, loss_given_default, correlation) {
    # Initialize total portfolio loss
    total_loss = 0
    
    # Draw a single realization of the systematic factor from a standard normal distribution
    systematic_factor = rnorm(1)
    
    # Vectorize the simulation of idiosyncratic factors and asset values
    idiosyncratic_factors = rnorm(num_obligors)
    asset_values = sqrt(correlation) * systematic_factor + sqrt(1 - correlation) * idiosyncratic_factors
    
    # Determine the default threshold for each obligor based on their probability of default
    default_thresholds = qnorm(prob_default)
    
    # Identify defaults where asset value falls below the default threshold
    defaults = asset_values < default_thresholds
    
    # Calculate losses for defaulted obligors and sum to get total portfolio loss
    total_loss = sum(exposure_at_default[defaults] * loss_given_default[defaults])
    
    return(total_loss)
}

# Simulation parameters
num_obligors = 100  # Number of obligors in the portfolio
num_simulations = 5000  # Number of loss realizations to simulate
exposure_at_default = rep(1, num_obligors)  # Exposure at default for each obligor
loss_given_default = rep(1, num_obligors)  # Loss given default for each obligor
prob_default = rep(0.25, num_obligors)  # Probability of default for each obligor
correlation = 0.2  # Correlation among obligors' defaults

# Run simulations and collect losses
simulated_losses = replicate(num_simulations, simulate_portfolio_loss(num_obligors, prob_default, exposure_at_default, loss_given_default, correlation))

# Plot the histogram of simulated total losses
hist(simulated_losses, freq = FALSE, main = "Histogram of Portfolio Losses", xlab = "Total Loss", ylab = "Density")
