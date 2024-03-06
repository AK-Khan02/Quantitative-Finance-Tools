price_asian_option_with_cv <- function(initial_price = 100, strike_price = 100, risk_free_rate = 0.02, 
                                       time_points = c(15, 30, 45, 60), volatility = 0.2, option_type = "call", 
                                       num_simulations = 10^5) {
    # Convert time points to years (assuming 252 trading days per year)
    annualized_times <- time_points / 252
    final_time <- tail(annualized_times, 1)  # Maturity of the option
    
    # Simulate initial stock price
    stock_prices <- initial_price * exp((risk_free_rate - volatility^2 / 2) * annualized_times[1] + 
                                        volatility * sqrt(annualized_times[1]) * rnorm(num_simulations))
    sum_stock_prices <- stock_prices
    
    # Simulate stock price paths and accumulate for average
    for (i in 2:length(annualized_times)) {
        delta_time <- annualized_times[i] - annualized_times[i - 1]
        stock_prices <- stock_prices * exp((risk_free_rate - volatility^2 / 2) * delta_time + 
                                           volatility * sqrt(delta_time) * rnorm(num_simulations))
        sum_stock_prices <- sum_stock_prices + stock_prices
    }
    
    # Calculate the Asian option price
    average_price <- sum_stock_prices / length(annualized_times)
    if (option_type == "call") {
        asian_option_payoff <- pmax(average_price - strike_price, 0)
    } else {
        asian_option_payoff <- pmax(strike_price - average_price, 0)
    }
    discounted_payoff <- asian_option_payoff * exp(-risk_free_rate * final_time)
    
    # Calculate European option payoff for control variate
    european_option_payoff <- if (option_type == "call") {
        pmax(stock_prices - strike_price, 0)
    } else {
        pmax(strike_price - stock_prices, 0)
    } * exp(-risk_free_rate * final_time)
    
    # Calculate Black-Scholes price for the European option
    bs_price <- black_scholes_eopt(s0 = initial_price, K = strike_price, r = risk_free_rate, 
                                   T_in_days = final_time * 252, sig = volatility, callOrPut = option_type)
    
    # Apply control variate technique
    theta_star <- cov(discounted_payoff, european_option_payoff) / var(european_option_payoff)
    corrected_payoff <- discounted_payoff - theta_star * (european_option_payoff - bs_price)
    
    # Compute statistics for the Asian option price
    mean_price <- mean(corrected_payoff)
    standard_error <- sd(corrected_payoff) / sqrt(num_simulations)
    confidence_interval_width <- 1.96 * standard_error
    lower_bound <- mean_price - confidence_interval_width
    upper_bound <- mean_price + confidence_interval_width
    
    return(c(Price = mean_price, SE = standard_error, Lower = lower_bound, Upper = upper_bound))
}

# Test the optimized and annotated function
price_asian_option_with_cv()
