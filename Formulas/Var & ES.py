import numpy as np
import scipy.stats as stats

def calculate_var_es(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a given series of returns.

    Parameters:
    returns (list or numpy array): Array of historical returns.
    confidence_level (float): Confidence level for VaR and ES calculation.

    Returns:
    float, float: VaR and ES
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    var = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
    es = mean_return - std_return * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)

    return var, es

# Example usage
historical_returns = np.random.normal(0, 0.01, 1000)  # Simulated daily returns
confidence_level = 0.95

var, es = calculate_var_es(historical_returns, confidence_level)
print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level: {var}")
print(f"Expected Shortfall (ES) at {confidence_level * 100}% confidence level: {es}")
