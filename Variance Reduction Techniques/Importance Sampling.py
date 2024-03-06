import numpy as np
import scipy.stats as stats

def importance_sampling(mu, sigma, alpha, n, mu_prime):
    """
    Estimate the probability of a rare event using importance sampling.

    Parameters:
        mu (float): Mean of the original distribution.
        sigma (float): Standard deviation of the original distribution.
        alpha (float): Loss threshold.
        n (int): Number of samples.
        mu_prime (float): Mean of the alternative distribution.

    Returns:
        probability_estimate (float): Estimated probability of the rare event.
    """
    # Generate samples from the alternative distribution Q
    samples = np.random.normal(mu_prime, sigma, n)

    # Compute the weights of the samples
    weights = stats.norm.pdf(samples, mu, sigma) / stats.norm.pdf(samples, mu_prime, sigma)

    # Indicator function for the rare event
    indicator = samples < -alpha

    # Estimate the probability of the rare event
    probability_estimate = np.mean(weights * indicator)

    return probability_estimate

# Parameters
mu = 0  # Mean of the original distribution
sigma = 1  # Standard deviation of the original distribution
alpha = 3  # Loss threshold (3 standard deviations)
n = 10000  # Number of samples
mu_prime = -1.5  # Mean of the alternative distribution

# Estimate the probability of the rare event
probability_estimate = importance_sampling(mu, sigma, alpha, n, mu_prime)
print("Estimated Probability of the Rare Event:", probability_estimate)

