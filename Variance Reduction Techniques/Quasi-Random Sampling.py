from scipy.stats.qmc import Sobol
import numpy as np

def estimate_pi_quasi_random(num_samples):
    """
    Estimate the value of Pi using quasi-random sampling with Sobol sequences.
    
    Parameters:
        num_samples (int): The number of samples to draw.
    
    Returns:
        pi_estimate (float): The estimated value of Pi.
    """
    # Initialize the Sobol generator with 2 dimensions
    sobol_gen = Sobol(d=2, scramble=False)

    # Generate quasi-random points in 2 dimensions
    points = sobol_gen.random(n=num_samples)

    # Map points from [0, 1) to [-1, 1) to simulate points in a unit square centered at the origin
    points = 2 * points - 1

    # Calculate the distance of each point from the origin
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

    # Count the number of points within the unit circle
    inside_circle = np.sum(distances <= 1)

    # Estimate Pi
    pi_estimate = 4 * inside_circle / num_samples

    return pi_estimate

# Number of samples
num_samples = 10000

# Estimate Pi using quasi-random sampling
pi_estimate = estimate_pi_quasi_random(num_samples)
print(f"Estimated value of Pi using quasi-random sampling: {pi_estimate}")

