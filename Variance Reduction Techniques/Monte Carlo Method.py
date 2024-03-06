import numpy as np

def estimate_pi_stratified(num_strata_per_side, num_samples_per_stratum):
    """
    Estimate the value of Pi using stratified sampling.
    
    Parameters:
        num_strata_per_side (int): The number of strata along one side of the unit square.
        num_samples_per_stratum (int): The number of samples to draw from each stratum.
    
    Returns:
        pi_estimate (float): The estimated value of Pi.
    """
    total_inside = 0
    stratum_side_length = 1 / num_strata_per_side

    # Loop over each stratum
    for i in range(num_strata_per_side):
        for j in range(num_strata_per_side):
            # Determine the (x, y) coordinates of the lower-left corner of the stratum
            x_min, y_min = i * stratum_side_length, j * stratum_side_length

            # Generate random points within this stratum
            x_rand = x_min + np.random.rand(num_samples_per_stratum) * stratum_side_length
            y_rand = y_min + np.random.rand(num_samples_per_stratum) * stratum_side_length

            # Check if the points are inside the quarter circle
            inside = x_rand**2 + y_rand**2 <= 1

            # Count how many points fell inside the quarter circle
            total_inside += np.sum(inside)

    # Calculate the estimate of Pi
    total_samples = num_strata_per_side**2 * num_samples_per_stratum
    pi_estimate = 4 * total_inside / total_samples

    return pi_estimate

# Parameters
num_strata_per_side = 10  # 10 strata along each side of the unit square
num_samples_per_stratum = 100  # 100 samples per stratum

# Estimate Pi
pi_estimate = estimate_pi_stratified(num_strata_per_side, num_samples_per_stratum)
print(f"Estimated value of Pi using stratified sampling: {pi_estimate}")
