import numpy as np
import matplotlib.pyplot as plt

def brownian_bridge(T, N):
    """
    Constructs a Brownian Bridge from 0 to T with N intermediate points.
    
    Parameters:
        T (float): The final time of the Brownian Bridge.
        N (int): The number of intermediate points to generate.
        
    Returns:
        times (numpy.ndarray): Array of time points.
        path (numpy.ndarray): The constructed Brownian Bridge path.
    """
    # Initialize the time and path arrays
    times = np.linspace(0, T, N+2)  # Include start and end times
    path = np.zeros(N+2)  # Start and end points are 0
    
    # Recursive function to fill in the path
    def fill_path(start, end, level):
        if end - start > 1:  # More than one point between start and end
            mid = (start + end) // 2
            var = (times[mid] - times[start]) * (times[end] - times[mid]) / (times[end] - times[start])
            path[mid] = (path[start] + path[end]) / 2 + np.random.normal(0, np.sqrt(var))
            fill_path(start, mid, level+1)
            fill_path(mid, end, level+1)

    # Start filling in the path
    fill_path(0, N+1, 1)
    
    return times, path

# Parameters
T = 1  # Total time
N = 2**10 - 1  # Number of intermediate points (2^k - 1 for some integer k)

# Generate the Brownian Bridge
times, path = brownian_bridge(T, N)

# Plot the Brownian Bridge
plt.figure(figsize=(10, 6))
plt.plot(times, path)
plt.title("Brownian Bridge")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

