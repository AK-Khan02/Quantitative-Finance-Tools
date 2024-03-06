import numpy as np
import matplotlib.pyplot as plt

def reduce_noise_fourier(data, cutoff_frequency):
    """
    Reduce noise in a time series using Fourier transforms.
    
    Parameters:
        data (numpy.ndarray): 1D array of time series data.
        cutoff_frequency (float): Frequency threshold for filtering. Components with frequencies higher than this will be attenuated.
        
    Returns:
        filtered_data (numpy.ndarray): 1D array of filtered time series data.
    """
    # Compute the Fourier transform of the data
    fourier_transform = np.fft.fft(data)
    frequencies = np.fft.fftfreq(data.size, d=1)  # Assuming time step d=1
    
    # Apply a low-pass filter by attenuating frequencies above the cutoff
    fourier_transform[np.abs(frequencies) > cutoff_frequency] = 0
    
    # Compute the inverse Fourier transform of the filtered data
    filtered_data = np.fft.ifft(fourier_transform)
    
    return filtered_data.real  # Return the real part

# Example usage with synthetic data
np.random.seed(0)
time_steps = np.arange(100)
original_data = np.sin(2 * np.pi * 0.05 * time_steps)  # Low-frequency signal
noise = np.random.normal(0, 0.5, size=time_steps.size)  # Synthetic noise
noisy_data = original_data + noise

# Apply noise reduction
cutoff_frequency = 0.1  # Adjust based on the frequency content of your data and desired filter strength
filtered_data = reduce_noise_fourier(noisy_data, cutoff_frequency)

# Plotting the results
plt.figure(figsize=(14, 6))
plt.plot(time_steps, noisy_data, label='Noisy Data')
plt.plot(time_steps, filtered_data, label='Filtered Data', linewidth=2)
plt.plot(time_steps, original_data, label='Original Data', linestyle='dashed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Noise Reduction using Fourier Transforms')
plt.show()

