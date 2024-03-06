import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def dynamic_cutoff_selection(power_spectrum, threshold_ratio):
    """
    Dynamically select a cutoff frequency based on a threshold ratio of the cumulative power spectrum.
    """
    cumulative_power = np.cumsum(power_spectrum)
    total_power = cumulative_power[-1]
    cutoff_index = np.where(cumulative_power > threshold_ratio * total_power)[0][0]
    return cutoff_index

def apply_window_function(data, window_type='hann'):
    """
    Apply a window function to the data to reduce spectral leakage.
    """
    window = np.ones(data.shape)
    if window_type in ['hann', 'hamming', 'blackman']:
        window = getattr(np, window_type)(data.size)
    return data * window

def reduce_noise_fourier_complex(data, threshold_ratio=0.9, band=(0.01, 0.1)):
    """
    Complex noise reduction using Fourier transforms with dynamic cutoff frequency selection and band-pass filtering.
    
    Parameters:
        data (numpy.ndarray): 1D array of time series data, assumed to be regularly spaced.
        threshold_ratio (float): Ratio for dynamic cutoff frequency selection based on cumulative power spectrum.
        band (tuple): Lower and upper frequencies for band-pass filtering.
        
    Returns:
        filtered_data (numpy.ndarray): Filtered time series data.
    """
    # Detrend and apply window function to the data
    detrended_data = detrend(data)
    windowed_data = apply_window_function(detrended_data)
    
    # Compute the Fourier transform
    fourier_transform = np.fft.fft(windowed_data)
    frequencies = np.fft.fftfreq(data.size, d=1)
    power_spectrum = np.abs(fourier_transform)**2
    
    # Dynamic cutoff frequency selection
    cutoff_index_low = dynamic_cutoff_selection(power_spectrum, threshold_ratio)
    cutoff_index_high = dynamic_cutoff_selection(power_spectrum[::-1], threshold_ratio)
    
    # Band-pass filtering
    band_indices = (np.abs(frequencies) >= band[0]) & (np.abs(frequencies) <= band[1])
    filter_mask = np.logical_or(band_indices, frequencies.size - band_indices)
    
    # Apply the filter
    filtered_transform = np.where(filter_mask, fourier_transform, 0)
    
    # Compute the inverse Fourier transform
    filtered_data = np.fft.ifft(filtered_transform)
    
    return filtered_data.real

# Example usage with synthetic data
np.random.seed(0)
time_steps = np.arange(100)
original_data = np.sin(2 * np.pi * 0.05 * time_steps)  # Low-frequency signal
noise = np.random.normal(0, 0.5, size=time_steps.size)  # Synthetic noise
noisy_data = original_data + noise

# Apply the complex noise reduction technique
filtered_data = reduce_noise_fourier_complex(noisy_data)

# Plotting the results
plt.figure(figsize=(14, 6))
plt.plot(time_steps, noisy_data, label='Noisy Data')
plt.plot(time_steps, filtered_data, label='Filtered Data', linewidth=2)
plt.plot(time_steps, original_data, label='Original Data', linestyle='dashed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Complex Noise Reduction using Fourier Transforms')
plt.show()
