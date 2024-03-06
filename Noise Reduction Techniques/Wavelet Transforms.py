import numpy as np
import matplotlib.pyplot as plt
import pywt

def thresholding(coeffs, method='soft', value=None):
    """
    Apply thresholding to wavelet coefficients.
    
    Parameters:
        coeffs (list of ndarray): Wavelet coefficients.
        method (str): Thresholding method ('soft' or 'hard').
        value (float): Threshold value. If None, a universal threshold is used.
        
    Returns:
        thresholded_coeffs (list of ndarray): Thresholded wavelet coefficients.
    """
    thresholded_coeffs = []
    if value is None:
        # Universal threshold (Donoho's method)
        value = np.sqrt(2 * np.log(len(coeffs[0]))) * np.median(np.abs(coeffs[-1])) / 0.6745
        
    for coeff in coeffs:
        if method == 'soft':
            thresholded = np.sign(coeff) * np.maximum(np.abs(coeff) - value, 0)
        elif method == 'hard':
            thresholded = coeff * (np.abs(coeff) > value)
        else:
            thresholded = coeff
        thresholded_coeffs.append(thresholded)
    return thresholded_coeffs

def reduce_noise_wavelet(data, wavelet_name='db4', level=None, threshold_method='soft', threshold_value=None):
    """
    Reduce noise in a time series using wavelet transforms.
    
    Parameters:
        data (numpy.ndarray): 1D array of time series data.
        wavelet_name (str): Name of the wavelet to use.
        level (int): Maximum level of wavelet decomposition. If None, the level is calculated based on data length.
        threshold_method (str): Method for thresholding ('soft' or 'hard').
        threshold_value (float): Threshold value. If None, a universal threshold is used.
        
    Returns:
        filtered_data (numpy.ndarray): 1D array of filtered time series data.
    """
    # Wavelet decomposition
    wavelet = pywt.Wavelet(wavelet_name)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Thresholding
    thresholded_coeffs = thresholding(coeffs, method=threshold_method, value=threshold_value)
    
    # Wavelet reconstruction
    filtered_data = pywt.waverec(thresholded_coeffs, wavelet)
    
    return filtered_data[:len(data)]  # Ensure the output is the same length as the input

# Example usage with synthetic data
np.random.seed(0)
time_steps = np.arange(100)
original_data = np.sin(2 * np.pi * 0.05 * time_steps)  # Low-frequency signal
noise = np.random.normal(0, 0.5, size=time_steps.size)  # Synthetic noise
noisy_data = original_data + noise

# Apply the wavelet-based noise reduction technique
filtered_data = reduce_noise_wavelet(noisy_data)

# Plotting the results
plt.figure(figsize=(14, 6))
plt.plot(time_steps, noisy_data, label='Noisy Data')
plt.plot(time_steps, filtered_data, label='Filtered Data', linewidth=2)
plt.plot(time_steps, original_data, label='Original Data', linestyle='dashed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Noise Reduction using Wavelet Transforms')
plt.show()

