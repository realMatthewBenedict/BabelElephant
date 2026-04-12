import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.filters import frangi
import time


def ridge_filter(s):
    log = -gaussian_laplace(s, sigma=1.5)
    ridge_s = np.clip(log, 0, None)
    ridge_s = np.clip(ridge_s, 1e-10, None)
    return ridge_s + s   # additive rather than replacing original signal


def compute_structure_tensor(s, sigma=1.0):
    """
    Enhance a spectrogram using structure tensor coherence.

    Parameters:
        s (np.ndarray): Input spectrogram (2D array, shape: [time, frequency])
        sigma (float): Gaussian smoothing sigma for tensor elements

    Returns:
        np.ndarray: Enhanced spectrogram
    """
    # Step 1: Apply ridge filtering
    ridge_s = ridge_filter(s)

    # Step 2: Calculate gradients
    time_gradient = np.gradient(ridge_s, axis=0)       # Along time axis
    frequency_gradient = np.gradient(ridge_s, axis=1)  # Along frequency axis

    # Step 3: Calculate structure tensor elements
    tensor_xx = time_gradient ** 2
    tensor_yy = frequency_gradient ** 2
    tensor_xy = time_gradient * frequency_gradient

    # Step 4: Smooth tensor elements with Gaussian filter
    tensor_xx_smooth = gaussian_filter(tensor_xx, sigma=sigma)
    tensor_yy_smooth = gaussian_filter(tensor_yy, sigma=sigma)
    tensor_xy_smooth = gaussian_filter(tensor_xy, sigma=sigma)

    # Step 5: Calculate eigenvalues of the 2x2 structure tensor at each pixel
    # For a 2x2 symmetric matrix [[a, b], [b, d]]:
    # eigenvalues = ((a+d) ± sqrt((a-d)^2 + 4*b^2)) / 2
    trace = tensor_xx_smooth + tensor_yy_smooth
    diff = tensor_xx_smooth - tensor_yy_smooth
    discriminant = np.sqrt(diff ** 2 + 4 * tensor_xy_smooth ** 2 + 1e-10)

    lambda1 = (trace + discriminant) / 2  # Larger eigenvalue
    lambda2 = (trace - discriminant) / 2  # Smaller eigenvalue

    # Step 6: Calculate coherence
    denominator = lambda1 + lambda2
    coherence = np.where(
        np.abs(denominator) > 1e-10,
        (lambda1 - lambda2) / denominator,
        0.0
    )
    coherence = np.clip(coherence, 0, 1)  # Coherence in [0, 1]

    # Step 7: Calculate enhanced spectrogram
    log_ridge = 10 * np.log10(ridge_s)
    enhanced_spectrogram = log_ridge * (1 + coherence)

    return enhanced_spectrogram

def threshold_based_enhancement(s1: np.ndarray) -> np.ndarray:
    """
    Threshold-based spectrogram enhancement (Algorithm 2).

    Parameters:
        s1 (np.ndarray): Input spectrogram (2D array)

    Returns:
        np.ndarray: Further enhanced spectrogram
    """
    # Calculate threshold values from s1
    threshold1 = np.percentile(s1, 25)   # 25th percentile (Q1)
    threshold2 = np.percentile(s1, 50)   # 50th percentile (median)
    threshold3 = np.percentile(s1, 75)   # 75th percentile (Q3)

    final_spectrogram = np.empty_like(s1)

    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            if s1[i, j] > threshold3:
                final_spectrogram[i, j] = s1[i, j] + 5
            elif s1[i, j] > threshold2:
                final_spectrogram[i, j] = s1[i, j] + 2
            elif s1[i, j] > threshold1:
                final_spectrogram[i, j] = s1[i, j] - 2
            else:
                final_spectrogram[i, j] = s1[i, j] - 5

    return final_spectrogram

def enhance_func(spectrogram):
    enhanced = compute_structure_tensor(spectrogram, sigma=3.0)
    enhanced = threshold_based_enhancement(enhanced)
    return enhanced

from main_structures import FileResult

def enhance_files(results: list[FileResult]) -> list[FileResult]:
    enhanced_files = []
    i = 0
    for file in results:
        spectrogram_np = file.spectrogram_tensor.numpy()
        spectrogram_db = 10 * np.log10(spectrogram_np + 1e-10)
        enhanced = enhance_func(spectrogram_np)
        enhanced_file = results[i]
        enhanced_file.spectrogram_tensor = enhanced
        enhanced_files.append(enhanced_file)
        i += 1
    
    return enhanced_files
