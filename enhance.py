import time

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import tensorflow as tf
from tqdm import tqdm

from main_structures import FileResult

def ridge_filter(s):
    log = -gaussian_laplace(s, sigma=1.5)
    ridge_s = np.clip(log, 1e-10, None)
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
    threshold1 = np.percentile(s1, 25)
    threshold2 = np.percentile(s1, 50)
    threshold3 = np.percentile(s1, 75)

    result = np.copy(s1)

    # Above Q3
    mask3 = s1 > threshold3
    result[mask3] += 5

    # Q2–Q3
    mask2 = (s1 <= threshold3) & (s1 > threshold2)
    result[mask2] += 2

    # Q1–Q2
    mask1 = (s1 <= threshold2) & (s1 > threshold1)
    result[mask1] -= 2

    # Q1 and below
    mask0 = s1 <= threshold1
    result[mask0] -= 5

    return result

def enhance_func(spectrogram):
    """Warning: These enhancements change the scaling nonlinearly and cannot be fully reversed."""
    enhanced = compute_structure_tensor(spectrogram, sigma=3.0)
    enhanced = threshold_based_enhancement(enhanced)
    return enhanced

def enhance_files(results: list[FileResult]) -> None:
    for i, file in enumerate(tqdm(results, desc="Enhancing spectrograms")):
        spectrogram_np = file.spectrogram_tensor.numpy()
        orig_dtype = file.spectrogram_tensor.dtype
        enhanced = enhance_func(spectrogram_np)
        results[i].spectrogram_tensor = tf.convert_to_tensor(enhanced, dtype=orig_dtype)
