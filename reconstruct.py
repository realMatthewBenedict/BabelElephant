from pathlib import Path

import numpy as np
import tensorflow as tf

from main_structures import FileResult, FourierProperties

def de_log_enhanced(enhanced_spectrogram: np.ndarray) -> np.ndarray:
    """
    Attempt to approximately undo the log scaling in enhance_func.

    This is best‑effort; coherence and threshold shifts cannot be inverted.
    """
    # If enhance_func did:
    #   log_ridge = 10 * np.log10(ridge_s)
    #   enhanced = log_ridge * (1 + coherence) + threshold_shifts
    # then this only undoes the log part, assuming ridge_s ≈ 10**(log_ridge / 10)
    # and ignores coherence and thresholds.

    # Here we just do a fake “undo log” (only reasonable if you know
    # that enhance_func applied no further scaling after log10):
    return 10 ** (enhanced_spectrogram / 10)

def reconstruct_linear_from_mel(file_res: FileResult, num_spectrogram_bins: int = 257):
    """
    Attempt to reconstruct the original linear‑scale spectrogram from a mel‑scaled one.

    This is approximate and lossy because mel‑weighting is not invertible.

    Parameters:
        file_res (FileResult):
            Must have spectrogram_tensor = mel_spectrogram of shape [time, num_mel_bins].
        num_spectrogram_bins (int):
            Number of linear frequency bins (e.g., 257 for fft_length=512).

    Returns:
        Linear‑scale spectrogram (tf.Tensor) with shape [time, num_spectrogram_bins],
        or None if mel_weighting cannot be reconstructed.
    """
    mel_spectrogram = file_res.spectrogram_tensor  # [time, num_mel_bins]
    mel_bins = mel_spectrogram.shape[-1]
    sample_rate = file_res.sample_rate

    # Recompute the same mel‑weight matrix used in convert_to_mel
    lower_edge_hertz, upper_edge_hertz = 0.0, 1000.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
    )  # [num_spectrogram_bins, num_mel_bins]

    # Convert to NumPy and compute pseudo‑inverse
    W = linear_to_mel_weight_matrix.numpy()  # [F, M]
    W_pinv = np.linalg.pinv(W)              # [M, F]

    # Convert back to TF
    W_pinv_tf = tf.convert_to_tensor(W_pinv, dtype=mel_spectrogram.dtype)

    # Invert: mel [T, M] @ W_pinv [M, F] -> [T, F]
    reconstructed_linear = tf.tensordot(mel_spectrogram, W_pinv_tf, axes=1)

    # Ensure shape is consistent with original
    reconstructed_linear.set_shape(
        tf.TensorShape(mel_spectrogram.shape[:-1].as_list() + [num_spectrogram_bins])
    )

    return reconstructed_linear

def reconstruct_audio_tensor(file_props: FileResult, new_spectrogram: tf.Tensor) -> tf.Tensor:
    # Original audio
    stft_original = tf.signal.stft(file_props.audio_tensor,
        frame_length=file_props.frame_length,
        frame_step=file_props.frame_step,
        fft_length=file_props.fft_length
    )

    # Modified spectrogram (do NOT pass mel-transformed spectrogram here)
    modified_magnitude = tf.abs(new_spectrogram)  # [257, time_frames]

    # Reconstruct with ORIGINAL phase:
    angle = tf.math.angle(stft_original)
    phase = tf.exp(tf.complex(tf.zeros_like(angle), angle))
    modified_magnitude_complex = tf.cast(modified_magnitude, phase.dtype)
    stft_modified = modified_magnitude_complex * phase

    # Back to audio:
    audio_reconstruct = tf.signal.inverse_stft(
        stft_modified,
        frame_length=file_props.frame_length,
        frame_step=file_props.frame_step,
        fft_length=file_props.fft_length
    )
    return audio_reconstruct

def save_reconstructed_audio(audio_tensor: tf.Tensor, sample_rate: int, filename: str):
    # Ensure 2D [T, 1]
    if len(audio_tensor.shape) == 1:
        audio_2d = tf.reshape(audio_tensor, [-1, 1])
    else:
        audio_2d = tf.reshape(tf.reduce_mean(audio_tensor, axis=-1), [-1, 1])

    audio_2d = tf.cast(audio_2d, tf.float32)

    wav_bytes = tf.audio.encode_wav(audio_2d, sample_rate)
    tf.io.write_file(filename, wav_bytes)


if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"
    
    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    
    spectrogram_mult = tf.math.scalar_mul(2.0, result.spectrogram_tensor)
    audio_reconstruct = reconstruct_audio_tensor(result, spectrogram_mult)
    output = Path(__file__).parent / "test" / "1989-06_airplane_01.wav"
    save_reconstructed_audio(audio_reconstruct, result.sample_rate, str(output))

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"

    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    
    spectrogram_mult = tf.math.scalar_mul(2.0, result.spectrogram_tensor)
    audio_reconstruct = reconstruct_audio_tensor(result, spectrogram_mult)
    output = Path(__file__).parent / "test" / "1989-06_airplane_01.wav"
    save_reconstructed_audio(audio_reconstruct, result.sample_rate, str(output))
    
    # Test the reconstruction by reading it and plotting its spectrogram
    from graph import graph_spectrogram
    new_result = read_specific_audio_file(index, Path(__file__).parent / "test", test_file)
    # Centered on selection 19 (rumble from 15.1885 to 21.89300667 seconds)
    graph_spectrogram(new_result, 1000, 10, 25)
