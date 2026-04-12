from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from noise_cnn_trainer import denoise_spectrogram
from main_structures import FileResult, FourierProperties

def de_log_enhanced(
    enhanced_spectrogram: tf.Tensor,
    allow_negative: bool = True,
) -> tf.Tensor:
    """Approximately undo log scale conversion"""
    return 10.0 ** (enhanced_spectrogram / 10.0)

def reconstruct_audio_tensor(file_props: FileResult, new_spectrogram: tf.Tensor) -> tf.Tensor:
    """Use original audio phase information and magnitude of spectrogram to reconstruct full audio tensor"""
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
    """Save reconstructed audio tensor"""
    # Ensure 2D [T, 1]
    if len(audio_tensor.shape) == 1:
        audio_2d = tf.reshape(audio_tensor, [-1, 1])
    else:
        audio_2d = tf.reshape(tf.reduce_mean(audio_tensor, axis=-1), [-1, 1])

    audio_2d = tf.cast(audio_2d, tf.float32)

    wav_bytes = tf.audio.encode_wav(audio_2d, sample_rate)
    tf.io.write_file(filename, wav_bytes)

def save_files(results: list[FileResult], model_tuple: tuple[tf.keras.Model, float]) -> None:
    """Use network to reconstruct and save audio from multiple files"""
    model, threshold = model_tuple
    for result in tqdm(results, desc="Saving", unit="file"):
        spectrogram = de_log_enhanced(denoise_spectrogram(result, model, threshold))
        audio_reconstruct = reconstruct_audio_tensor(result, spectrogram)
        output = Path(__file__).parent / "test" / result.file_name
        save_reconstructed_audio(audio_reconstruct, result.sample_rate, str(output))

if __name__ == "__main__":
    # --- Code for loading file data ---
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"

    from parse_audio import read_specific_audio_file
    results_dict = {test_file : read_specific_audio_file(index, directory, test_file)}
    results = list(results_dict.values())
    test_result = results_dict[test_file]

    from graph import graph_spectrogram
    graph_spectrogram(test_result, None, 1000)
    
    # --- Code for running enhancements ---
    from enhance import enhance_files
    enhance_files(results)

    save_path = Path(__file__).parent / "saved_models" / "noise_autoencoder.keras"
    from noise_cnn_trainer import train_noise_cnn, load_noise_cnn, denoise_spectrogram
    try:
        model, threshold = load_noise_cnn(save_path)
        print("Recovered model from saved data!")
    except FileNotFoundError as e:
        print("Training new model!")
        model, threshold = train_and_save_noise_model(results, save_path=save_path)

    # --- Code for testing model ---
    save_files(results, (model, threshold))

    # Test the reconstruction by reading it and plotting its spectrogram
    new_result = read_specific_audio_file(index, Path(__file__).parent / "test", test_file)
    # Centered on selection 19 (rumble from 15.1885 to 21.89300667 seconds)
    graph_spectrogram(new_result, None, 1000)
