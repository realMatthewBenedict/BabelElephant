from pathlib import Path

import numpy as np
import tensorflow as tf

from enhance import enhance_files
from graph import graph_spectrogram
from main_structures import FileResult
from parse_audio import read_audio_files, convert_to_mel
from noise_cnn_trainer import train_noise_cnn, load_noise_cnn, denoise_spectrogram
from reconstruct import de_log_enhanced, reconstruct_linear_from_mel

def train_and_save_noise_model(
    file_results: list[FileResult],
    save_path: Path | str | None = None,
) -> tuple[tf.keras.Model, float]:
    """Convert data and train a noise-cnn model from FileResult objects."""
    for file_res in file_results:
        convert_to_mel(file_res)

    model, threshold = train_noise_cnn(file_results, save_model_path=save_path)
    if save_path is not None:
        print(f"Saved trained model to {save_path}")
    print(f"Model reconstruction threshold: {threshold:.6f}")
    return model, threshold

def graph_model_spectrogram(result: FileResult, model_tuple: tuple[tf.keras.Model, float]) -> None:
    model, threshold = model_tuple
    spectrogram = denoise_spectrogram(result, model, threshold)
    reconstructed = de_log_enhanced(reconstruct_linear_from_mel(result, spectrogram))
    graph_spectrogram(result, reconstructed)

def main() -> None:
    # --- Code for loading file data ---
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    results_dict = read_audio_files(index, directory)
    results = list(results_dict.values())
    
    test_result = results_dict["1989-06_airplane_01.wav"]
    
    # --- Code for running enhancements ---
    enhance_files(results)
    #enhance_files([test_result])
    #graph_spectrogram(results[0], None, 1000, 10, 25)

    save_path = Path(__file__).parent / "saved_models" / "noise_autoencoder.keras"
    try:
        model, threshold = load_noise_cnn(save_path)
        print("Recovered model from saved data!")
    except FileNotFoundError as e:
        print("Training new model!")
        model, threshold = train_and_save_noise_model(results, save_path=save_path)

    # --- Code for testing model ---
    graph_model_spectrogram(test_result, (model, threshold))

if __name__ == "__main__":
    main()
