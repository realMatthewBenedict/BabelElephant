from pathlib import Path

from parse_audio import read_audio_files, convert_to_mel
import enhance
from noise_cnn_trainer import train_noise_cnn
import tensorflow as tf
import numpy as np
from main_structures import FileResult
from graph import graph_spectrogram

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


if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    results = read_audio_files(index, directory)
    print(f"{len(results)} files parsed")
    
    print(f"Loaded {len(results)} files for training")
    
    # --- Code for running enhancements ---
    enhanced_files = []
    i = 0
    for file in results:
        spectrogram_np = file.spectrogram_tensor.numpy()
        spectrogram_db = 10 * np.log10(spectrogram_np + 1e-10)
        enhanced = enhance.enhance_func(spectrogram_np)
        enhanced_file = results[i]
        enhanced_file.spectrogram_tensor = enhanced
        enhanced_files.append(enhanced_file)
        i += 1
    # -------------------------------------
    
    save_path = Path(__file__).parent / "saved_models" / "noise_autoencoder.keras"
    train_and_save_noise_model(enhanced_files, save_path=save_path)
