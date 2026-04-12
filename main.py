from pathlib import Path

from parse_audio import read_audio_files


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
    #convert_to_mel(results[0])
    print(f"Loaded {len(results)} files for training")

    save_path = Path(__file__).parent / "saved_models" / "noise_autoencoder.keras"
    train_and_save_noise_model(results, save_path=save_path)
