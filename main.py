from pathlib import Path

from parse_audio import read_audio_files

import enhance
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    results = read_audio_files(index, directory)
    print(f"{len(results)} files parsed")
    #convert_to_mel(results[0])
    print(results[10])
    spectrogram_np = results[10].spectrogram_tensor.numpy()
    spectrogram_db = 10 * np.log10(spectrogram_np + 1e-10)

    enhanced = enhance.enhance_func(spectrogram_np)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(spectrogram_db.T, aspect="auto", origin="lower",
                   cmap="inferno", extent=[0, 1, 0, 1])
    axes[0].set_title("Original Spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    axes[1].imshow(enhanced.T, aspect="auto", origin="lower",
                   cmap="inferno", extent=[0, 1, 0, 1])
    axes[1].set_title("Enhanced Spectrogram")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")