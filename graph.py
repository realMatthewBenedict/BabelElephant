from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from main_structures import FileResult

def graph_spectrogram(
    file: FileResult,
    spectrogram: tf.Tensor | np.ndarray | None,
    max_hz: float | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
):
    if spectrogram is None:
        spectrogram_np = file.spectrogram_tensor
    if isinstance(spectrogram, np.ndarray):
        spectrogram_np = spectrogram
    else: # assume it's a tf.Tensor still
        spectrogram_np = spectrogram.numpy()
    spectrogram_T = spectrogram_np.T
    num_bins, num_frames = spectrogram_T.shape

    hop_s = file.frame_step / file.sample_rate
    time_seconds_full = np.arange(num_frames) * hop_s

    # Optionally clamp time range
    if end_time is None:
        end_time = time_seconds_full[-1]

    # Mask for time frames inside [start_time, end_time]
    mask_time = (time_seconds_full >= start_time) & (time_seconds_full <= end_time)
    if np.sum(mask_time) == 0:
        raise ValueError(
            f"No frames in [{start_time}, {end_time}] seconds "
            f"(available: {time_seconds_full[0]}–{time_seconds_full[-1]} s)"
        )

    # Trim time dimension
    spectrogram_T = spectrogram_T[:, mask_time]
    time_seconds = time_seconds_full[mask_time]

    # Compute full frequency axis
    freq_full = np.linspace(0, file.sample_rate / 2, num_bins)

    # Optionally cap the frequency range
    if max_hz is not None:
        mask_freq = freq_full <= max_hz
        if np.sum(mask_freq) == 0:
            raise ValueError(
                f"max_hz={max_hz} Hz is below the lowest bin frequency."
            )
        spectrogram_T = spectrogram_T[mask_freq]
        freq_hz = freq_full[mask_freq]
    else:
        freq_hz = freq_full

    vmin, vmax = np.percentile(spectrogram_np, [1, 99])

    plt.figure(figsize=(10, 5))
    plt.imshow(
        spectrogram_T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[time_seconds[0], time_seconds[-1], freq_hz[0], freq_hz[-1]],
        vmin=vmin, vmax=vmax
    )
    plt.title(f"{file.file_name} Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"
    
    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    # Centered on selection 19 (rumble from 15.1885 to 21.89300667 seconds)
    graph_spectrogram(result, 1000, 10, 25)
