from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from main_structures import FileResult
from parse_audio import get_spectrogram

def _prepare_spectrogram_data(
    file: FileResult,
    spectrogram: tf.Tensor | np.ndarray | None,
    max_hz: float | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper: return (spectrogram_2d, time_seconds, freq_hz),
    trimmed to [start_time, end_time] and [0, max_hz].
    """
    if spectrogram is None:
        sp = get_spectrogram(file.audio_tensor, file.sample_rate)
    else:
        sp = spectrogram

    if isinstance(sp, tf.Tensor):
        sp = sp.numpy()

    if sp.ndim == 3:
        sp = sp.squeeze(-1)

    T, F = sp.shape

    hop_s = file.frame_step / file.sample_rate
    time_seconds = np.arange(T) * hop_s

    # Optionally clamp time
    if end_time is None:
        end_time_val = time_seconds[-1] if len(time_seconds) > 0 else 0.0
    else:
        end_time_val = end_time

    mask_time = (time_seconds >= start_time) & (time_seconds <= end_time_val)
    if np.sum(mask_time) == 0:
        raise ValueError(
            f"No frames in [{start_time}, {end_time_val}] seconds "
            f"(available: {time_seconds[0]}–{time_seconds[-1]} s)"
        )

    sp = sp[mask_time]
    time_seconds = time_seconds[mask_time]

    # Frequency axis
    freq_full = np.linspace(0, file.sample_rate / 2, F)

    if max_hz is not None:
        mask_freq = freq_full <= max_hz
        if np.sum(mask_freq) == 0:
            raise ValueError(f"max_hz={max_hz} Hz is below the lowest bin frequency.")
        freq_hz = freq_full[mask_freq]
        sp = sp[:, mask_freq]
    else:
        freq_hz = freq_full

    return sp, time_seconds, freq_hz

def graph_spectrogram(
    file: FileResult,
    spectrogram: tf.Tensor | np.ndarray | None = None,
    max_hz: float | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> None:
    sp, time_seconds, freq_hz = _prepare_spectrogram_data(
        file=file,
        spectrogram=spectrogram,
        max_hz=max_hz,
        start_time=start_time,
        end_time=end_time,
    )

    vmin, vmax = np.percentile(sp, [1, 99])
    print(vmin, vmax)
    plt.figure(figsize=(10, 5))
    plt.imshow(
        sp.T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[time_seconds[0], time_seconds[-1], freq_hz[0], freq_hz[-1]],
        vmin=vmin, vmax=vmax,
    )
    plt.title(f"{file.file_name} Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

def graph_spectrogram_dual(
    file: FileResult,
    enhanced_spectrogram: tf.Tensor | np.ndarray | None = None,
    max_hz: float | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
    title: str | None = None,
) -> None:
    """
    Plot both original file.spectrogram_tensor and enhanced_spectrogram side‑by‑side,
    with colorbars.
    """
    orig_sp, time_seconds, freq_hz = _prepare_spectrogram_data(
        file=file,
        spectrogram=None,  # original spectrogram
        max_hz=max_hz,
        start_time=start_time,
        end_time=end_time,
    )

    if enhanced_spectrogram is not None:
        enh_sp, _, _ = _prepare_spectrogram_data(
            file=file,
            spectrogram=enhanced_spectrogram,
            max_hz=max_hz,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        enh_sp = orig_sp.copy()

    vmin, vmax = np.percentile(np.concatenate([orig_sp.ravel(), enh_sp.ravel()]), [1, 99])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    im1 = ax1.imshow(
        orig_sp.T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[time_seconds[0], time_seconds[-1], freq_hz[0], freq_hz[-1]],
        vmin=vmin, vmax=vmax,
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Original")

    im2 = ax2.imshow(
        enh_sp.T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[time_seconds[0], time_seconds[-1], freq_hz[0], freq_hz[-1]],
        vmin=vmin, vmax=vmax,
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Modified (Enhanced/Reconstructed)")

    # Add colorbar for each subplot
    fig.colorbar(im1, ax=ax1, label="Magnitude")
    fig.colorbar(im2, ax=ax2, label="Magnitude")

    if title is None:
        title = f"{file.file_name} Spectrogram comparison"

    fig.suptitle(title)
    plt.show()

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"
    
    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    # Centered on selection 19 (rumble from 15.1885 to 21.89300667 seconds)
    graph_spectrogram(result, None, 1000, 10, 25)