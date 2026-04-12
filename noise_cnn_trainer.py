from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Tuple

import json
import numpy as np
import tensorflow as tf

from main_structures import FileResult

def _normalize_spectrogram(
    spectrogram: tf.Tensor,
    spectrogram_bins: int,
) -> tf.Tensor:
    """Normalize spectrogram to [time, frequency, channels] layout."""
    spectrogram = tf.convert_to_tensor(spectrogram, dtype=tf.float32)
    if spectrogram.ndim == 2:
        if int(spectrogram.shape[0]) == spectrogram_bins:
            spectrogram = tf.transpose(spectrogram)
        spectrogram = spectrogram[:, :, tf.newaxis]
    elif spectrogram.ndim == 3:
        if int(spectrogram.shape[0]) == spectrogram_bins:
            spectrogram = tf.transpose(spectrogram, perm=[1, 0, 2])
    else:
        raise ValueError(
            "spectrogram_tensor must be 2D or 3D with shape [time, frequency, channels] or [frequency, time, channels]."
        )
    return spectrogram

"""mark as noise or not"""
def _build_noise_mask(
    num_time_frames: int,
    rumble_frames: list[Tuple[int, int]],
) -> np.ndarray:
    mask = np.ones(num_time_frames, dtype=bool)
    for start_frame, end_frame in rumble_frames:
        start_frame = max(0, start_frame)
        end_frame = min(num_time_frames, end_frame + 1)
        mask[start_frame:end_frame] = False
    return mask

def _sample_noise_patches(
    file_result: FileResult,
    patch_size: Tuple[int, int],
    patches_per_file: int,
) -> np.ndarray:
    """convert noise samples to np array (random samples)"""
    spectrogram = _normalize_spectrogram(
        file_result.spectrogram_tensor,
        file_result.spectrogram_bins(),
    )

    num_time_frames = int(spectrogram.shape[0])
    num_freq_bins = int(spectrogram.shape[1])
    if num_time_frames == 0 or num_freq_bins == 0:
        return np.zeros((0, patch_size[0], patch_size[1], 1), dtype=np.float32)

    noise_mask = _build_noise_mask(num_time_frames, file_result.rumble_frames)
    valid_time_indices = np.where(noise_mask)[0]
    if len(valid_time_indices) == 0:
        return np.zeros((0, patch_size[0], patch_size[1], 1), dtype=np.float32)

    pad_top = patch_size[0] // 2
    pad_left = patch_size[1] // 2
    pad_mode = "REFLECT" if num_time_frames > pad_top and num_freq_bins > pad_left else "CONSTANT"
    padded = tf.pad(
        spectrogram,
        [[pad_top, pad_top], [pad_left, pad_left], [0, 0]],
        mode=pad_mode,
    )

    rng = np.random.default_rng()
    patches = []
    for _ in range(patches_per_file):
        time_center = rng.choice(valid_time_indices) + pad_top
        freq_center = rng.integers(pad_left, num_freq_bins + pad_left)
        time_start = time_center - pad_top
        freq_start = freq_center - pad_left
        patch = padded[
            time_start : time_start + patch_size[0],
            freq_start : freq_start + patch_size[1],
            :,
        ]
        patches.append(patch.numpy())

    return np.stack(patches, axis=0)


def _build_autoencoder(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """autoencoder"""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D(2, padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(2, padding="same")(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    model = tf.keras.Model(inputs, outputs, name="noise_autoencoder")
    return model


import json
from typing import Iterable, Tuple, List
from pathlib import Path
import numpy as np
import tensorflow as tf


def train_noise_cnn(
    file_results: Iterable[FileResult],
    patch_size: Tuple[int, int] = (32, 32),
    patches_per_file: int = 256,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_split: float = 0.1,
    save_model_path: str | Path | None = None,
) -> tuple[tf.keras.Model, float]:
    all_patches = []
    for result in file_results:
        patches = _sample_noise_patches(result, patch_size, patches_per_file)
        if patches.size > 0:
            all_patches.append(patches)

    if not all_patches:
        raise ValueError("No noise patches extracted; check frames and spectrogram shape")

    x_train = np.concatenate(all_patches, axis=0).astype(np.float32)
    max_value = np.maximum(np.max(x_train), 1e-8)
    x_train = x_train / max_value

    input_shape = x_train.shape[1:]
    model = _build_autoencoder(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
    )

    history = model.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=True,
        verbose=2,
    )

    # Compute reconstruction error and threshold
    reconstructed = model.predict(x_train, batch_size=batch_size)
    reconstruction_error = np.mean(np.square(reconstructed - x_train), axis=(1, 2, 3))
    threshold = float(np.mean(reconstruction_error) + 3.0 * np.std(reconstruction_error))

    # Save model and threshold
    if save_model_path is not None:
        save_path = Path(save_model_path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".keras")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save(str(save_path))

        # Save threshold in a companion JSON file
        meta_path = save_path.with_suffix(".threshold.json")
        with open(meta_path, "w") as f:
            json.dump({"threshold": threshold, "reconstruction_error_mean": float(np.mean(reconstruction_error))}, f, indent=2)

    return model, threshold

def load_noise_cnn(save_path: str | Path) -> tuple[tf.keras.Model, float]:
    """
    Load a saved noise CNN autoencoder and its reconstruction threshold.

    Args:
        save_path: Path to the .keras file or its parent directory.

    Returns:
        (model, threshold)
    """
    save_path = Path(save_path)

    # If given a directory, find the .keras file
    if save_path.is_dir():
        keras_files = list(save_path.glob("*.keras"))
        if not keras_files:
            raise FileNotFoundError(f"No .keras file found in {save_path}")
        save_path = keras_files[0]

    if not save_path.exists():
        raise FileNotFoundError(f"Model file not found: {save_path}")

    # Load model
    model = tf.keras.models.load_model(str(save_path))

    # Load threshold metadata
    meta_path = save_path.with_suffix(".threshold.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Threshold metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    threshold = float(meta["threshold"])

    return model, threshold

def make_noise_dataset(
    file_results: Iterable[FileResult],
    patch_size: Tuple[int, int] = (32, 32),
    patches_per_file: int = 256,
) -> tf.data.Dataset:
    """dataset is noise only for training"""
    all_patches = []
    for result in file_results:
        patches = _sample_noise_patches(result, patch_size, patches_per_file)
        if patches.size > 0:
            all_patches.append(patches)

    if not all_patches:
        raise ValueError("No noise patches extracted check frames and spectrogram shape")

    x = np.concatenate(all_patches, axis=0).astype(np.float32)
    max_value = np.maximum(np.max(x), 1e-8)
    x = x / max_value
    dataset = tf.data.Dataset.from_tensor_slices((x, x))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(patches_per_file).prefetch(tf.data.AUTOTUNE)
    return dataset

def denoise_spectrogram(
    file_result: FileResult,
    model: tf.keras.Model,
    threshold: float,
    patch_size: Tuple[int, int] = (32, 32),
    time_step: int = 8,
    freq_step: int = 8,
) -> tf.Tensor:
    """remove noise from existing spectrogram"""
    raw_spectrogram = tf.convert_to_tensor(file_result.spectrogram_tensor, dtype=tf.float32)
    original_ndim = raw_spectrogram.ndim
    spectrogram = _normalize_spectrogram(raw_spectrogram, file_result.spectrogram_bins())

    num_time_frames = int(spectrogram.shape[0])
    num_freq_bins = int(spectrogram.shape[1])
    if num_time_frames == 0 or num_freq_bins == 0:
        return raw_spectrogram

    target_mask = ~_build_noise_mask(num_time_frames, file_result.rumble_frames)
    target_mask = tf.convert_to_tensor(target_mask)
    target_mask_float = tf.cast(target_mask[:, None, None], tf.float32)

    output = spectrogram * target_mask_float

    pad_top = patch_size[0] // 2
    pad_left = patch_size[1] // 2
    pad_mode = "REFLECT" if num_time_frames > pad_top and num_freq_bins > pad_left else "CONSTANT"
    padded = tf.pad(
        spectrogram,
        [[pad_top, pad_top], [pad_left, pad_left], [0, 0]],
        mode=pad_mode,
    )

    time_indices = np.arange(pad_top, pad_top + num_time_frames, time_step)
    target_mask_np = target_mask.numpy()
    time_indices = [t for t in time_indices if target_mask_np[t - pad_top]]
    freq_indices = np.arange(pad_left, pad_left + num_freq_bins, freq_step)
    if len(time_indices) == 0 or len(freq_indices) == 0:
        return tf.squeeze(output, axis=-1) if original_ndim == 2 else output

    patches = []
    coords = []
    for time_center in time_indices:
        for freq_center in freq_indices:
            patch = padded[
                time_center - pad_top : time_center - pad_top + patch_size[0],
                freq_center - pad_left : freq_center - pad_left + patch_size[1],
                :,
            ]
            patches.append(patch)
            coords.append((time_center - pad_top, freq_center - pad_left))

    patches_tensor = tf.stack(patches, axis=0)
    max_value = tf.maximum(tf.reduce_max(spectrogram), 1e-8)
    normalized_patches = patches_tensor / max_value

    reconstructed_patches = model.predict(normalized_patches, batch_size=max(1, min(128, normalized_patches.shape[0])), verbose=0)
    reconstructed_patches = reconstructed_patches * max_value

    errors = np.mean(np.square(patches_tensor.numpy() - reconstructed_patches), axis=(1, 2, 3))
    noise_like = errors < threshold

    noise_accum = np.zeros((num_time_frames, num_freq_bins, 1), dtype=np.float32)
    count_accum = np.zeros((num_time_frames, num_freq_bins, 1), dtype=np.float32)
    for i, (time_start, freq_start) in enumerate(coords):
        if not noise_like[i]:
            continue
        noise_patch = reconstructed_patches[i]
        noise_accum[
            time_start : time_start + patch_size[0],
            freq_start : freq_start + patch_size[1],
            :,
        ] += noise_patch
        count_accum[
            time_start : time_start + patch_size[0],
            freq_start : freq_start + patch_size[1],
            :,
        ] += 1.0

    avg_noise = noise_accum / np.maximum(count_accum, 1.0)
    avg_noise = tf.convert_to_tensor(avg_noise, dtype=tf.float32)

    output = output - avg_noise
    output = tf.maximum(output, 0.0)
    output = output * target_mask_float

    if original_ndim == 2:
        return tf.squeeze(output, axis=-1)
    return output
