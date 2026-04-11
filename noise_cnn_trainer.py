from __future__ import annotations

from typing import Iterable, Protocol, Tuple

import numpy as np
import tensorflow as tf


class FileResultLike(Protocol):
    file_name: str
    audio_tensor: tf.Tensor
    spectrogram_tensor: tf.Tensor
    spectrogram_bins: int
    sample_rate: int
    category: object
    rumble_samples: list[Tuple[int, int]]
    rumble_frames: list[Tuple[int, int]]
    call_type: str


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


"""convert noise samples to np array (random samples)"""
def _sample_noise_patches(
    file_result: FileResultLike,
    patch_size: Tuple[int, int],
    patches_per_file: int,
) -> np.ndarray:
    spectrogram = _normalize_spectrogram(
        file_result.spectrogram_tensor,
        file_result.spectrogram_bins,
    )

    num_time_frames = int(spectrogram.shape[0])
    num_freq_bins = int(spectrogram.shape[1])
    noise_mask = _build_noise_mask(num_time_frames, file_result.rumble_frames)
    valid_time_indices = np.where(noise_mask)[0]
    if len(valid_time_indices) == 0:
        return np.zeros((0, patch_size[0], patch_size[1], 1), dtype=np.float32)

    pad_top = patch_size[0] // 2
    pad_left = patch_size[1] // 2
    padded = tf.pad(
        spectrogram[:, :, tf.newaxis],
        [[pad_top, pad_top], [pad_left, pad_left], [0, 0]],
        mode="REFLECT",
    )

    rng = np.random.default_rng()
    patches = []
    for _ in range(patches_per_file):
        time_center = rng.choice(valid_time_indices) + pad_top
        freq_center = rng.integers(pad_left, num_freq_bins + pad_left)
        patch = padded[
            time_center - pad_top : time_center + pad_top + 1,
            freq_center - pad_left : freq_center + pad_left + 1,
            :,
        ]
        patches.append(patch.numpy())

    return np.stack(patches, axis=0)

"""autoencoder"""
def _build_autoencoder(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
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


def train_noise_cnn(
    file_results: Iterable[FileResultLike],
    patch_size: Tuple[int, int] = (32, 32),
    patches_per_file: int = 256,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_split: float = 0.1,
) -> tuple[tf.keras.Model, float]:

    all_patches = []
    for result in file_results:
        patches = _sample_noise_patches(result, patch_size, patches_per_file)
        if patches.size > 0:
            all_patches.append(patches)

    if not all_patches:
        raise ValueError("No noise patches extracted check frames and spectrogram shape")

    x_train = np.concatenate(all_patches, axis=0).astype(np.float32)
    max_value = np.maximum(np.max(x_train), 1e-8)
    x_train = x_train / max_value

    input_shape = x_train.shape[1:]
    model = _build_autoencoder(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
    )

    model.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        shuffle=True,
        verbose=2,
    )

    reconstructed = model.predict(x_train, batch_size=batch_size)
    reconstruction_error = np.mean(np.square(reconstructed - x_train), axis=(1, 2, 3))
    threshold = float(np.mean(reconstruction_error) + 3.0 * np.std(reconstruction_error))

    return model, threshold

    """dataset is noise only for training"""
def make_noise_dataset(
    file_results: Iterable[FileResultLike],
    patch_size: Tuple[int, int] = (32, 32),
    patches_per_file: int = 256,
) -> tf.data.Dataset:
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
