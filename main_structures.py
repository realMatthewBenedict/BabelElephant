from dataclasses import dataclass
from enum import IntFlag

import tensorflow as tf

class Category(IntFlag):
    """Category of noise types present in a file"""
    Airplane = 1 << 0
    Background = 1 << 1
    Generator = 1 << 2
    Vehicle = 1 << 3

@dataclass
class FileResult:
    """Record containing all available data for an audio file"""
    file_name: str
    audio_tensor: tf.Tensor           # [samples, channels]
    spectrogram_tensor: tf.Tensor     # [257, time_frames] (unless modified by convert_to_mel)
    spectrogram_bins: int             # 257
    sample_rate: int
    category: Category
    rumble_samples: list[tuple[int, int]] # corresponds to audio_tensor
    rumble_frames: list[tuple[int, int]] # corresponds to spectrogram_tensor
    call_type: str

class FourierProperties:
    frame_length = 512
    frame_step = 256
    fft_length = 512