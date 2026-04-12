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
    sample_rate: int
    category: Category
    rumble_samples: list[tuple[int, int]] # corresponds to audio_tensor
    rumble_frames: list[tuple[int, int]] # corresponds to spectrogram_tensor
    call_type: str

    # Properties for spectrogram conversion
    frame_length: int
    frame_step: int
    fft_length: int

    def spectrogram_bins() -> int:
        return fft_length // 2 + 1

class FourierProperties:
    frame_length = 512
    fft_length = 512

    preferred_frame_step = 256    # 256 samples
    max_hop_s = 0.05              # 50 ms

    @staticmethod
    def frame_step_for_sample_rate(sr: int) -> int:
        # Maximum frame_step allowed by time constraint
        max_step_by_time = int(FourierProperties.max_hop_s * sr)
        # Prefer 256 if it’s acceptable
        if FourierProperties.preferred_frame_step <= max_step_by_time:
            return FourierProperties.preferred_frame_step
        else:
            return max_step_by_time
