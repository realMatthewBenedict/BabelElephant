from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import sklearn
import warnings
warnings.filterwarnings("ignore", message="file system plugins")
warnings.filterwarnings("ignore", message="libtensorflow_io")

import tensorflow as tf
import tensorflow_io as tfio


class Category(IntFlag):
    Airplane = 1 << 0
    Background = 1 << 1
    Generator = 1 << 2
    Vehicle = 1 << 3

@dataclass
class FileResult:
    audioTensor: tf.Tensor
    sampleTensor: tf.Tensor
    category: Category

def read_audio_files(directory: Path) -> [FileResult]:
    result = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            audio_bytes = tf.io.read_file(file_path)
            audio, sample_rate = tf.audio.decode_wav(audio_bytes)
            category = Category(0)
            path = file_path.to_lower()
            if path.contains("airplane"):
                category |= Category.Airplane
            if path.contains("background"):
                category |= Category.Background
            if path.contains("generator"):
                category |= Category.Background
            if path.contains("vehicle"):
                category |= Category.Vehicle
            result.append(FileResult(audioTensor, sampleTensor, category))
    return result

if __name__ == "main":
    directory = Path("data")
    results = read_audio_files(directory)

"""
tfio.audio.spectrogram(
    input, nfft, window, stride, name=None
)
"""

"""
tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms, name=None
)
"""
