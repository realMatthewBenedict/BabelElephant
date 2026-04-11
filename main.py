from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

class Category(IntFlag):
    Airplane = 1 << 0
    Background = 1 << 1
    Generator = 1 << 2
    Vehicle = 1 << 3

@dataclass
class FileResult:
    audio_tensor: tf.Tensor
    spectrogram_tensor: tf.Tensor
    spectrogram_bins: np.int32
    sample_rate: np.int32
    category: Category
    start_second: float
    end_second: float
    call_type: str

def read_audio_files(index_path: Path, directory: Path) -> list[FileResult]:
    df = pd.read_csv(str(index_path), index_col="Selection")
    result = []

    for index, row in df.iterrows():
        file_path = directory / row["Sound_file"]
        audio_bytes = tf.io.read_file(str(file_path))
        audio, sample_rate = tf.audio.decode_wav(audio_bytes)
        sample_rate_int = sample_rate.numpy()
        stft = tf.signal.stft(audio, frame_length=512, frame_step=256, fft_length=512)
        spectrogram = tf.abs(stft)
        category = Category(0)
        path = file_path.name.lower()
        if "airplane" in path:
            category |= Category.Airplane
        if "background" in path:
            category |= Category.Background
        if "generator" in path:
            category |= Category.Generator
        if "vehicle" in path:
            category |= Category.Vehicle
        result.append(FileResult(
            audio, stft.shape[-1], spectrogram, sample_rate_int, category,
            row["Start_time"], row["End_time"], row["Call_type"]
        ))
    return result

def convert_to_mel(file_res):
    num_spectrogram_bins = file_res.num_bins
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 1000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, file_res.sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
    file_res.spectrogram_tensor, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(file_res.spectrogram_tensor.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
    file_res.spectrogram_tensor = mel_spectrograms

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    results = read_audio_files(index, directory)
    #print("Results:", results)
    convert_to_mel(results[0])
