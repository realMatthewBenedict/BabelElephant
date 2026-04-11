from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
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
    spectrogram_tensor: tf.Tensor     # [257, time_frames]
    spectrogram_bins: int             # 257
    sample_rate: int
    category: Category
    rumble_samples: list[tuple[int, int]] # corresponds to audio_tensor
    rumble_frames: list[tuple[int, int]] # corresponds to spectrogram_tensor
    call_type: str

def convert_time(row, sample_rate_int, frame_step) -> tuple[tuple]:
    # Note: Samples of audio, not spectrogram
    start_sample=int(row["Start_time"] * sample_rate_int)
    end_sample=int(row["End_time"] * sample_rate_int)

    # Note: Frames of spectrogram, not audio
    start_frame = int(start_sample / frame_step)
    end_frame = int(end_sample / frame_step)

    return ((start_sample, end_sample), (start_frame, end_frame))

def read_audio_files(index_path: Path, directory: Path) -> list[FileResult]:
    df = pd.read_csv(str(index_path), index_col="Selection")
    result = []

    for index, row in df.iterrows():
        file_name = row["Sound_file"]
        frame_step = 256
        fft_length = 512

        # Assume CSV is sorted by file
        if len(result) > 0 and result[-1].file_name == file_name:
            sample, frame = convert_time(row, result[-1].sample_rate, frame_step)
            result[-1].rumble_samples.append(sample)
            result[-1].rumble_frames.append(frame)

        file_path = directory / file_name
        audio_bytes = tf.io.read_file(str(file_path))
        audio, sample_rate = tf.audio.decode_wav(audio_bytes)
        sample_rate_int = sample_rate.numpy()

        stft = tf.signal.stft(audio, frame_length=512, frame_step=frame_step, fft_length=fft_length)
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
        
        sample, frame = convert_time(row, sample_rate_int, frame_step)

        result.append(FileResult(
            file_name=file_name,
            audio_tensor=audio,
            spectrogram_tensor=spectrogram, 
            spectrogram_bins=fft_length // 2 + 1,
            sample_rate=sample_rate_int,
            category=category,
            rumble_samples=[sample],
            rumble_frames=[frame],
            call_type=row["Call_type"]
        ))
    return result

def reconstruct_audio_tensor(audio_orig: tf.Tensor, new_spectrogram: tf.Tensor) -> tf.Tensor:
    # Original audio
    stft_original = tf.signal.stft(audio_orig, frame_length=512, frame_step=256, fft_length=512)

    # Modified spectrogram (mel, filtered, etc.)
    modified_magnitude = tf.abs(new_spectrogram)  # [257, time_frames]

    # Reconstruct with ORIGINAL phase:
    stft_modified = modified_magnitude * tf.exp(1j * tf.angle(stft_original))

    # Back to audio:
    audio_reconstruct = tf.signal.inverse_stft(
        stft_modified, frame_length=512, frame_step=256, fft_length=512
    )
    return audio_reconstruct

def save_reconstructed_audio(audio_tensor: tf.Tensor, sample_rate: int, filename: str):
    # Ensure mono float32 [-1.0, 1.0] and correct dtype
    audio_mono = tf.squeeze(audio_tensor, axis=-1)  # Remove channels if stereo
    audio_mono = tf.cast(audio_mono, tf.float32)
    
    # Encode as 16-bit PCM WAV
    wav_bytes = tf.audio.encode(
        audio_mono, 
        sample_rate, 
        name='encoded_wav'
    )
    
    # Save to file
    tf.io.write_file(filename, wav_bytes)

def convert_to_mel(file_res: FileResult):
    num_spectrogram_bins = file_res.spectrogram_bins
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
