from pathlib import Path

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from main_structures import Category, FourierProperties, FileResult

def convert_time(row: pd.Series, audio_tensor: tf.Tensor, sample_rate_int: int, frame_step: int) -> tuple[tuple[int, int], tuple[int, int]]:
    # Note: Samples of audio, not spectrogram
    start_sample=int(row["Start_time"] * sample_rate_int)
    end_sample=int(row["End_time"] * sample_rate_int)
    end_sample = min(end_sample, audio_tensor.shape[0] - 1)  # clip to waveform length

    # Note: Frames of spectrogram, not audio
    start_frame = int(start_sample / frame_step)
    end_frame = int(end_sample / frame_step)

    return ((start_sample, end_sample), (start_frame, end_frame))

def get_spectrogram(audio_tensor: tf.Tensor, sample_rate: int) -> tf.Tensor:
    frame_step = FourierProperties.frame_step_for_sample_rate(sample_rate)
    stft = tf.signal.stft(audio_tensor,
        frame_length=FourierProperties.frame_length,
        frame_step=frame_step,
        fft_length=FourierProperties.fft_length
    )
    return tf.abs(stft)

def read_audio_file_internal(result: dict[str, FileResult], row: pd.Series, directory: Path) -> None:
    file_name = row["Sound_file"]

    if file_name in result:
        sample, frame = convert_time(row,
            result[file_name].audio_tensor,
            result[file_name].sample_rate,
            result[file_name].frame_step
        )
        # These arrays will be sorted in the same order incoming file is sorted
        result[file_name].rumble_samples.append(sample)
        result[file_name].rumble_frames.append(frame)
        return

    file_path = directory / file_name
    audio_bytes = tf.io.read_file(str(file_path))
    audio, sample_rate = tf.audio.decode_wav(audio_bytes)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate_int = sample_rate.numpy()

    frame_step = FourierProperties.frame_step_for_sample_rate(sample_rate_int)
    spectrogram = get_spectrogram(audio, sample_rate_int)
    sample, frame = convert_time(row, audio, sample_rate_int, frame_step)

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

    result[file_name] = FileResult(
        file_name=file_name,
        audio_tensor=audio,
        spectrogram_tensor=spectrogram,
        sample_rate=sample_rate_int,
        category=category,
        rumble_samples=[sample],
        rumble_frames=[frame],
        call_type=row["Call_type"],
        frame_length=FourierProperties.frame_length,
        frame_step=frame_step,
        fft_length=FourierProperties.fft_length
    )

def read_audio_files(index_path: Path, directory: Path) -> dict[str, FileResult]:
    df = pd.read_csv(str(index_path), index_col="Selection")
    results: dict[str, FileResult] = {}

    # Only read files with indexed rumbles
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Reading audio files"):
        # Mutate result dictionary
        read_audio_file_internal(results, row, directory)
    return results

def read_specific_audio_file(index_path: Path, directory: Path, basename: str) -> FileResult:
    df = pd.read_csv(str(index_path), index_col="Selection")
    result: dict[str, FileResult] = {}

    # Only read files with indexed rumbles
    for index, row in df.iterrows():
        # Ignore other files
        if row["Sound_file"] != basename:
            continue

        # Mutate result dictionary
        read_audio_file_internal(result, row, directory)
    return result[basename]

def convert_to_mel(file_res: FileResult):
    """A lossy conversion to mel-scale spectrogram"""
    num_spectrogram_bins = file_res.spectrogram_bins()
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 1000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, file_res.sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        file_res.spectrogram_tensor, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(file_res.spectrogram_tensor.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    file_res.spectrogram_tensor = mel_spectrograms
