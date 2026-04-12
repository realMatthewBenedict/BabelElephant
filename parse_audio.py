from pathlib import Path

import pandas as pd
import tensorflow as tf

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

def read_audio_files(index_path: Path, directory: Path) -> list[FileResult]:
    df = pd.read_csv(str(index_path), index_col="Selection")
    result: dict[str, FileResult] = {}

    # Only read files with indexed rumbles
    for index, row in df.iterrows():
        file_name = row["Sound_file"]

        # Assume CSV is sorted by file
        if file_name in result:
            sample, frame = convert_time(row,
                result[file_name].audio_tensor,
                result[file_name].sample_rate,
                FourierProperties.frame_step
            )
            result[file_name].rumble_samples.append(sample)
            result[file_name].rumble_frames.append(frame)
            continue

        file_path = directory / file_name
        audio_bytes = tf.io.read_file(str(file_path))
        audio, sample_rate = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=-1)
        sample_rate_int = sample_rate.numpy()

        stft = tf.signal.stft(audio,
            frame_length=FourierProperties.frame_length,
            frame_step=FourierProperties.frame_step,
            fft_length=FourierProperties.fft_length
        )
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
        
        sample, frame = convert_time(row, audio, sample_rate_int, FourierProperties.frame_step)

        result[file_name] = FileResult(
            file_name=file_name,
            audio_tensor=audio,
            spectrogram_tensor=spectrogram, 
            spectrogram_bins=FourierProperties.fft_length // 2 + 1,
            sample_rate=sample_rate_int,
            category=category,
            rumble_samples=[sample],
            rumble_frames=[frame],
            call_type=row["Call_type"]
        )
    return list(result.values())

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