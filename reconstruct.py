from pathlib import Path

import tensorflow as tf

from main_structures import FileResult, FourierProperties

def reconstruct_audio_tensor(file_props: FileResult, new_spectrogram: tf.Tensor) -> tf.Tensor:
    # Original audio
    stft_original = tf.signal.stft(file_props.audio_tensor,
        frame_length=file_props.frame_length,
        frame_step=file_props.frame_step,
        fft_length=file_props.fft_length
    )

    # Modified spectrogram (do NOT pass mel-transformed spectrogram here)
    modified_magnitude = tf.abs(new_spectrogram)  # [257, time_frames]

    # Reconstruct with ORIGINAL phase:
    angle = tf.math.angle(stft_original)
    phase = tf.exp(tf.complex(tf.zeros_like(angle), angle))
    modified_magnitude_complex = tf.cast(modified_magnitude, phase.dtype)
    stft_modified = modified_magnitude_complex * phase

    # Back to audio:
    audio_reconstruct = tf.signal.inverse_stft(
        stft_modified,
        frame_length=file_props.frame_length,
        frame_step=file_props.frame_step,
        fft_length=file_props.fft_length
    )
    return audio_reconstruct

def save_reconstructed_audio(audio_tensor: tf.Tensor, sample_rate: int, filename: str):
    # Ensure 2D [T, 1]
    if len(audio_tensor.shape) == 1:
        audio_2d = tf.reshape(audio_tensor, [-1, 1])
    else:
        audio_2d = tf.reshape(tf.reduce_mean(audio_tensor, axis=-1), [-1, 1])

    audio_2d = tf.cast(audio_2d, tf.float32)

    wav_bytes = tf.audio.encode_wav(audio_2d, sample_rate)
    tf.io.write_file(filename, wav_bytes)


if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"
    
    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    
    spectrogram_mult = tf.math.scalar_mul(2.0, result.spectrogram_tensor)
    audio_reconstruct = reconstruct_audio_tensor(result, spectrogram_mult)
    output = Path(__file__).parent / "test" / "1989-06_airplane_01.wav"
    save_reconstructed_audio(audio_reconstruct, result.sample_rate, str(output))

if __name__ == "__main__":
    directory = Path(__file__).parent / "data"
    index = Path(__file__).parent / "index.csv"
    test_file = "1989-06_airplane_01.wav"

    from parse_audio import read_specific_audio_file
    result = read_specific_audio_file(index, directory, test_file)
    
    spectrogram_mult = tf.math.scalar_mul(2.0, result.spectrogram_tensor)
    audio_reconstruct = reconstruct_audio_tensor(result, spectrogram_mult)
    output = Path(__file__).parent / "test" / "1989-06_airplane_01.wav"
    save_reconstructed_audio(audio_reconstruct, result.sample_rate, str(output))
    
    # Test the reconstruction by reading it and plotting its spectrogram
    from graph import graph_spectrogram
    new_result = read_specific_audio_file(index, Path(__file__).parent / "test", test_file)
    # Centered on selection 19 (rumble from 15.1885 to 21.89300667 seconds)
    graph_spectrogram(new_result, 1000, 10, 25)
