from main_structures import FourierProperties

def reconstruct_audio_tensor(audio_orig: tf.Tensor, new_spectrogram: tf.Tensor) -> tf.Tensor:
    # Original audio
    global frame_length, frame_step, fft_length
    stft_original = tf.signal.stft(audio_orig,
        frame_length=FourierProperties.frame_length,
        frame_step=FourierProperties.frame_step,
        fft_length=FourierProperties.fft_length
    )

    # Modified spectrogram (mel, filtered, etc.)
    modified_magnitude = tf.abs(new_spectrogram)  # [257, time_frames]

    # Reconstruct with ORIGINAL phase:
    stft_modified = modified_magnitude * tf.exp(1j * tf.angle(stft_original))

    # Back to audio:
    audio_reconstruct = tf.signal.inverse_stft(
        stft_modified,
        frame_length=FourierProperties.frame_length,
        frame_step=FourierProperties.frame_step,
        fft_length=FourierProperties.fft_length
    )
    return audio_reconstruct

def save_reconstructed_audio(audio_tensor: tf.Tensor, sample_rate: int, filename: str):
    # Ensure mono float32 [-1.0, 1.0] and correct dtype
    audio_mono = tf.squeeze(audio_tensor, axis=-1)  # Remove channels if stereo
    audio_mono = tf.cast(audio_mono, tf.float32)
    
    # Encode as 16-bit PCM WAV
    wav_bytes = tf.audio.encode_wav(audio_mono, sample_rate)
    
    # Save to file
    tf.io.write_file(filename, wav_bytes)