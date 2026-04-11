import scikit-learn
import tensorflow as tf
import tensorflow_io as tfio

if __name__ == "main":
    audio_bytes = tf.io.read_file("data/99-22A_airplane_01.wav")
    audio, sample_rate = tf.audio.decode_wav(audio_bytes)


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


