import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


def generate_mic_input_spectrogram(audio):

    # Converting wav byte string to numpy array
    numpy_array = np.frombuffer(audio.get_wav_data(), dtype=np.int16) #float16

    # Replacing the NAN values with zeros
    new_array = np.nan_to_num(numpy_array).astype(np.float32)

    # Generating spectrograms
    mel_spectrogram = librosa.feature.melspectrogram(new_array, sr=audio.sample_rate, n_fft=2048, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(42 * px, 42 * px))
    plt.axis('off')
    librosa.display.specshow(log_mel_spectrogram, sr=audio.sample_rate)
    plt.set_cmap('magma')

    # Saving plot figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, pad_inches=0.0)
    buf.seek(0)
    plt.close()

    # Converting to PIL.Image np array
    image = np.array(Image.open(buf))
    buf.close()
    return image


def generate_mic_input_spectrogram_from_file():
    sample, sample_rate = librosa.load('microphone-results.wav')

    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=sample_rate, n_fft=2048, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(figsize=(42 * px, 42 * px))
    plt.axis('off')
    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate)
    plt.set_cmap('magma')
    plt.savefig('microphone-results.png', bbox_inches='tight',
                transparent=True, pad_inches=0.0)

    image = np.array(Image.open('microphone-results.png'))
    return image


def generate_spectrograms(directory):
    audio_file_names = Path(directory).rglob('*.wav')
    command = directory.split('/')[-1].strip()
    id = 1

    for audio_file_name in audio_file_names:
        sample, sr = librosa.load(audio_file_name)
        mel_spectrogram = librosa.feature.melspectrogram(sample, sr=sr, n_fft=2048, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        plt.figure(figsize=(42*px, 42*px))
        plt.axis('off')
        librosa.display.specshow(log_mel_spectrogram, sr=sr)
        plt.set_cmap('magma')
        plt.savefig('srb-sgram/{}/{}-{}.png'.format(command, id, command), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)
        plt.close()
        id += 1


if __name__ == '__main__':
    generate_spectrograms('srb-audio/skoci')
    generate_spectrograms('srb-audio/dole')
    generate_spectrograms('srb-audio/levo')
    generate_spectrograms('srb-audio/desno')
