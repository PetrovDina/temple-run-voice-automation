from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_mic_input_spectrogram():
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
        plt.figure(figsize=(161*px, 162*px))
        plt.axis('off')
        librosa.display.specshow(log_mel_spectrogram, sr=sr)
        plt.set_cmap('magma')
        plt.savefig('speech-commands-sgram-124x124/{}/{}-{}.png'.format(command, id, command), bbox_inches='tight',
                    transparent=True, pad_inches=0.0)
        id += 1


if __name__ == '__main__':
    generate_spectrograms('speech-commands-audio/up')
    generate_spectrograms('speech-commands-audio/down')
    generate_spectrograms('speech-commands-audio/left')
    generate_spectrograms('speech-commands-audio/right')
