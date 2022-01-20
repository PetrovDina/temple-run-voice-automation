
# ----------------------------------------------------------------------
# Instaliranje PyAudio -> otvoriti cmd u rootu projekta i kucati "pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl"
# ----------------------------------------------------------------------

# NOTE: this example requires PyAudio because it uses the Microphone class

import time
import speech_recognition as sr

import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from keras.models import load_model

from pynput.keyboard import Key, Controller

MODEL = load_model('test-1.h5')

COMMANDS = ['up', 'down', 'left', 'right']

CONTROLLER = Controller()


# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    # TODO pozvati našu metodu za generisanje spektograma od detektovanog zvuka
    #  i proslediti ga istreniranom modelu na predikciju komande
    try:
        print("CALLBACK")
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))

        # TODO -------------------------------- SPORO I KRITICNO -------------------------------------------------------------------
        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())

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
        # ----------------------------------------------------------------------------------------------------------------------

        image = np.array(Image.open('microphone-results.png'))
        pred = MODEL.predict(np.array([image]))
        detected_command = COMMANDS[np.argmax(pred, axis=1)[0]]
        print(detected_command)
        if detected_command == 'up':
            CONTROLLER.press(Key.up)
            return
        if detected_command == 'down':
            CONTROLLER.press(Key.down)
            return
        if detected_command == 'left':
            CONTROLLER.press(Key.left)
            return
        if detected_command == 'right':
            CONTROLLER.press(Key.right)
            return

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


r = sr.Recognizer()
m = sr.Microphone()

with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback, phrase_time_limit=3)
# `stop_listening` is now a function that, when called, stops background listening

# calling this function requests that the background listener stop listening
# stop_listening(wait_for_stop=False) # Zakomentarisano: mi necemo da se zaustavi slušanje

# endless loop (because we don't want the main thread to end as it would also kill the listening thread)
while True:
    time.sleep(0.1)