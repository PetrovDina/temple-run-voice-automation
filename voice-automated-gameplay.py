# NOTE: this example requires PyAudio because it uses the Microphone class
import time
import numpy as np
import speech_recognition as sr
from predict import predict_voice_input
from keymapper import run_command
from spectrogram_generator import generate_mic_input_spectrogram


# this is called from the background thread
def callback(recognizer, audio):
    image = generate_mic_input_spectrogram(audio)
    predicted_command, percentage = predict_voice_input(np.array([image]))
    if percentage > 0.8:
        run_command(predicted_command)


if __name__ == '__main__':
    print(" ======================== T E M P L E  R U N ========================")
    print("Configuring microphone.. Please wait.")
    r = sr.Recognizer()
    m = sr.Microphone()

    with m as source:
        r.adjust_for_ambient_noise(source)

    stop_listening = r.listen_in_background(m, callback, phrase_time_limit=2)
    print("Microphone listening... Start talking!")

    # calling this function requests that the background listener stop listening
    # stop_listening(wait_for_stop=False) # Commenting this out because we don't want listening to stop

    # endless loop (because we don't want the main thread to end as it would also kill the listening thread)
    while True:
        time.sleep(0.1)



