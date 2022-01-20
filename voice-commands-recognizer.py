
# ----------------------------------------------------------------------
# Instaliranje PyAudio -> otvoriti cmd u rootu projekta i kucati "pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl"
# ----------------------------------------------------------------------

# NOTE: this example requires PyAudio because it uses the Microphone class

import time
import speech_recognition as sr


# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    # TODO pozvati našu metodu za generisanje spektograma od detektovanog zvuka
    #  i proslediti ga istreniranom modelu na predikciju komande
    try:
        print("CALLBACK")
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
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