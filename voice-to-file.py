#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Primer koji jednom ukljuci mikrofon i sacuva u fajl detektovani audio
# Snima dok god pricas i tek onda prekine i sacuva
# Instaliranje PyAudio -> otvoriti cmd u rootu projekta i kucati "pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl"
# ----------------------------------------------------------------------

# NOTE: this example requires PyAudio because it uses the Microphone class


import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    # audio = r.adjust_for_ambient_noise(source)

# write audio to a WAV file
with open("saved-mic-input/microphone-results.wav", "wb") as f:
    f.write(audio.get_wav_data())

# ovde dodati kod za pretvaranje u spektogram

