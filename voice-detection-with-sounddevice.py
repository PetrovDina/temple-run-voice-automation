# Print out realtime audio volume as ascii bars

# ----------------------------------------------------------------------
# Primer koji radi 100 sekundi i za to vreme sve vreme slusa
# Kada mic input bude veci od 100, tada ispise da je glas detektovan
# 100 je dobra vrednost jer je ambijentalan zvuk i sum obicno mnogo manji (0-10), a kada se prica direkt u mikrofon budu veci brojevi (100-)
# ----------------------------------------------------------------------

import sounddevice as sd
import numpy as np


def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    # print ("|" * int(volume_norm))
    print (volume_norm)
    if volume_norm > 100:
        print("voice detected")


with sd.Stream(callback=print_sound):
    sd.sleep(100000) # posle koliko da stane, sada je na 100 sekundi