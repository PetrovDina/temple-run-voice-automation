from pynput import keyboard
from pynput.keyboard import Key, Controller
import time


def keyboard_test():

    controller = Controller()
    print(dir(Key))  # listing available keys

    # To try out the game, open https://poki.com/en/g/temple-run-2#, and wait for game to load
    # Run this script, and then you have 3 seconds to switch focus on the game area
    # Do this by clicking anywhere on the game area

    print("Switch to temple run window now by clicking on the game area!")
    time.sleep(3) # time to switch to temple run window
    print("Starting ga me")
    controller.press(keyboard.Key.space)

    for x in range(10): # jump 10 times in 1 second intervals
        controller.press(keyboard.Key.up)
        controller.release(keyboard.Key.up)
        time.sleep(1)


if __name__ == '__main__':
    keyboard_test()
