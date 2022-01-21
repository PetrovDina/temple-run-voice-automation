from pynput import keyboard
from pynput.keyboard import Key, Controller
import time

COMMANDS = ['up', 'down', 'left', 'right']
controller = Controller()


def run_command(command_index):
    detected_command = COMMANDS[command_index]
    print("PRESSING: " + str(detected_command))

    if detected_command == 'up':
        controller.press(Key.up)
        controller.release(Key.up)

    elif detected_command == 'down':
        controller.press(Key.down)
        controller.release(Key.down)

    elif detected_command == 'left':
        controller.press(Key.left)
        controller.release(Key.left)

    elif detected_command == 'right':
        controller.press(Key.right)
        controller.release(Key.right)
    else:
        print("UNKNOWN COMMAND KEY PRESS")
