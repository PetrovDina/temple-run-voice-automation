from pynput.keyboard import Key, Controller
from common import COMMANDS

KEYS = [Key.up, Key.down, Key.left, Key.right]
CONTROLLER = Controller()


def run_command(command_index):
    CONTROLLER.press(KEYS[command_index])
    CONTROLLER.release(KEYS[command_index])
    print("pressing: " + COMMANDS[command_index])



