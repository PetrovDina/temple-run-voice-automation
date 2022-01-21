from pynput.keyboard import Key, Controller

COMMANDS = ['up', 'down', 'left', 'right']
KEYS = [Key.up, Key.down, Key.left, Key.right]
CONTROLLER = Controller()


def run_command(command_index):
    CONTROLLER.press(KEYS[command_index])
    CONTROLLER.release(KEYS[command_index])
    print(COMMANDS[command_index])



