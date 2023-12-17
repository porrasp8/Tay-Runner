import keyboard
import time
import psutil
import pygetwindow as gw

KEY_TO_PRESS = 'space'
TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'

class GdControl:
    def __init__(self ,jump_key = KEY_TO_PRESS):
        self.jump_key = jump_key

    def jump(self):
        if self._is_program_running_and_active(TARGET_PROGRAM, WINDOW_NAME):
            self._key_press_simulation(self.jump_key)

    @staticmethod
    def _key_press_simulation(key):
        keyboard.press(key)
        keyboard.release(key)

    @staticmethod
    def _is_program_running_and_active(program_name, window_name):
        for process in psutil.process_iter(['pid', 'name']):
            if program_name.lower() in process.info['name'].lower():
                active_window = gw.getActiveWindow()
                if active_window.title == window_name:
                    return True
        return False
