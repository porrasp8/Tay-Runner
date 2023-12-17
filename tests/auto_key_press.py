import keyboard
import time
import psutil
import pygetwindow as gw

KEY_TO_PRESS = 'space'
TIME_INTERVAL = 1
TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'

def key_press_simulation(key):
    keyboard.press(key)
    keyboard.release(key)

def is_program_running_and_active(program_name):
    for process in psutil.process_iter(['pid', 'name']):
        if program_name.lower() in process.info['name'].lower():
            active_window = gw.getActiveWindow()
            if active_window.title == WINDOW_NAME:
                return True
    return False

def main():
    try:
        while True:
            if is_program_running_and_active(TARGET_PROGRAM):
                key_press_simulation(KEY_TO_PRESS)
                print(TARGET_PROGRAM, " running")
            else:
                print(TARGET_PROGRAM, " not running")

            time.sleep(TIME_INTERVAL)

    except KeyboardInterrupt:
        print("Finish")


if __name__ == "__main__":
    main()