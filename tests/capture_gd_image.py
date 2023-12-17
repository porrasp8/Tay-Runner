import cv2
import mss
import numpy as np
import time
import psutil
import pygetwindow as gw

TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'
MONITOR_INDEX = 1

def is_program_running_and_active(program_name, window_name):
    for process in psutil.process_iter(['pid', 'name']):
        if program_name.lower() in process.info['name'].lower():
            active_window = gw.getActiveWindow()
            if active_window.title == window_name:
                return True
    return False

def capture_game_image(monitor_index):

    if(is_program_running_and_active(TARGET_PROGRAM, WINDOW_NAME)):
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index]
            img = np.array(sct.grab(monitor))

            return img
        
    return None


def main():
    while True:

        screenshot_bgr =  capture_game_image(MONITOR_INDEX)
        
        if screenshot_bgr is not None:
            cv2.imshow("Game Screenshot", screenshot_bgr)
            key = cv2.waitKey(1)
            if key == 27:  #-- ESC
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
