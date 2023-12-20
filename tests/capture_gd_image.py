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

    
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        screen = {
            "top": monitor["top"],  # 100px from the top
            "left": monitor["left"] + monitor["width"] // 3,  # 100px from the left
            "width": monitor["width"] * 2 // 3,
            "height": monitor["height"],
            "mon": monitor_index,
        }
        img = np.array(sct.grab(screen))

        return img
    

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
