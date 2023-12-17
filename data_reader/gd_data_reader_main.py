from gd_data import GdDataReader
from gd_data import GdData
import time
import cv2

TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'
DATA_LENGTH = 4
TIME_INTERVAL = 0.1
MONITOR_INDEX = 1

#-- Check it with CheatEngine
memory_addresses = {
    'gd_frame': 0x6B7DBDC0,
    'gd_percent': 0x6B7DBD80,
    'gd_speed': 0x6B7DB780,
}

def main():

    #-- Objects init
    gd_data = GdData()
    gd_data_reader = GdDataReader(TARGET_PROGRAM, WINDOW_NAME, DATA_LENGTH)

    last_frame = 0

    try:
        gd_data_reader.open_process()

        while True:
            #-- Data reading
            gd_data.gd_current_frame = gd_data_reader.read_memory(memory_addresses['gd_frame'], 'int')
            gd_data.gd_percent = gd_data_reader.read_memory(memory_addresses['gd_percent'], 'float')
            gd_data.gd_speed = gd_data_reader.read_memory(memory_addresses['gd_speed'], 'float')
            screenshot_bgr =  gd_data_reader.capture_game_image(MONITOR_INDEX)

            #-- Check if is playing
            gd_data.gd_isplaying = gd_data.gd_current_frame != last_frame
            last_frame = gd_data.gd_current_frame

            #-- Img show
            if screenshot_bgr is not None:
                cv2.imshow("Game Screenshot", screenshot_bgr)
                key = cv2.waitKey(1)
                if key == 27:  #-- ESC
                    break

            #-- Print data        
            print(gd_data)

            time.sleep(TIME_INTERVAL)

    except KeyboardInterrupt:
        print("Finish")
        cv2.destroyAllWindows()

    finally:
        gd_data_reader.close_process()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()