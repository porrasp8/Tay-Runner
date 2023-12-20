from gd_control import GdControl
import time

TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'
DATA_LENGTH = 4
TIME_INTERVAL = 1.0
MONITOR_INDEX = 1


def main():

    #-- Objects init
    gd_controller = GdControl()

    try:
        while True:
            gd_controller.jump()
            time.sleep(TIME_INTERVAL)

    except KeyboardInterrupt:
        print("Finish")


if __name__ == "__main__":
    main()