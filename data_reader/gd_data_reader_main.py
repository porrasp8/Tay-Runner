from gd_data import GdDataReader
from gd_data import GdData
import time

TARGET_PROGRAM = 'GeometryDash.exe'
DATA_LENGTH = 4
TIME_INTERVAL = 0.1

memory_addresses = {
    'gd_frame': 0x6B7DBDC0,
    'gd_percent': 0x6B7DBD80,
    'gd_speed': 0x6B7DB780,
}

def main():

    gd_data = GdData()
    gd_data_reader = GdDataReader(TARGET_PROGRAM, DATA_LENGTH)

    last_frame = 0

    try:
        gd_data_reader.open_process()

        while True:
            gd_data.gd_current_frame = gd_data_reader.read_memory(memory_addresses['gd_frame'], 'int')
            gd_data.gd_percent = gd_data_reader.read_memory(memory_addresses['gd_percent'], 'float')
            gd_data.gd_speed = gd_data_reader.read_memory(memory_addresses['gd_speed'], 'float')

            #-- Check if is playing
            gd_data.gd_isplaying = gd_data.gd_current_frame != last_frame
            last_frame = gd_data.gd_current_frame

            if (gd_data.gd_current_frame is not None and 
                gd_data.gd_speed is not None):
                print(gd_data)

            time.sleep(TIME_INTERVAL)

    except KeyboardInterrupt:
        print("Finish")

    finally:
        gd_data_reader.close_process()

if __name__ == "__main__":
    main()