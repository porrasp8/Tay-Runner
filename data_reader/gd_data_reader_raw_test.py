import ctypes
import psutil
import time

TARGET_PROGRAM = 'GeometryDash.exe'
MEMORY_DIR = 0x6B71BDC0
DATA_LENGHT = 4
TIME_INTERVAL = 0.1

#-- Extract pid of the 'pname' procces
def obtein_pid(pname):
    for proceso in psutil.process_iter(['pid', 'name']):
        if pname.lower() in proceso.info['name'].lower():
            return proceso.info['pid']

    return None

def main():

    #-- Pid extraction
    ppid = obtein_pid(TARGET_PROGRAM)

    #-- Procces opening and buffer init
    process = ctypes.windll.kernel32.OpenProcess(0x10, False, ppid)
    buffer = ctypes.create_string_buffer(DATA_LENGHT)

    try:
        while True:
            #-- Read from memory and decode it
            ctypes.windll.kernel32.ReadProcessMemory(process, MEMORY_DIR, buffer, DATA_LENGHT, None)
            value = int.from_bytes(buffer.raw, byteorder='little')
            print(f"Data read in memory direction {hex(MEMORY_DIR)}: {value}")
            time.sleep(TIME_INTERVAL)

    except KeyboardInterrupt:
        #-- Close handle and exit
        ctypes.windll.kernel32.CloseHandle(process)
        print("Finish")

if __name__ == "__main__":
    main()