import ctypes
import psutil
import struct
import time

class GdData:
    def __init__(self, current_frame=None, percent=None, speed=None, is_playing=None):
        self.gd_current_frame = current_frame
        self.gd_percent = percent
        self.gd_speed = speed
        self.gd_isplaying = is_playing

    def __str__(self):
        return f"GdData(current_frame={self.gd_current_frame}, advance_percent={self.gd_percent}, speed={self.gd_speed}, is_playing={self.gd_isplaying})"
       

class GdDataReader:
    def __init__(self, target_program, data_length=4):
        self.target_program = target_program
        self.data_length = data_length
        self.process = None
        self.buffer = ctypes.create_string_buffer(self.data_length)

    def obtain_pid(self):
        for proceso in psutil.process_iter(['pid', 'name']):
            if self.target_program.lower() in proceso.info['name'].lower():
                return proceso.info['pid']
        return None

    def open_process(self):
        pid = self.obtain_pid()
        if pid:
            self.process = ctypes.windll.kernel32.OpenProcess(0x10, False, pid)
        else:
            raise ValueError(f"Process '{self.target_program}' not found.")

    def read_memory(self, memory_dir, data_type='int'):
        try:
            ctypes.windll.kernel32.ReadProcessMemory(
                self.process, memory_dir, self.buffer, self.data_length, None
            )

            if data_type == 'int':
                value = int.from_bytes(self.buffer.raw, byteorder='little')
            elif data_type == 'float':
                value = struct.unpack('f', self.buffer.raw)[0]
            else:
                raise ValueError("Invalid data_type. Use 'int' or 'float'.")

            return value
        
        except Exception as e:
            print(f"Error reading memory: {e}")
            return None

    def close_process(self):
        if self.process:
            ctypes.windll.kernel32.CloseHandle(self.process)