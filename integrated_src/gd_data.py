import ctypes
import psutil
import struct
import mss
import numpy as np
import cv2
import pygetwindow as gw

class GdData:
    def __init__(self, current_frame=None, percent=None, speed=None, is_playing=None):
        self.gd_current_frame = current_frame
        self.gd_percent = percent
        self.gd_speed = speed
        self.gd_isplaying = is_playing

    def __str__(self):
        return f"GdData(current_frame={self.gd_current_frame}, advance_percent={self.gd_percent}, speed={self.gd_speed}, is_playing={self.gd_isplaying})"
       

class GdDataReader:
    def __init__(self, target_program, window_name, data_length=4):
        self.target_program = target_program
        self.window_name = window_name
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
    
    def capture_game_image(self, monitor_index):

        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index]
            screen = {
                "top": monitor["top"],  # 100px from the top
                "left": monitor["left"] + monitor["width"] // 3,  # 100px from the left
                "width": monitor["width"] * 2 // 4,
                "height": monitor["height"],
                "mon": monitor_index,
            }
            img = np.array(sct.grab(screen))

            return img

    @staticmethod
    def _is_program_running_and_active(program_name, window_name):
        for process in psutil.process_iter(['pid', 'name']):
            if program_name.lower() in process.info['name'].lower():
                active_window = gw.getActiveWindow()
                if active_window.title == window_name:
                    return True
        return False