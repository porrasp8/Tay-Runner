import pygame
import sys
import random
import gym
from gym import spaces
import numpy as np
from gd_data import GdDataReader
from gd_data import GdData
from gd_control import GdControl

TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'
DATA_LENGTH = 4
MONITOR_INDEX = 1
JUMP_PENALTY = 0.2

#-- Check it with CheatEngine
memory_addresses = {
    'gd_frame': 0x6B7DBDC0,
    'gd_percent': 0x6B7DBD80,
    'gd_speed': 0x6B7DB780,
}


class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        self.gd_data_reader = GdDataReader(TARGET_PROGRAM, WINDOW_NAME, DATA_LENGTH)
        self.gd_controller = GdControl()
        self.last_frame = 0

        self.gd_data_reader.open_process()

    def step(self, action):

        done = False
        img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)
        reward = self.gd_data_reader.read_memory(memory_addresses['gd_percent'], 'float')
        current_frame = self.gd_data_reader.read_memory(memory_addresses['gd_frame'], 'int')

        done = current_frame < self.last_frame
        self.last_frame = current_frame
        
        if action == 1:
            reward -= JUMP_PENALTY
            self.gd_controller.jump()
        
        return img, reward, done

    def reset(self):
        self.last_frame = 0
        img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)
        return img

    def close(self):
        self.gd_data_reader.close_process()
        sys.exit()