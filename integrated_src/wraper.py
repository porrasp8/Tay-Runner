import sys
import random
import gym
from gym import spaces
import numpy as np
from gd_data import GdDataReader
from gd_data import GdData
from gd_control import GdControl
import time

TARGET_PROGRAM = 'GeometryDash.exe'
WINDOW_NAME = 'Geometry Dash'
DATA_LENGTH = 4
MONITOR_INDEX = 1
JUMP_PENALTY = 1.0

#-- Check it with CheatEngine
memory_addresses = {
    'gd_frame': 0x5BADBDC0,
    'gd_percent': 0x5BADBD80,
    'gd_speed': 0x5BADB780,
}


class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        self.gd_data_reader = GdDataReader(TARGET_PROGRAM, WINDOW_NAME, DATA_LENGTH)
        self.gd_controller = GdControl()
        self.last_frame = 0
        self.action_space = spaces.Discrete(2)  # actions: jump or stay still

        self.gd_data_reader.open_process()

    def step(self, action):

        done = False

        
        reward = self.gd_data_reader.read_memory(memory_addresses['gd_percent'], 'float')
        current_frame = self.gd_data_reader.read_memory(memory_addresses['gd_frame'], 'int')
        img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)


        #-- Check if playing
        playing = current_frame != self.last_frame

        
        while (img is None or not playing):
            img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)
            current_frame = self.gd_data_reader.read_memory(memory_addresses['gd_frame'], 'int')
            playing = current_frame != self.last_frame
            #time.sleep(1)
        
        #-- Check death
        done = current_frame < self.last_frame
        self.last_frame = current_frame
        
        if action == 1:
            reward -= JUMP_PENALTY
            self.gd_controller.jump()
        
        # print("done: " + str(done) + ", reward: " + str(reward) + ", current_frame: " + str(current_frame) + ", last frame: " + str(self.last_frame) + ", playing: " + str(playing))
        
        return img, reward, done

    def reset(self):
        self.last_frame = 0
        img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)
        while img is None:
            img =  self.gd_data_reader.capture_game_image(MONITOR_INDEX)
            #time.sleep(1)

        return img

    def close(self):
        self.gd_data_reader.close_process()
        sys.exit()