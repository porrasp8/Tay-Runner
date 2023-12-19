import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wraper
import numpy as np
import os
import time
import sys
from algoritmoQLearning import DQN, ReplayMemory, load_model

env = wraper.GeometryDashEnv()


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Training params
BATCH_SIZE = 256 #128
GAMMA = 0.95 # 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1
TAU = 0.005
LR = 1e-4     # 1e-5

# Files paths
filepath_policy_net = "./policy_net"
filepath_test_net = "./test_net"
filename_rewards_log = "rewards.npy"



n_actions = env.action_space.n

# Creates the convolutional networks and loads the previous models
policy_net = DQN(n_actions).to(device)
load_model(policy_net, filepath_policy_net)

target_net = DQN(n_actions).to(device)
load_model(target_net, filepath_test_net)

target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)






def transformFrame(frame):
    frame = np.array(frame)
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = grayscale_image.shape

    start_row = 0  
    end_row = height  
    start_col = int(width / 4)  
    end_col = width  

    # Half screen
    cropped_image = grayscale_image[start_row:end_row, start_col:end_col]

    image = cv2.resize(cropped_image, (84, 84))
    input_state = np.array(image)
    

    input_state = input_state / 255.0

    cropped = input_state.reshape((1,84,84))
    stacked = np.vstack([cropped, cropped, cropped, cropped])
    return stacked


def main():
    if torch.cuda.is_available():
        num_episodes = 40
    else:
        num_episodes = 10


    for i_episode in range(num_episodes):
        frame = env.reset()
        frame = torch.tensor(transformFrame(frame), dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(frame)
            obs, rew, done = env.step(action.item())
            observation = transformFrame(obs)

            reward = torch.tensor([rew], device=device)

            if done:
                next_frame = None
            else:
                next_frame = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            memory.push(frame, action, next_frame, reward)


            frame = next_frame
            if done:
                break


    print('Complete')


if __name__ == "__main__":
    main()

