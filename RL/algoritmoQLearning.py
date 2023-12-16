
import gymnasium as gym
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
import juego
import numpy as np

env = juego.GeometryDashEnv()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change the channel order (from [1, 84, 84, 1] to [1, 1, 84, 84])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)




BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action():
    return torch.tensor([[random.choice([0, 1])]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                      if s is not None])
    
    # Convert    to tensor
    state_batch = torch.cat(batch.state)
    next_state = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



def transformFrame(frame):
    frame = np.array(frame)
    grayscale_image = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    image = cv2.resize(grayscale_image, (84, 84))  # Cambiar tamaño a 84x84 píxeles
    input_state = np.array(image)
    
    cv2.imshow("Sheep", input_state)
    cv2.waitKey(1)
    plt.show()
    input_state = input_state / 255.0

    return input_state


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50
    

for i_episode in range(num_episodes):
    frame = env.reset()
    frame = torch.tensor(transformFrame(frame), dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action()
        img_raw, reward, done = env.step(action.item())

        next_frame = None if done else torch.tensor(transformFrame(img_raw), dtype=torch.float32, device=device).unsqueeze(0)

        reward = torch.tensor([reward], device=device)
        print(len(frame), action, len(next_frame), reward)
        memory.push(frame, action, next_frame, reward)

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        frame = next_frame
        if done:
            break


print('Complete')
plt.ioff()
plt.show()

