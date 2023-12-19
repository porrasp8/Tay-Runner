
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

env = wraper.GeometryDashEnv()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA: ", torch.cuda.is_available())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Training params
BATCH_SIZE = 128
GAMMA = 0.97
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 0.0001
TAU = 0.005
LR = 1e-4

# Files paths
filepath_policy_net = "policy_net"
filepath_test_net = "test_net"
filename_rewards_log = "rewards.npy"

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
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """
    def __init__(self, output_size, h=84, w=84):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q


def load_model(net, filepath):
    try:
        checkpoint = torch.load(filepath)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print("Model found and loaded")

    except FileNotFoundError:
        print("Creating new model")


n_actions = env.action_space.n

# Creates the convolutional networks and loads the previous models
policy_net = DQN(n_actions).to(device)
load_model(policy_net, filepath_policy_net)

target_net = DQN(n_actions).to(device)
load_model(target_net, filepath_test_net)

target_net.load_state_dict(policy_net.state_dict())

# AdamW optimizer and memory declaration
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state, epsilon, temperature=1.0):
    # Update number of steps taken:
    global steps_done
    steps_done += 1
    
    # Select rand uniform num between 0 and 1
    sample = random.uniform(0, 1)
    
    # If chosen num > epsilon, explote (select most optimal action)
    if sample > epsilon:
        with torch.no_grad():
            # Apply softmax policy formula
            action_values = policy_net(state).squeeze()
            probabilities = F.softmax(action_values / temperature, dim=0)
            action = torch.multinomial(probabilities, 1).view(1, 1)
            return action
    # If not, explore:
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

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
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    '''
    height, width = grayscale_image.shape

    start_row = 0  
    end_row = height  
    start_col = int(width / 4)  
    end_col = width  

    # Half screen
    cropped_image = grayscale_image[start_row:end_row, start_col:end_col]
    '''
    cropped_image = grayscale_image
    image = cv2.resize(cropped_image, (84, 84))
    input_state = np.array(image)
    

    input_state = input_state / 255.0

    cropped = input_state.reshape((1,84,84))

    # cv2.imshow("Game Screenshot", cropped_image)
    # key = cv2.waitKey(1)

    return cropped

def main():
#    if torch.cuda.is_available():
#        num_episodes = 600
#    else:
#        num_episodes = 50

    if len(sys.argv) == 2:
        global filepath_policy_net, filepath_test_net, filename_rewards_log
        folder = sys.argv[1]
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filepath_policy_net = folder + "/" + filepath_policy_net
        filepath_test_net = folder + "/" + filepath_test_net
        filename_rewards_log = folder + "/" + filename_rewards_log

    rewards = []
    inferences = []
    lastFrames = []

    while True:
    #for i_episode in range(num_episodes):
        frame = env.reset()
        frame = transformFrame(frame)
        lastFrames = [frame, frame, frame]
        stacked = np.vstack([frame, frame, frame, frame])
        
        # Update epsilon every episode
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-EPS_DECAY * steps_done)

        frame = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Take as many steps as specify for each episode
        for t in count(): 
            
            tim = time.time()
            
            # Select best action based on softmax policy
            action = select_action(frame, epsilon)

            initTime = time.time()
            obs, rew, done = env.step(action.item())
            # print("**** Step time = ", time.time() - initTime)

            observation = transformFrame(obs)

            # Updates the frames list
            lastFrames.insert(0, observation)
            del lastFrames[-1]

            # Creates the tensor
            observation = np.vstack([lastFrames[2], lastFrames[1], lastFrames[0], observation])

            reward = torch.tensor([rew], device=device)

            if done:
                rewards.append(rew)
                next_frame = None
            else:
                next_frame = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            memory.push(frame, action, next_frame, reward)

            optimize_model()
            
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            
            frame = next_frame
            if done:
                np.save(filename_rewards_log, rewards)
                torch.save({'model_state_dict': policy_net.state_dict()}, filepath_policy_net)
                torch.save({'model_state_dict': target_net.state_dict()}, filepath_test_net)
                break
            
            inference_time = time.time() - tim
            inferences.append(inference_time)
            print("**** Inference time = ", inference_time, " |||   Inference mean = ", np.mean(inferences))

            # print("------------------------------------------")


    print('Complete')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

