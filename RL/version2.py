import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import sys
from gym import spaces

# Define el modelo DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define la clase del agente DQN
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(output_size)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        target_q = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        loss = nn.MSELoss()(q_values[0][action], target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Función principal de entrenamiento
def train_dqn(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, next_state, reward, done)
            total_reward += reward
            state = next_state

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % 10 == 0:
            agent.update_target_model()

        print(f"Episodio {episode}, Recompensa total: {total_reward}")

    # Guarda el modelo entrenado
    torch.save(agent.model.state_dict(), "dqn_model.pth")

if __name__ == "__main__":
    env = GeometryDashEnv()
    input_size = env.observation_space.shape[0]  # Ajusta según tu entorno
    output_size = env.action_space.n  # Ajusta según tu entorno
    agent = DQNAgent(input_size, output_size)

    # Entrenar el modelo
    train_dqn(env, agent)

    # Evaluar el modelo entrenado
    total_reward = 0
    done = False
    state = env.reset()

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Juego terminado. Recompensa total: {total_reward}")
    env.close()
