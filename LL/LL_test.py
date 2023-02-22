import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ENV_NAME = "LunarLander-v2"  # environment name
N_STEPS = 2000  # number of steps to play

# Gym environment
env = gym.make(ENV_NAME, render_mode="human")

observation, info = env.reset()
n_actions = env.action_space.n
n_observations = len(observation)

# CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust policy network structure according to the trained network
# NN
class NN(nn.Module):
    def __init__(self, sizes=[]) -> None:
        super(NN, self).__init__()
        
        self.sizes = sizes
        self.layers = nn.ModuleList()  # len(self.size)-1 layers in self.layers
        for i in range(len(self.sizes)-1):
            self.layers.append(nn.Linear(self.sizes[i], self.sizes[i+1]))
    
    def forward(self, x):
        for i in range(len(self.sizes)-1):
            x = self.layers[i](x)
            if i == len(self.sizes)-2:  # skip activation for the last layer
                return x
            else:
                x = F.relu(x)


trained_policy_net = NN([n_observations, 128, 128, n_actions]).to(device)
trained_policy_net.load_state_dict(torch.load("./LL/LL_trained_DQN"))
trained_policy_net.eval()

N_EPISODE = 5
for i in range(N_EPISODE):
    for j in count():
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = trained_policy_net(observation).max(1)[1].view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if done:
            observation, info = env.reset()
            break

env.close()