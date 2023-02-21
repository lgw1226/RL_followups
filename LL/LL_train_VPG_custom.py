import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

import math
import random
import numpy as np

# Setting up gym, cuda
ENV_NAME = "LunarLander-v2"
env = gym.make(ENV_NAME, render_mode="human")

obs, _ = env.reset()
n_obs = len(obs)  # dimension of the state space
n_act = env.action_space.n  # number of actions an agent can take

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
LR = 1e-4

# Policy network
class NN(nn.Module):
    def __init__(self, hidden_size=[]) -> None:
        super(NN, self).__init__()
        
        self.size = [n_obs] + hidden_size + [n_act]
        print(self.size)
        self.layers = []
        for i in range(len(self.size)-1):
            self.layers.append(nn.Linear(self.size[i], self.size[i+1]))
    
    def forward(self, x):
        for i in range(len(self.size)-1):
            x = self.layers[i](x)
            if i == len(self.size)-2:
                return x
            else:
                x = F.relu(x)

test_net = NN([128, 128])
x = test_net(torch.rand(8))
print(x)

N_STEPS = 100
for i_step in range(N_STEPS):
    act = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(act)
    done = terminated or truncated

    if done:
        break

env.close()