import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ENV_NAME = "CartPole-v1"  # environment name
N_STEPS = 100  # number of steps to play

# Gym environment
env = gym.make(ENV_NAME, render_mode="human")

observation, info = env.reset()
n_actions = env.action_space.n
n_observations = len(observation)

# CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust policy network structure according to the trained network
class PolicyNet(nn.Module):

    def __init__(self, n_observations, n_actions) -> None:
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


trained_policy_net = PolicyNet(n_observations, n_actions).to(device)
trained_policy_net.load_state_dict(torch.load("./trained_policy_state"))
trained_policy_net.eval()

# Simulation
for _ in range(N_STEPS):
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = trained_policy_net(observation).max(1)[1].view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    
    # done = terminated or truncated

    # if done:
    #     break

env.close()