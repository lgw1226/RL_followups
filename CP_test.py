import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Setting up the environment
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
n_actions = env.action_space.n
n_observations = len(observation)

trained_policy_net = DQN(n_observations, n_actions).to(device)
trained_policy_net.load_state_dict(torch.load("./trained_policy_state"))
trained_policy_net.eval()


for _ in range(10):
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = trained_policy_net(observation).max(1)[1].view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    # observation, info = env.reset()

env.close()