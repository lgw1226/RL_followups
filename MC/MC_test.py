import math
import random
from collections import namedtuple, deque
from itertools import count

import gymnasium as gym
from gymnasium.utils.save_video import save_video

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


ENV_NAME = "MountainCarContinuous-v0"  # environment name
N_EPISODE = 1  # number of episodes to play

# Gym environment
env = gym.make(ENV_NAME, render_mode="rgb_array_list")
print(env.metadata)

observation, info = env.reset()
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

# CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust policy network structure according to the trained network
# NN
class NN(nn.Module):
    '''Simple multi-layered neural network.'''
    def __init__(self, network_size) -> None:
        '''Create neural network with given network size.
        
        Parameter:
            network_size (list): A list of the number of nodes in each layer
        '''
        super(NN, self).__init__()

        self.network_size = network_size
        self.n_layer = len(self.network_size) - 1
        self.layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.layers.append(nn.Linear(self.network_size[i], self.network_size[i+1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Do forward pass of the neural net, return activation torch.tensor.
        
        Parameter:
            x (torch.tensor)
        Return:
            x (torch.tensor)
        '''
        # insert activation function between layers
        for i in range(self.n_layer):
            x = self.layers[i](x)
            if i == self.n_layer - 1:
                return x
            else:
                x = F.leaky_relu(x)

trained_policy_net = NN([n_observations] + [16, 32, 16] + [n_actions]).to(device)
trained_policy_net.load_state_dict(torch.load("./MC/MC_trained_DDPG"))
trained_policy_net.eval()

def get_action(p_net: NN, obs:torch.Tensor, noise=False) -> torch.Tensor:
    '''Get action from the policy network given the observation, return action'''
    action = p_net(obs)
    if noise:
        action += random.gauss(0, 1) 
    action_t =  torch.clamp(action, min=-1, max=1)
    return action_t

for i in range(N_EPISODE):
    ep_rewards = []
    for j in count():
        action = get_action(trained_policy_net, torch.as_tensor(observation, dtype=torch.float32, device=device))
        observation, reward, terminated, truncated, _ = env.step([action.item()])
        ep_rewards.append(reward)
        done = terminated or truncated

        if done:
            save_video(
                env.render(),
                "./MC",
                name_prefix="MC_DDPG",
                fps=env.metadata["render_fps"])
            print("Return of the episode: %.3f" % (sum(ep_rewards)))
            observation, info = env.reset()
            break

env.close()