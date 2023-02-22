"""
LL_train_VPG.py

The original version of the code can be found at 
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import AdamW

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

import math
import random
import numpy as np
from itertools import count

# Setting up gym, cuda
ENV_NAME = "LunarLander-v2"
RENDER = False

if RENDER:
    env = gym.make(ENV_NAME, render_mode="human")
else:
    env = gym.make(ENV_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("environment:", ENV_NAME)
print("torch.device:", DEVICE)

# Hyperparameters
EPOCHS = 300
HIDDEN_SIZES = [150, 150]
BATCH_SIZE = 5  # the number of episodes to calculate gradient
LR = 1e-3

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


# Making environment and policy net
obs, _ = env.reset()
n_obs = len(obs)  # dimension of the state space
n_act = env.action_space.n  # number of actions an agent can take

sizes = [n_obs] + HIDDEN_SIZES + [n_act]
policy_net = NN(sizes).to(DEVICE)

def get_policy(obs):
    logits = policy_net(obs)
    return Categorical(logits=logits)  # softmax of network outcomes

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

def reward_to_go(ep_reward):
    n = len(ep_reward)
    rtgs = np.zeros_like(ep_reward)
    for i in reversed(range(n)):
        rtgs[i] = ep_reward[i] + (rtgs[i+1] if i+1 < n else 0)
    return list(rtgs)

# Computing gradient for (BATCH_SIZE) episodes
def compute_batch_loss():
    batch_obs = []
    batch_act = []
    batch_weight = []
    batch_return = []
    batch_len = []

    for i in range(BATCH_SIZE):
        ep_obs = []
        ep_act = []
        ep_reward = []
        ep_weight = []

        obs, info = env.reset()

        for t in count():  # timestep t
            ep_obs.append(obs)

            act = get_action(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE))
            ep_act.append(act)

            obs, reward, terminated, truncated, info = env.step(act)
            ep_reward.append(reward)

            done = terminated or truncated

            if done:  # if the episode is terminated or truncated
                ep_len = t + 1  # length of the episode
                ep_return = sum(ep_reward)  # total reward from start to end
                ep_weight = reward_to_go(ep_reward)

                batch_obs += (ep_obs)
                batch_act += (ep_act)
                batch_weight += (ep_weight)
                batch_len.append(ep_len)
                batch_return.append(ep_return)

                break

    batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=DEVICE),
                            act=torch.as_tensor(batch_act, dtype=torch.int32, device=DEVICE),
                            weights=torch.as_tensor(batch_weight, dtype=torch.float32, device=DEVICE))
    
    return batch_loss, batch_return, batch_len

returns = []

def plot_returns(show_result=False):
    plt.figure(1)
    returns_t = torch.tensor(returns, dtype=torch.float)  # np.ndarray to torch.tensor
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Epochs')
    plt.ylabel('Return')
    plt.plot(returns_t.numpy())
    if len(returns_t) >= 100:
        means = returns_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

optimizer = AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    batch_loss, batch_return, batch_len = compute_batch_loss()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch+1, batch_loss, np.mean(batch_return), np.mean(batch_len)))
    returns.append(np.mean(batch_return))

    batch_loss.backward()
    optimizer.step()

    plot_returns()

torch.save(policy_net.state_dict(), "./LL/LL_trained_VPG")

plot_returns(show_result=True)
plt.show()

env.close()