import gymnasium as gym
import numpy as np

ENV_NAME = "BipedalWalker-v3"

env = gym.make(ENV_NAME, hardcore=False, render_mode="human")
observation, info = env.reset()

actions = []
T = 1000
for t in range(T):
    random_action = env.action_space.sample()
    actions.append(random_action)
    observation, reward, terminated, truncated, info = env.step(random_action)
    done = terminated or truncated

    if done: break