import gymnasium as gym
import numpy as np

ENV_NAME = "MountainCarContinuous-v0"

env = gym.make(ENV_NAME, render_mode="human")
observation, info = env.reset()

actions = []
T = 50
for t in range(T):
    random_action = env.action_space.sample()
    actions.append(random_action.item())
    observation, reward, terminated, truncated, info = env.step(random_action)

print(actions)