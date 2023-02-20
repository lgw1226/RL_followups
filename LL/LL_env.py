import gymnasium as gym
import numpy as np

ENV_NAME = "LunarLander-v2"

env = gym.make(ENV_NAME, render_mode="human")
observation, info = env.reset()

n_action = env.action_space.n
n_state = len(observation)

print("The number of actions the agent can take: ", n_action)
print("10 samples from the action space")
for i in range(10):
    print(env.action_space.sample(), end=' ')
print('\n')

print("Sampled observation from the observation space")
print(observation)
print(np.shape(observation))