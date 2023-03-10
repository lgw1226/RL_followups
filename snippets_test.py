import torch
import gymnasium as gym
import numpy as np

from rlblock import Environment

def ac_test():
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    env.reset()

    # list action
    env.step([1])

    # np.array action
    ac_np = np.array([1])
    env.step(ac_np)

    # torch.Tensor action
    ac_t = torch.tensor([1])
    env.step(ac_t)

    """ # float action
    env.step(1)

    # np.float action
    ac_np = np.array(1)
    env.step(ac_np)

    # torch.Tensor with torch.Size([]) action
    ac_t = torch.tensor(1)
    env.step(ac_t) """

def env_test():
    env = gym.make("ALE/Breakout-v5")
    env.reset()

    print(env.observation_space)
    print(env.action_space)

    if type(env.action_space) == gym.spaces.Discrete:
        print("Discrete action space")
        ob, rwd, terminated, truncated, _ = env.step(0)
        print(env.action_space.n)
        print(env.action_space.sample())
    else:
        print("Continuous action space")
        ob, rwd, terminated, truncated, _ = env.step([0])
        print(env.action_space.shape[0])
        print(env.action_space.sample())

def tensor_test():
    t = torch.tensor([True], dtype=torch.float32)
    print(t)

def main_test():
    testenv = Environment("ALE/Breakout-v5")
    print(testenv)


if __name__ == "__main__":
    main_test()