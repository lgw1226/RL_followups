import torch
import torch.optim as optim
import torch.cuda as cuda

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import sys

sys.path.append("../RL_followups")

from rlblock import ReplayBuffer
from rlblock import NN
from rlblock import Transition


class Environment():
    def __init__(self,
                 env_name: str,
                 render_mode: str,
                 device: str
                 ):
        
        self.env_name = env_name
        self.render_mode = render_mode
        self.device = device

        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        self.n_ob = self.env.observation_space.shape[0]
        self.n_ac = self.env.action_space.shape[0]

    def reset(self):
        ob, info = self.env.reset()
        ob_t = torch.as_tensor(ob, dtype=torch.float32, device=self.device)

        return ob_t, info

    def step(self, ac):
        ob, rew, terminated, truncated, info = self.env.step(ac)

        ob_t = torch.as_tensor(ob, dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor([rew], dtype=torch.float32, device=self.device)

        done = terminated or truncated
        done_t = torch.as_tensor([done], dtype=torch.bool, device=self.device)\

        return ob_t, rew_t, done_t, info
        

class AgentDDPG():
    def __init__(self,
                 p_net: NN,
                 q_net:NN,
                 replay_buffer: ReplayBuffer,
                 p_optimizer: optim.Optimizer,
                 q_optimizer: optim.Optimizer,
                 batch_size: int = 1000,
                 gamma: float = 0.99,
                 tau_p: float = 0.3,
                 tau_q: float = 0.1):
        
        self.p_net = p_net
        self.q_net = q_net
        self.replay_buffer = replay_buffer
        self.p_optimizer = p_optimizer
        self.q_optimizer = q_optimizer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau_p = tau_p
        self.tau_q = tau_q

        self.p_targ_net = p_net
        self.q_targ_net = q_net

        self.ob = None
        self.ac = None
        self.rew = None
        self.next_ob = None
        self.done = None

    def get_batch(self):
        """Return batch data."""
        batch_transitions = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_acs, batch_rews, batch_next_obs, batch_done \
            = [*zip(*batch_transitions)]  # each is tuple of tensors
        
        return batch_obs, batch_acs, batch_rews, batch_next_obs, batch_done
    
    def save_transition(self, *args):
        """Save a transition in replay buffer."""
        self.replay_buffer.push(Transition(*args))

    def get_action(self, ob: torch.Tensor, noise=False):
        ac = self.p_net(ob)
        if noise: ac += torch.normal(0, 1, size=ac.size())
        ac = torch.clamp(ac, min=-1, max=1)

        return ac
    
    def get_q(self, ob: torch.Tensor, ac: torch.Tensor):
        x = torch.cat((ob, ac), 1)
        return self.q_net(x)

    def update(self):
        # all 5 variables below are in batch form!!!
        obs, acs, rews, next_obs, done = self.get_batch()

        # step for each neural net
        q_targ = rews \
               + self.gamma * (1 - done) * self.get_q(next_obs, self.get_action(next_obs))
        q_loss = torch.mean(torch.square(self.get_q(obs, acs) - q_targ), 0)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        p_loss = -torch.mean(self.get_q(obs, self.get_action(obs)), 0)

        self.p_optimizer.zero_grad()
        p_loss.backward()
        self.p_optimizer.step()

        # Soft update of the target networks' weights
        # θ′ ← τ θ + (1 −τ )θ′
        p_targ_net_state_dict = self.p_targ_net.state_dict()
        p_net_state_dict = self.p_net.state_dict()
        for key in p_net_state_dict:
            p_targ_net_state_dict[key] = p_net_state_dict[key] * self.tau_p \
                                        + p_targ_net_state_dict[key] * (1 - self.tau_p)
        self.p_targ_net.load_state_dict(p_targ_net_state_dict)

        q_targ_net_state_dict = self.q_targ_net.state_dict()
        q_net_state_dict = self.q_net.state_dict()
        for key in q_net_state_dict:
            q_targ_net_state_dict[key] = q_net_state_dict[key] * self.tau_q \
                                        + q_targ_net_state_dict[key] * (1 - self.tau_q)
        self.q_targ_net.load_state_dict(q_targ_net_state_dict)


if __name__ == "__main__":
    if cuda.is_available(): device = "cuda"
    else: device = "cpu"

    test_env = Environment("BipedalWalker-v3",
                           "human",
                           device)
    
    ob, info = test_env.reset()
    print(ob, info)

    ob, rew, done, info = test_env.step(test_env.env.action_space.sample())
    print(ob, rew, done, info)