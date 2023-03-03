from itertools import count
import copy
import sys

sys.path.append("../RL_followups")

import torch
import torch.optim as optim
import torch.cuda as cuda

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import rlutil
from rlblock import ReplayBuffer
from rlblock import NN
from rlblock import Transition


class Environment():
    def __init__(self,
                 env_name: str,
                 render_mode=None,
                 device: str = "cpu"
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
        done_t = torch.as_tensor([done], dtype=torch.int32, device=self.device)\

        return ob_t, rew_t, done_t, info
        

class AgentDDPG():
    def __init__(self,
                 p_net: NN,
                 q_net:NN,
                 p_targ_net: NN,
                 q_targ_net: NN,
                 replay_buffer: ReplayBuffer,
                 batch_size: int = 1000,
                 gamma: float = 0.99,
                 tau_p: float = 0.3,
                 tau_q: float = 0.1,
                 lr_p: float = 0.0001,
                 lr_q: float = 0.0001,
                 device: str = "cpu"
                 ):
        
        self.p_net = p_net
        self.q_net = q_net
        self.p_targ_net = p_targ_net
        self.q_targ_net = q_targ_net
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau_p = tau_p
        self.tau_q = tau_q
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.device = device

        self.p_targ_net.load_state_dict(self.p_net.state_dict())
        self.q_targ_net.load_state_dict(self.q_net.state_dict())

        self.p_optim = optim.AdamW(self.p_net.parameters(), lr=self.lr_p, amsgrad=True)
        self.q_optim = optim.AdamW(self.q_net.parameters(), lr=self.lr_q, amsgrad=True)

    def get_batch(self):
        """Return batch data."""
        batch_transitions = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_acs, batch_rews, batch_next_obs, batch_done \
            = [*zip(*batch_transitions)]  # each is tuple of tensors
        
        batch_obs = rlutil.tuple_of_tensors_to_tensor(batch_obs)
        batch_acs = rlutil.tuple_of_tensors_to_tensor(batch_acs)
        batch_rews = rlutil.tuple_of_tensors_to_tensor(batch_rews)
        batch_next_obs = rlutil.tuple_of_tensors_to_tensor(batch_next_obs)
        batch_done = rlutil.tuple_of_tensors_to_tensor(batch_done)
                
        return batch_obs, batch_acs, batch_rews, batch_next_obs, batch_done
    
    def save_transition(self, *args):
        """Save a transition in replay buffer."""
        self.replay_buffer.push(*args)

    def get_action(self, ob: torch.Tensor, target=False, noise=False):
        if not target: ac = self.p_net(ob)
        else: ac = self.p_targ_net(ob)
        
        if noise: ac += torch.normal(0, 1, size=ac.size(), device=self.device)
        ac = torch.clamp(ac, min=-1, max=1)

        return ac
    
    def get_q(self, ob: torch.Tensor, ac: torch.Tensor, target=False):
        x = torch.cat((ob, ac), 1)

        if not target: y = self.q_net(x)
        else: y = self.q_targ_net(x)

        return y

    def update(self):
        # all 5 variables below are in batch form!!
        obs, acs, rews, next_obs, done = self.get_batch()

        # step for each neural net
        q_targ = rews \
                + self.gamma \
                * (1 - done) \
                * self.get_q(next_obs, self.get_action(next_obs, target=True), target=True)
        q_loss = torch.mean(torch.square(self.get_q(obs, acs) - q_targ), 0)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        p_loss = -torch.mean(self.get_q(obs, self.get_action(obs)), 0)

        self.p_optim.zero_grad()
        p_loss.backward()
        self.p_optim.step()

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


def train(agent: AgentDDPG, env: Environment, n_episode=100):

    ep_rets = torch.empty(0, device=env.device)

    ob, _ = env.reset()

    for i_episode in range(n_episode):
        if i_episode % 100 == 0:
            rlutil.save_nn(p_net, "BW/BW_trained_DDPG.nn")
        
        rews = torch.empty(0, device=env.device)
        for t in count():
            ac = agent.get_action(ob, noise=True)
            next_ob, rew, done, _ = env.step([ac.item()])
            rews = torch.cat((rews, rew))

            agent.save_transition(ob, ac, rew, next_ob, done)
            ob = next_ob

            if done:
                ob, _ = env.reset()
                ep_ret = torch.sum(rews, 0, keepdim=True)
                ep_rets = torch.cat((ep_rets, ep_ret))
                print(ep_rets)
                rlutil.plot_returns(ep_rets)
                break
            
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.update()

    rlutil.plot_returns(ep_rets)
    plt.show()
            


if __name__ == "__main__":
    if cuda.is_available(): device = "cuda"
    else: device = "cpu"

    env = Environment("MountainCarContinuous-v0",
                      device=device)
    
    p_net_sizes = [env.n_ob] + [100, 100, 50] + [env.n_ac]
    p_net = NN(p_net_sizes).to(device)
    p_targ_net = NN(p_net_sizes).to(device)

    q_net_sizes = [env.n_ob + env.n_ac] + [150, 100, 30] + [1]
    q_net = NN(q_net_sizes).to(device)
    q_targ_net = NN(q_net_sizes).to(device)

    replay_buffer = ReplayBuffer(100000)

    agent = AgentDDPG(p_net,
                      q_net,
                      p_targ_net,
                      q_targ_net,
                      replay_buffer,
                      device=device)
    
    train(agent, env, 100)
