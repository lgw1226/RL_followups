import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
import gymnasium as gym

# Replay Buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer(object):
    '''Replay buffer which saves transition: (s, a, r, s', d).'''
    def __init__(self, maxlen: int) -> None:
        '''Create a replay buffer with given max length.'''
        self.memory = deque([], maxlen=maxlen)

    def push(self, *args) -> None:
        '''Save a transition.'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        '''Sample transitions, return sampled transitions.'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        '''Return the number of transitions in the replay buffer.'''
        return len(self.memory)


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
                x = f.relu(x)

# get_action function can differ according to the environment
def get_action(p_net: NN, obs:torch.Tensor, noise=False) -> torch.Tensor:
    '''Get action from the policy network given the observation, return action'''
    action = p_net(obs)
    if noise:
        action += random.gauss(0, 1) 
    action_t =  torch.clamp(action, min=-1, max=1)
    return action_t

def get_q_val(q_net: NN, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    '''Get action value (q-value) from the q network given the observation, return q value'''
    x = torch.cat((obs, action), 1)
    return q_net(x)

def train_DDPG(env: gym.Env, replay_buffer: ReplayBuffer,
               p_net: NN, p_targ_net: NN, q_net: NN, q_targ_net: NN, device, 
               gamma=0.9, steps=1000, lr=1e-4, batch_size=100, rho=0.99):
    '''Train given p_net & q_net and save the result.'''
    p_optimizer = optim.AdamW(p_net.parameters(), lr=lr, amsgrad=True)
    q_optimizer = optim.AdamW(q_net.parameters(), lr=lr, amsgrad=True)

    p_targ_net.load_state_dict(p_net.state_dict())
    q_targ_net.load_state_dict(q_net.state_dict())

    obs, info = env.reset()

    for step in range(steps):
        # observe state and select action
        action = get_action(p_net,
                            torch.as_tensor(obs, dtype=torch.float32, device=device), 
                            noise=True).item()

        # execute action in the environment and observe
        next_obs, reward, terminated, truncated, info = env.step([action])
        done = terminated or truncated
        
        # store (s, a, r, s', d) in replay buffer
        replay_buffer.push(obs, action, reward, next_obs, done)

        # if s' is terminal, reset environment state
        if done:
            observation, info = env.reset()
        
        # update if length of replay buffer is greater of equal to 100
        if len(replay_buffer) >= batch_size:
            # sample from replay buffer
            batch_transitions = replay_buffer.sample(batch_size)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done \
                = [*zip(*batch_transitions)]

            # change data type to torch.tensor
            # change list of np.array to np.array first for computation
            batch_obs_t = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=device)
            batch_next_obs_t = torch.tensor(np.array(batch_next_obs), dtype=torch.float32, device=device)
            batch_actions_t = torch.tensor(batch_actions, dtype=torch.float32, device=device)
            if batch_actions_t.dim() == 1:
                batch_actions_t.unsqueeze(1)
            batch_rewards_t = torch.tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(1)
            batch_done_t = torch.tensor(batch_done, dtype=torch.int32, device=device).unsqueeze(1)  # True = 1, False = 0

            # compute q-target
            q_targ_t = batch_rewards_t \
                       + gamma \
                       * (1 - batch_done_t) \
                       * get_q_val(q_targ_net, batch_next_obs_t, get_action(p_targ_net, batch_next_obs_t))

            # compute q-function loss
            q_loss_t = torch.mean(torch.square(get_q_val(q_net, batch_obs_t, batch_actions_t) - q_targ_t), 0)

            # update q-function
            q_optimizer.zero_grad()
            q_loss_t.backward()
            q_optimizer.step()

            # compute policy loss
            p_loss_t = torch.mean(get_q_val(q_net, batch_obs_t, get_action(p_net, batch_obs_t)), 0)

            # update policy
            p_optimizer.zero_grad()
            p_loss_t.backward()
            p_optimizer.step()

            # Soft update of the target networks' weights
            # θ′ ← τ θ + (1 −τ )θ′
            p_targ_net_state_dict = p_targ_net.state_dict()
            p_net_state_dict = p_net.state_dict()
            for key in p_net_state_dict:
                p_targ_net_state_dict[key] = p_targ_net_state_dict[key] * rho \
                                           + p_net_state_dict[key] * (1 - rho)
            p_targ_net.load_state_dict(p_targ_net_state_dict)

            q_targ_net_state_dict = q_targ_net.state_dict()
            q_net_state_dict = q_net.state_dict()
            for key in q_net_state_dict:
                q_targ_net_state_dict[key] = q_targ_net_state_dict[key] * rho \
                                           + q_net_state_dict[key] * (1 - rho)
            q_targ_net.load_state_dict(q_targ_net_state_dict)

if __name__ == "__main__":
    ENV_NAME = "MountainCarContinuous-v0"
    env = gym.make(ENV_NAME, render_mode="human")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_obs = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]

    hidden_p = [50, 50]
    hidden_q = [50, 50]

    sizes_p = [n_obs] + hidden_p + [n_action]
    sizes_q = [n_obs + n_action] + hidden_q + [1]

    p_net = NN(sizes_p).to(DEVICE)
    p_targ_net = NN(sizes_p).to(DEVICE)
    q_net = NN(sizes_q).to(DEVICE)
    q_targ_net = NN(sizes_q).to(DEVICE)

    buffer = ReplayBuffer(100)

    train_DDPG(env, buffer, p_net, p_targ_net, q_net, q_targ_net, DEVICE, steps=300)