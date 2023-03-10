import random
from collections import namedtuple, deque

import gymnasium as gym

import torch
import torch.nn as nn
import torch.cuda as cuda


# cuda device
if cuda.is_available(): device = "cuda"
else: device = "cpu"

# transition
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class Environment():
    def __init__(self, env_name: str, render_mode=None) -> None:
        self.name = env_name
        self.render_mode = render_mode

        self._env = gym.make(self.name, render_mode=self.render_mode)  # hidden
        self.is_vector = len(self._env.observation_space.shape) == 1  # if False -> observation is image
        self.is_discrete = type(self._env.action_space) == gym.spaces.Discrete  # if False -> action is continuous

        # observation & action dimension
        if self.is_vector:
            self.dim_ob = self._env.observation_space.shape[0]
        else:
            self.dim_ob = self._env.observation_space.shape
        
        if self.is_discrete:
            self.dim_ac = self._env.action_space.n
        else:
            self.dim_ac = self._env.action_space.shape[0]

    def step(self, ac: torch.Tensor):
        if self.is_discrete: ac = ac.item()

        ob, rwd, terminated, truncated, _ = self._env.step(ac)

        ob_t = torch.tensor(ob, dtype=torch.float32, device=device)
        rwd_t = torch.tensor([rwd], dtype=torch.float32, device=device)
        done_t = torch.tensor([terminated or truncated], dtype=torch.float32, device=device)

        return ob_t, rwd_t, done_t  # tensor(ob), tensor([rew]), tensor([done (as 0 or 1)])
    
    def reset(self):
        ob, _ = self._env.reset()

        return torch.tensor(ob, dtype=torch.float32, device=device)
    
    def __repr__(self) -> str:
        ret = "Name: " + self.name

        ret += ", Observation type: "
        if self.is_vector: ret += "1D Vector"
        else: ret += "Image"

        ret += ", Action type: "
        if self.is_discrete: ret += "Discrete"
        else: ret += "Continuous"

        ret += f", Observation: {self.dim_ob}"
        ret += f", Action: {self.dim_ac}"

        return ret

class Agent():
    def __init__(self, ac_space) -> None:
        self.ac_space = ac_space
        
        if self.ac_space == gym.spaces.Discrete:
            self.is_discrete = True
        else:
            self.is_discrete = False
        
        self.dim_ac = self.ac_space.n

        self.ob = None
        self.ac = None
        self.rwd = None
        self.next_ob = None

    def get_ac(self):
        """return random action"""
        ac = self.ac_space.sample()
        
        if self.is_discrete:
            self.ac = torch.tensor([ac], dtype=torch.int32, device=device)
        else:
            self.ac = torch.tensor(ac, dtype=torch.float32, device=device)

        return self.ac  # [ac_1, ac_2, ...]

class ReplayBuffer(object):
    """Replay buffer which saves transition: (s, a, r, s', d)."""
    def __init__(self, maxlen: int) -> None:
        """Create a replay buffer with given max length."""
        self.memory = deque([], maxlen=maxlen)

    def push(self, *args) -> None:
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        """Sample transitions, return sampled transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Return the number of transitions in the replay buffer."""
        return len(self.memory)

class NN(nn.Module):
    """Simple multi-layered neural network."""
    def __init__(self, network_size: list, activation=nn.LeakyReLU()) -> None:
        """Create neural network with given network size and activation function.
        
        Parameter:
            network_size (list): A list of the number of nodes in each layer
            activation (function): Activation function to be inserted between layers
        """
        super(NN, self).__init__()

        self.network_size = network_size  # the number of nodes to be connected
        self.activation = activation

        self.n_layer = len(self.network_size) - 1  # the number of layers in NN
        self.layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.layers.append(nn.Linear(self.network_size[i], self.network_size[i+1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do forward pass of the neural net, return activation torch.tensor.
        
        Parameter:
            x (torch.tensor)
        Return:
            x (torch.tensor)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_layer -1:
                x = self.activation(x)
            
        return x