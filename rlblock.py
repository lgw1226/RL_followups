import random
from collections import namedtuple, deque

import torch
import torch.nn as nn

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

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