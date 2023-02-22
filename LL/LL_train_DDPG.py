import torch
import torch.nn as nn
import torch.functional as f
from collections import namedtuple, deque
import random

# Replay Buffer
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayBuffer(object):
    '''Replay buffer which saves transition: (s, a, s', r).'''
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