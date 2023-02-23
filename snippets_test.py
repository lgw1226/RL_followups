import torch
import numpy as np
import random
from collections import namedtuple, deque

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

test_buffer = ReplayBuffer(10)
for i in range(5):
    test_buffer.push([i, i], i, i, [i, i], i)

batch_transitions = test_buffer.sample(3)
obs, actions, rewards, _, _ = [*zip(*batch_transitions)]

rewards_t = torch.tensor(rewards).unsqueeze(1)
print(rewards_t)
print(torch.mean(rewards_t, 0, dtype=torch.float32))