import torch
import numpy as np
import random
from collections import namedtuple, deque

x = torch.tensor([1, 2], dtype=torch.float32)
y = torch.tensor([3, 4], dtype=torch.float32)

avg = torch.mean(torch.square(x - y))
print(avg)