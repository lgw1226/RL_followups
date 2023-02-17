import torch
import torch.nn as nn

def save(network: nn.Module, filename: str):
    """saves network's states(parameters) as filename"""
    torch.save(network.state_dict(), filename)

# Code snippet to import modules that are in parent directory

# import sys
# sys.path.append("../RL_followups")
# from util import save