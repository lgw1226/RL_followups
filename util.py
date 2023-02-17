import torch
import torch.nn as nn

def save(network: nn.Module, filename: str):
    """saves network's states(parameters) as filename"""
    torch.save(network.state_dict(), filename)
