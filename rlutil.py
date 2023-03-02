import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

def plot_returns(episode_returns: torch.Tensor, show_result=False):
    """Plot return per each episode.
    
    Parameter:
        episode_returns: torch.Tensor containing return for each episode
        show_result: If true, plot result
    """
    plt.figure(1)
    if show_result:  # if training is over
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(episode_returns.numpy())

    if len(episode_returns) >= 100:
        means = episode_returns.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(99) * means[0], means))
        plt.plot(means.numpy())

    plt.pause(0.001)

def save_nn(neural_net: nn.Module, save_path: str):
    """Save neural network parameters to the given path."""
    torch.save(neural_net.state_dict(), save_path)

def load_nn(neural_net: nn.Module, load_path: str):
    """Load neural network parameters from the given path."""
    neural_net.load_state_dict(torch.load(load_path))

def mp4_to_gif(mp4_path: str, gif_path: str, filename: str):
    """Convert .mp4 to .gif"""
    videoClip = VideoFileClip(mp4_path + filename + ".mp4")
    videoClip.write_gif(gif_path + filename + ".gif")