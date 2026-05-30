"""
utils/plotting.py
Utilities for plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_win_rates(episodes, win_rates, title="Training Progress", save_path=None):
    """Plot win rates over training episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, win_rates, label='Win Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_crossplay_heatmap(results, agents, title="Crossplay Results", save_path=None):
    """Plot crossplay results as heatmap."""
    win_matrix = np.zeros((len(agents), len(agents)))
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if agent1 == agent2:
                win_matrix[i, j] = 0.5  # Draw against self
            else:
                # Find result
                for r in results:
                    if r['agent1'] == agent1 and r['agent2'] == agent2:
                        win_matrix[i, j] = r['agent1_win_rate']
                        break

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(agents)))
    ax.set_yticks(np.arange(len(agents)))
    ax.set_xticklabels(agents)
    ax.set_yticklabels(agents)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if save_path:
        plt.savefig(save_path)
    plt.show()