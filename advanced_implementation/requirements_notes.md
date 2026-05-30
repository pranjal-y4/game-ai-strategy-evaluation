# Requirements Notes

## Runtime Dependencies

### Required
- **numpy >= 1.20.0** — Used everywhere for board representation and state encoding
- **torch >= 1.12.0** — Required for AdvancedDQNAgent (Double DQN with PER)

### Optional (for plotting and CSV analysis)
- **matplotlib >= 3.5.0** — Training curves, heatmaps, outcome bar charts
- **pandas >= 1.4.0** — CSV loading in `plot_training_curves()`
- **seaborn** — Optional for prettier heatmaps (falls back to matplotlib)

## No Additional Dependencies Needed

The following work with only numpy + stdlib:
- TicTacToe and Connect4 game environments
- All search agents (Minimax, Alpha-Beta, Advanced Alpha-Beta)
- Tabular Q-learning
- PrioritizedReplayBuffer (sum-tree uses only numpy)
- NStepBuffer
- GameEnv
- All utilities (seed, metrics, logger)
- All training scripts except DQN
- All evaluation scripts

## Why PyTorch for DQN?

PyTorch is justified for the DQN agent because:
1. Correct gradient computation for Double DQN requires autograd
2. IS-weighted loss `(weights * (y - q)^2).mean()` is natural in PyTorch
3. Gradient clipping `nn.utils.clip_grad_norm_()` is built-in
4. Target network sync `load_state_dict()` is a single line
5. PyTorch is already a dependency in the original project's requirements.txt

The rest of the system (games, search agents, Q-learning, evaluation)
remains pure Python/NumPy and is academically explainable without PyTorch.

## Installation

```bash
# Already satisfied if original project runs:
pip install numpy matplotlib pandas seaborn torch

# Or using the original requirements.txt from the parent project:
pip install -r requirements.txt
```

## Python Version
Requires Python 3.9+ (uses `from __future__ import annotations` for
`type | None` syntax compatibility).
