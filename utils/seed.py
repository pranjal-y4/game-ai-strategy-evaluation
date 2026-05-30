"""
utils/seed.py
Set reproducible random seeds across all libraries.
Torch import is optional – if not installed the function still seeds random and numpy.
"""

import random
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def set_seed(seed: int) -> None:
    """Set seed for random, numpy, and (optionally) torch."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
