"""
agents/random_agent.py
Random agent for baseline.
"""

import random
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random legal move agent."""

    @property
    def name(self):
        return "Random"

    def reset(self):
        pass

    def select_action(self, game, training=False):
        moves = game.legal_moves()
        return random.choice(moves) if moves else None