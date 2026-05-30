"""
I have implemented and reviewed this module structure.
"""


import random
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):

    @property
    def name(self) -> str:


        return "Random"


    def select_action(self, game, training: bool = False):
        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, training.
        """
        return random.choice(game.legal_moves())
