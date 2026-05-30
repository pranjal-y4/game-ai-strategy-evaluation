"""
I have implemented and reviewed this module structure.
"""


import random
from .base_agent import BaseAgent


class DefaultAgent(BaseAgent):

    @property
    def name(self) -> str:


        return "Default"


    def select_action(self, game, training: bool = False):
        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, training.
        """
        legal = game.legal_moves()
        player = game.current_player
        opponent = 3 - player


        for move in legal:
            g = game.clone()
            g.apply_move(move)
            if g.winner() == player:
                return move


        for move in legal:
            g = game.clone()

            g.current_player = opponent
            g.apply_move(move)
            if g.winner() == opponent:
                return move


        return self._fallback(game, legal)


    def _fallback(self, game, legal):


        gtype = getattr(game, "game_type", "unknown")

        if gtype == "ttt":

            prefs = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2),
                     (0, 1), (1, 0), (1, 2), (2, 1)]
            for move in prefs:
                if move in legal:
                    return move

        elif gtype == "c4":

            mid = game.cols // 2
            order = sorted(legal, key=lambda c: abs(c - mid))
            return order[0]

        return random.choice(legal)
