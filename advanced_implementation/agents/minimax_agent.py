"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
import math
from .base_agent import BaseAgent


class MinimaxAgent(BaseAgent):


    # I have implemented this callable with parameters: self, max_depth.
    def __init__(self, max_depth: int = 100):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, max_depth.
        """
        self.max_depth = max_depth
        self.nodes_expanded = 0

    @property


    # I have implemented this callable with parameters: self.
    def name(self) -> str:


        return f"Minimax(d={self.max_depth})"


    # I have implemented this callable with parameters: self.
    def reset(self) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        self.nodes_expanded = 0


    # I have implemented this callable with parameters: self, game, training.
    def select_action(self, game, training: bool = False):


        self.nodes_expanded = 0
        root_player = game.current_player
        best_score = -math.inf
        best_move = None

        for move in game.legal_moves():
            g = game.clone()
            g.apply_move(move)
            score = self._minimax(g, root_player, False, self.max_depth - 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    # I have implemented this callable with parameters: self, game, root_player, is_max, depth.
    def _minimax(self, game, root_player: int, is_max: bool, depth: int) -> float:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, root_player, is_max, depth.
        """
        self.nodes_expanded += 1

        w = game.winner()
        if w is not None:
            if w == root_player:
                return 1000.0 + depth
            elif w == 0:
                return 0.0
            else:
                return -(1000.0 + depth)

        if depth == 0:
            return 0.0

        moves = game.legal_moves()
        if is_max:
            best = -math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                best = max(best, self._minimax(g, root_player, False, depth - 1))
            return best
        else:
            best = math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                best = min(best, self._minimax(g, root_player, True, depth - 1))
            return best
