"""
agents/ttt_minimax_agent.py
Minimax agent for Tic-Tac-Toe.
"""

import math
from .base_agent import BaseAgent


class TTTMinimaxAgent(BaseAgent):
    """Full minimax for Tic-Tac-Toe."""

    @property
    def name(self):
        return "TTT_Minimax"

    def reset(self):
        self.nodes_expanded = 0

    def select_action(self, game, training=False):
        self.nodes_expanded = 0
        me = game.current_player
        moves = game.legal_moves()
        best_score = -math.inf
        best_move = None

        for move in moves:
            game_clone = game.clone()
            game_clone.apply_move(move)
            score = self._minimax(game_clone, me, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _minimax(self, game, root_player, is_max):
        self.nodes_expanded += 1
        winner = game.winner()
        if winner == root_player:
            return 10 - game.move_count  # Depth-sensitive
        elif winner:
            return game.move_count - 10
        if game.is_terminal():
            return 0

        moves = game.legal_moves()
        if is_max:
            best = -math.inf
            for move in moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                best = max(best, self._minimax(game_clone, root_player, False))
            return best
        else:
            best = math.inf
            for move in moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                best = min(best, self._minimax(game_clone, root_player, True))
            return best