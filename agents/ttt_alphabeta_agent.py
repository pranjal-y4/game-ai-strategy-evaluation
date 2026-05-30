"""
agents/ttt_alphabeta_agent.py
Alpha-beta pruning agent for Tic-Tac-Toe.
"""

import math
from .base_agent import BaseAgent


class TTTAlphaBetaAgent(BaseAgent):
    """Alpha-beta pruning for Tic-Tac-Toe."""

    @property
    def name(self):
        return "TTT_AlphaBeta"

    def reset(self):
        self.nodes_expanded = 0

    def select_action(self, game, training=False):
        self.nodes_expanded = 0
        me = game.current_player
        moves = game.legal_moves()
        best_score = -math.inf
        best_move = None
        alpha = -math.inf

        for move in moves:
            game_clone = game.clone()
            game_clone.apply_move(move)
            score = self._alphabeta(game_clone, me, False, alpha, math.inf)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    def _alphabeta(self, game, root_player, is_max, alpha, beta):
        self.nodes_expanded += 1
        winner = game.winner()
        if winner == root_player:
            return 10 - game.move_count
        elif winner:
            return game.move_count - 10
        if game.is_terminal():
            return 0

        moves = game.legal_moves()
        if is_max:
            val = -math.inf
            for move in moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                val = max(val, self._alphabeta(game_clone, root_player, False, alpha, beta))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = math.inf
            for move in moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                val = min(val, self._alphabeta(game_clone, root_player, True, alpha, beta))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val