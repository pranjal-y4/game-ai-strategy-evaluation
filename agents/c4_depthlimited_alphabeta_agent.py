"""
agents/c4_depthlimited_alphabeta_agent.py
Depth-limited alpha-beta with heuristic for practical Connect 4 play.
"""

import math
from .base_agent import BaseAgent


class C4DepthLimitedAlphaBetaAgent(BaseAgent):
    """Depth-limited alpha-beta with heuristic for Connect 4."""

    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    @property
    def name(self):
        return f"C4_AlphaBeta_Depth{self.max_depth}"

    def reset(self):
        self.nodes_expanded = 0

    def select_action(self, game, training=False):
        self.nodes_expanded = 0
        me = game.current_player
        opp = 3 - me
        moves = game.legal_moves()
        best_score = -math.inf
        best_move = None
        alpha = -math.inf

        # Move ordering: center first
        ordered_moves = sorted(moves, key=lambda c: abs(c - game.cols//2))

        for move in ordered_moves:
            game_clone = game.clone()
            game_clone.apply_move(move)
            score = self._alphabeta(game_clone, opp, me, False, alpha, math.inf, self.max_depth - 1)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    def _alphabeta(self, game, player, root_player, is_max, alpha, beta, depth):
        self.nodes_expanded += 1
        winner = game.winner()
        if winner == root_player:
            return 1000 + depth  # Prefer quicker wins
        elif winner:
            return -1000 - depth
        if game.is_terminal():
            return 0
        if depth == 0:
            return self._heuristic(game, root_player)

        moves = game.legal_moves()
        ordered_moves = sorted(moves, key=lambda c: abs(c - game.cols//2))

        if is_max:
            val = -math.inf
            for move in ordered_moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                val = max(val, self._alphabeta(game_clone, 3-player, root_player, False, alpha, beta, depth - 1))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = math.inf
            for move in ordered_moves:
                game_clone = game.clone()
                game_clone.apply_move(move)
                val = min(val, self._alphabeta(game_clone, 3-player, root_player, True, alpha, beta, depth - 1))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def _heuristic(self, game, player):
        """Heuristic evaluation function."""
        opp = 3 - player
        score = 0

        # Center column bonus
        center_col = game.cols // 2
        center_count = sum(1 for r in range(game.rows) if game.board[r][center_col] == player)
        score += center_count * 3

        # Evaluate windows
        def evaluate_window(window):
            p_count = window.count(player)
            opp_count = window.count(opp)
            empty_count = window.count(0)

            if p_count == 4:
                return 100
            elif p_count == 3 and empty_count == 1:
                return 10
            elif p_count == 2 and empty_count == 2:
                return 2
            elif opp_count == 3 and empty_count == 1:
                return -80
            elif opp_count == 2 and empty_count == 2:
                return -2
            return 0

        # Check all windows
        for r in range(game.rows):
            for c in range(game.cols - 3):
                window = [game.board[r][c+i] for i in range(4)]
                score += evaluate_window(window)

        for c in range(game.cols):
            for r in range(game.rows - 3):
                window = [game.board[r+i][c] for i in range(4)]
                score += evaluate_window(window)

        for r in range(game.rows - 3):
            for c in range(game.cols - 3):
                window = [game.board[r+i][c+i] for i in range(4)]
                score += evaluate_window(window)
                window = [game.board[r+3-i][c+i] for i in range(4)]
                score += evaluate_window(window)

        return score