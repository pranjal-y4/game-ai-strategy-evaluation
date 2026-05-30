"""
agents/c4_alphabeta_infeasible_agent.py
────────────────────────────────────────────────────────────────────────────
Full-tree Alpha-Beta for Connect 4 — FOR INFEASIBILITY DEMONSTRATION ONLY.

Like the Minimax variant this will not finish on a 6×7 board.  It is used
alongside C4MinimaxInfeasibleAgent in the infeasibility experiment to show
that even with alpha-beta pruning and center-first move ordering the
exhaustive search is completely intractable.

Design differences vs plain Minimax
─────────────────────────────────────
- Alpha-beta pruning is applied (reduces branching factor significantly).
- Center-first move ordering: columns are tried nearest-to-center first
  (3, 2, 4, 1, 5, 0, 6) to maximise pruning opportunities.
- Depth tracking and timeout mechanism are identical to the Minimax agent.
"""

import math
import time
from .base_agent import BaseAgent


class _Timeout(Exception):
    pass


class C4AlphaBetaInfeasibleAgent(BaseAgent):
    """
    Full-tree Alpha-Beta for Connect 4.
    FOR INFEASIBILITY DEMONSTRATION ONLY — will time out on a real game.
    """

    def __init__(self):
        self.nodes_expanded    = 0
        self.max_depth_reached = 0
        self._deadline         = None

    @property
    def name(self) -> str:
        return "C4_AlphaBeta_Infeasible"

    def reset(self):
        self.nodes_expanded    = 0
        self.max_depth_reached = 0

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def select_action(self, game, training: bool = False):
        self.nodes_expanded    = 0
        self.max_depth_reached = 0
        me  = game.current_player
        opp = 3 - me
        best_score = -math.inf
        best_move  = None
        alpha = -math.inf

        ordered = self._center_first(game.legal_moves(), game.cols)
        for move in ordered:
            game_clone = game.clone()
            game_clone.apply_move(move)
            score = self._alphabeta(game_clone, opp, me, False, alpha, math.inf, 1)
            if score > best_score:
                best_score = score
                best_move  = move
            alpha = max(alpha, best_score)

        return best_move

    @staticmethod
    def _center_first(moves: list, cols: int) -> list:
        """Sort moves by distance from center column (ascending)."""
        center = cols // 2
        return sorted(moves, key=lambda c: abs(c - center))

    def _alphabeta(self, game, player, root_player, is_max: bool,
                   alpha: float, beta: float, depth: int) -> float:
        self.nodes_expanded   += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)

        if self._deadline and (self.nodes_expanded % 10_000 == 0):
            if time.time() > self._deadline:
                raise _Timeout()

        winner = game.winner()
        if winner == root_player:
            return 1000 - depth
        if winner:
            return depth - 1000
        if game.is_terminal():
            return 0

        moves = self._center_first(game.legal_moves(), game.cols)

        if is_max:
            val = -math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = max(val, self._alphabeta(g, 3 - player, root_player,
                                               False, alpha, beta, depth + 1))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break  # beta cut-off
            return val
        else:
            val = math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = min(val, self._alphabeta(g, 3 - player, root_player,
                                               True, alpha, beta, depth + 1))
                beta = min(beta, val)
                if beta <= alpha:
                    break  # alpha cut-off
            return val
