"""
agents/c4_minimax_infeasible_agent.py
────────────────────────────────────────────────────────────────────────────
Full-tree Minimax for Connect 4 — FOR INFEASIBILITY DEMONSTRATION ONLY.

This agent will NOT finish on a 6×7 board within any reasonable time budget.
It exists solely to measure how far plain Minimax gets before timing out,
thereby proving the infeasibility of exhaustive search for Connect 4.

Usage: see experiments/connect4_infeasibility.py

Design
──────
- Depth is tracked explicitly (not via game.move_count) so max_depth_reached
  is meaningful even when the search is cut short by a timeout.
- A TimeoutError is raised every 10 000 nodes (minimal overhead) when the
  external deadline has passed.  The caller must catch TimeoutError.
- No alpha-beta pruning (intentional — we want pure Minimax node counts).
- Center-column move ordering is NOT applied here so that comparison with
  alpha-beta is as fair as possible for the infeasibility experiment.
"""

import math
import time
from .base_agent import BaseAgent


class _Timeout(Exception):
    pass


class C4MinimaxInfeasibleAgent(BaseAgent):
    """
    Full-tree Minimax for Connect 4.
    FOR INFEASIBILITY DEMONSTRATION ONLY — will time out on a real game.
    """

    def __init__(self):
        self.nodes_expanded   = 0
        self.max_depth_reached = 0
        self._deadline        = None   # set by connect4_infeasibility.py

    @property
    def name(self) -> str:
        return "C4_Minimax_Infeasible"

    def reset(self):
        self.nodes_expanded    = 0
        self.max_depth_reached = 0

    def set_deadline(self, deadline: float) -> None:
        """Set a UNIX timestamp after which the search should stop."""
        self._deadline = deadline

    def select_action(self, game, training: bool = False):
        """Start full Minimax from the current position."""
        self.nodes_expanded    = 0
        self.max_depth_reached = 0
        me  = game.current_player
        opp = 3 - me
        best_score = -math.inf
        best_move  = None

        for move in game.legal_moves():
            game_clone = game.clone()
            game_clone.apply_move(move)
            score = self._minimax(game_clone, opp, me, is_max=False, depth=1)
            if score > best_score:
                best_score = score
                best_move  = move

        return best_move

    def _minimax(self, game, player, root_player, is_max: bool, depth: int) -> float:
        self.nodes_expanded   += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)

        # Time-check every 10 000 nodes (cheap amortised cost)
        if self._deadline and (self.nodes_expanded % 10_000 == 0):
            if time.time() > self._deadline:
                raise _Timeout()

        winner = game.winner()
        if winner == root_player:
            return 1000 - depth          # prefer faster wins
        if winner:
            return depth - 1000          # prefer slower losses
        if game.is_terminal():
            return 0                     # draw

        moves = game.legal_moves()
        if is_max:
            val = -math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = max(val, self._minimax(g, 3 - player, root_player, False, depth + 1))
            return val
        else:
            val = math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = min(val, self._minimax(g, 3 - player, root_player, True, depth + 1))
            return val
