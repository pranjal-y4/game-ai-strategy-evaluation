"""
agents/default_agent.py
Default agent: better than random, takes wins and blocks.
"""

import random
from .base_agent import BaseAgent


class DefaultAgent(BaseAgent):
    """Smart default opponent: win > block > fallback."""

    @property
    def name(self):
        return "Default"

    def reset(self):
        pass

    def select_action(self, game, training=False):
        me = game.current_player
        opp = 3 - me
        moves = game.legal_moves()

        # 1. Win if possible
        for move in moves:
            if self._would_win(game, move, me):
                return move

        # 2. Block opponent win
        for move in moves:
            if self._would_win(game, move, opp):
                return move

        # 3. Fallback policy
        return self._fallback(game, moves)

    def _would_win(self, game, move, player):
        """Check if move would win for the specified player."""
        game_clone = game.clone()
        game_clone.current_player = player
        game_clone.apply_move(move)
        return game_clone.winner() == player

    def _fallback(self, game, moves):
        """Fallback policy: center preference for C4, corners for TTT."""
        if hasattr(game, 'cols'):  # Connect 4
            center = game.cols // 2
            order = [center]
            for i in range(1, game.cols):
                if center - i >= 0:
                    order.append(center - i)
                if center + i < game.cols:
                    order.append(center + i)
            for col in order:
                if col in moves:
                    return col
        else:  # Tic-Tac-Toe
            # Center, then corners, then edges
            preferred = [(1,1), (0,0), (0,2), (2,0), (2,2), (0,1), (1,0), (1,2), (2,1)]
            for r, c in preferred:
                if (r, c) in moves:
                    return (r, c)
        return random.choice(moves) if moves else None