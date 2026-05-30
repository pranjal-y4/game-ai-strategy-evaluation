"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
import math
from .base_agent import BaseAgent


class AlphaBetaAgent(BaseAgent):


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


        return f"AlphaBeta(d={self.max_depth})"


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
        alpha = -math.inf

        for move in self._order_moves(game, game.legal_moves()):
            g = game.clone()
            g.apply_move(move)
            score = self._alphabeta(g, root_player, False, self.max_depth - 1,
                                    alpha, math.inf)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move


    # I have implemented this callable with parameters: self, game, root_player, is_max, depth, alpha, beta.
    def _alphabeta(self, game, root_player: int, is_max: bool,
                   depth: int, alpha: float, beta: float) -> float:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, root_player, is_max, depth, alpha, beta.
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

        moves = self._order_moves(game, game.legal_moves())

        if is_max:
            val = -math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = max(val, self._alphabeta(g, root_player, False, depth - 1, alpha, beta))
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = math.inf
            for move in moves:
                g = game.clone()
                g.apply_move(move)
                val = min(val, self._alphabeta(g, root_player, True, depth - 1, alpha, beta))
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val


    # I have implemented this callable with parameters: self, game, moves.
    def _order_moves(self, game, moves):


        gtype = getattr(game, "game_type", "unknown")
        if gtype == "c4":
            mid = game.cols // 2
            return sorted(moves, key=lambda c: abs(c - mid))
        elif gtype == "ttt":
            pref = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2),
                    (0, 1), (1, 0), (1, 2), (2, 1)]
            order = {m: i for i, m in enumerate(pref)}
            return sorted(moves, key=lambda m: order.get(m, 99))
        return moves
