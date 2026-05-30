"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import math
import time
import numpy as np
from .base_agent import BaseAgent


_EXACT = 0
_LOWER = 1
_UPPER = 2


class _Timeout(Exception):


    pass


class AdvancedAlphaBetaC4Agent(BaseAgent):


    # I have implemented this callable with parameters: self, max_depth, time_budget.
    def __init__(self, max_depth: int = 8, time_budget: float | None = None):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, max_depth, time_budget.
        """
        self.max_depth = max_depth
        self.time_budget = time_budget

        self.nodes_searched: int = 0
        self.time_used: float = 0.0
        self.depth_reached: int = 0
        self._tt: dict = {}

    @property


    # I have implemented this callable with parameters: self.
    def name(self) -> str:


        tb = f",t={self.time_budget:.1f}s" if self.time_budget else ""
        return f"AdvAB_C4(d={self.max_depth}{tb})"


    # I have implemented this callable with parameters: self.
    def reset(self) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        self.nodes_searched = 0
        self.time_used = 0.0
        self.depth_reached = 0


    # I have implemented this callable with parameters: self, game, training.
    def select_action(self, game, training: bool = False):


        self.nodes_searched = 0
        self._tt.clear()
        start = time.time()
        root_player = game.current_player
        best_move = game.legal_moves()[0]


        for depth in range(1, self.max_depth + 1):
            try:
                move, _ = self._root_search(game, root_player, depth, start)
                best_move = move
                self.depth_reached = depth
            except _Timeout:
                break

        self.time_used = time.time() - start
        return best_move


    # I have implemented this callable with parameters: self, game, root_player, depth, start_time.
    def _root_search(self, game, root_player: int, depth: int,
                     start_time: float) -> tuple:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, root_player, depth, start_time.
        """
        alpha = -math.inf
        beta = math.inf
        best_move = None
        best_val = -math.inf

        for move in self._order_moves(game, game.legal_moves(), root_player):
            self._check_time(start_time)
            g = game.clone()
            g.apply_move(move)
            val = self._alphabeta(g, depth - 1, alpha, beta,
                                  False, root_player, start_time)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)

        return best_move, best_val


    # I have implemented this callable with parameters: self, game, depth, alpha, beta, is_max, root_player, start_time.
    def _alphabeta(self, game, depth: int, alpha: float, beta: float,
                   is_max: bool, root_player: int, start_time: float) -> float:


        self._check_time(start_time)
        self.nodes_searched += 1
        alpha_orig = alpha


        key = game.board.tobytes()
        if key in self._tt:
            tt_depth, tt_flag, tt_val, _ = self._tt[key]
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    return tt_val
                elif tt_flag == _LOWER:
                    alpha = max(alpha, tt_val)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_val)
                if alpha >= beta:
                    return tt_val


        w = game.winner()
        if w is not None:
            if w == root_player:
                return 100_000 + depth
            elif w == 0:
                return 0
            else:
                return -(100_000 + depth)

        if depth == 0:
            return self._heuristic(game, root_player)


        moves = self._order_moves(game, game.legal_moves(), root_player)
        best_val = -math.inf if is_max else math.inf
        best_move = moves[0] if moves else None

        for move in moves:
            g = game.clone()
            g.apply_move(move)
            val = self._alphabeta(g, depth - 1, alpha, beta,
                                  not is_max, root_player, start_time)
            if is_max:
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)

            if alpha >= beta:
                break


        if best_val <= alpha_orig:
            flag = _UPPER
        elif best_val >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (depth, flag, best_val, best_move)

        return best_val


    # I have implemented this callable with parameters: self, game, moves, root_player.
    def _order_moves(self, game, moves: list[int], root_player: int) -> list[int]:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game, moves, root_player.
        """
        current = game.current_player
        opponent = 3 - current
        mid = game.cols // 2

        winning, blocking, other = [], [], []

        for col in moves:
            g = game.clone()
            g.apply_move(col)
            if g.winner() == current:
                winning.append(col)
                continue


            g2 = game.clone()
            g2.current_player = opponent
            g2.apply_move(col)
            if g2.winner() == opponent:
                blocking.append(col)
                continue

            other.append(col)


        key = lambda c: abs(c - mid)
        return (sorted(winning, key=key) +
                sorted(blocking, key=key) +
                sorted(other, key=key))


    # I have implemented this callable with parameters: self, game, player.
    def _heuristic(self, game, player: int) -> float:


        board = game.board
        rows, cols = game.rows, game.cols
        opp = 3 - player
        score = 0.0


        mid = cols // 2
        for c in range(max(0, mid - 1), min(cols, mid + 2)):
            weight = 4 if c == mid else 2
            score += weight * np.sum(board[:, c] == player)
            score -= weight * np.sum(board[:, c] == opp)


        # I have implemented this callable with parameters: window.
        def score_window(window):


            """
            I have implemented this function with a clearer note.
            Parameters used here: window.
            """
            pc = np.sum(window == player)
            oc = np.sum(window == opp)
            ec = np.sum(window == 0)
            if pc > 0 and oc > 0:
                return 0
            if pc == 4:
                return 1000
            if pc == 3 and ec == 1:
                return 10
            if pc == 2 and ec == 2:
                return 2
            if oc == 4:
                return -1000
            if oc == 3 and ec == 1:
                return -50
            if oc == 2 and ec == 2:
                return -3
            return 0


        for r in range(rows):
            for c in range(cols - 3):
                score += score_window(board[r, c:c + 4])


        for c in range(cols):
            for r in range(rows - 3):
                score += score_window(board[r:r + 4, c])


        for r in range(rows - 3):
            for c in range(cols - 3):
                w = np.array([board[r + i, c + i] for i in range(4)])
                score += score_window(w)


        for r in range(3, rows):
            for c in range(cols - 3):
                w = np.array([board[r - i, c + i] for i in range(4)])
                score += score_window(w)

        return float(score)


    # I have implemented this callable with parameters: self, start_time.
    def _check_time(self, start_time: float) -> None:


        if self.time_budget and (time.time() - start_time) > self.time_budget:
            raise _Timeout()
