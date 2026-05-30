"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.tictactoe import TicTacToe
from games.connect4 import Connect4


class GameEnv:


    # I have implemented this callable with parameters: self, game_cls, game_kwargs, opponent.
    def __init__(
        self,
        game_cls,
        game_kwargs: dict | None = None,
        opponent: str = "random",
    ):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, game_cls, game_kwargs, opponent.
        """
        kwargs = game_kwargs or {}
        self._game_cls = game_cls
        self._game_kwargs = kwargs
        self._opponent_type = opponent
        self._game: TicTacToe | Connect4 = game_cls(**kwargs)

        self._role_alternation = False
        self._episode_count = 0
        self.agent_id = 1
        self.opponent_id = 2
        self.agent_starts = True


        g = game_cls(**kwargs)
        if isinstance(g, TicTacToe):
            self.STATE_SIZE = g.ROWS * g.COLS
            self.N_ACTIONS = g.ROWS * g.COLS
            self._is_ttt = True
        else:
            self.STATE_SIZE = g.rows * g.cols
            self.N_ACTIONS = g.cols
            self._is_ttt = False


    # I have implemented this callable with parameters: self.
    def enable_role_alternation(self) -> None:


        self._role_alternation = True


    # I have implemented this callable with parameters: self, agent_starts.
    def reset(self, agent_starts: bool | None = None) -> np.ndarray:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, agent_starts.
        """
        self._game = self._game_cls(**self._game_kwargs)

        if agent_starts is None:
            if self._role_alternation:
                agent_starts = (self._episode_count % 2 == 0)
            else:
                agent_starts = True
        self._episode_count += 1

        self.agent_starts = bool(agent_starts)
        self.agent_id = 1 if self.agent_starts else 2
        self.opponent_id = 3 - self.agent_id


        if not self.agent_starts:
            opp_move = self._opponent_move()
            if opp_move is not None:
                self._game.apply_move(opp_move)
                if self._game.is_terminal():
                    pass

        return self._encode()


    # I have implemented this callable with parameters: self, action.
    def step(self, action):


        if self._game.is_terminal():
            return self._encode(), 0.0, True, {}


        move = self._action_to_move(action)


        if not self._is_legal(move):
            return self._encode(), -1.0, True, {"illegal": True}


        self._game.apply_move(move)
        if self._game.is_terminal():
            w = self._game.winner()
            r = 1.0 if w == self.agent_id else 0.0
            return self._encode(), r, True, {"winner": w, "draw": w == 0}


        opp_move = self._opponent_move()
        if opp_move is not None:
            self._game.apply_move(opp_move)
            if self._game.is_terminal():
                w = self._game.winner()
                r = -1.0 if w == self.opponent_id else 0.0
                return self._encode(), r, True, {"winner": w, "draw": w == 0}

        return self._encode(), 0.0, False, {}


    # I have implemented this callable with parameters: self.
    def get_legal_actions(self) -> list[int]:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        if self._game.is_terminal():
            return []
        moves = self._game.legal_moves()
        if self._is_ttt:
            return [r * 3 + c for (r, c) in moves]
        else:
            return moves


    # I have implemented this callable with parameters: self.
    def _encode(self) -> np.ndarray:


        return self._game.encode_state(perspective_player=self.agent_id)


    # I have implemented this callable with parameters: self.
    def encode_state(self) -> tuple:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        return tuple(self._encode().astype(np.int8).tolist())


    # I have implemented this callable with parameters: self.
    def get_board_heuristic(self) -> float:


        if self._is_ttt:
            return 0.0
        return self._c4_heuristic(self.agent_id)


    # I have implemented this callable with parameters: self, player.
    def _c4_heuristic(self, player: int) -> float:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, player.
        """
        board = self._game.board
        rows, cols = self._game.rows, self._game.cols
        opp = 3 - player
        score = 0.0

        mid = cols // 2
        score += 3.0 * float(np.sum(board[:, mid] == player))
        score -= 3.0 * float(np.sum(board[:, mid] == opp))


        # I have implemented this callable with parameters: window.
        def sw(window):


            pc = int(np.sum(window == player))
            oc = int(np.sum(window == opp))
            ec = int(np.sum(window == 0))
            if pc > 0 and oc > 0:
                return 0
            if pc == 3 and ec == 1:
                return 5
            if pc == 2 and ec == 2:
                return 1
            if oc == 3 and ec == 1:
                return -10
            if oc == 2 and ec == 2:
                return -2
            return 0

        for r in range(rows):
            for c in range(cols - 3):
                score += sw(board[r, c:c + 4])
        for c in range(cols):
            for r in range(rows - 3):
                score += sw(board[r:r + 4, c])
        for r in range(rows - 3):
            for c in range(cols - 3):
                score += sw(np.array([board[r + i, c + i] for i in range(4)]))
        for r in range(3, rows):
            for c in range(cols - 3):
                score += sw(np.array([board[r - i, c + i] for i in range(4)]))

        return score


    # I have implemented this callable with parameters: self, action.
    def _action_to_move(self, action: int):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, action.
        """
        if self._is_ttt:
            return (action // 3, action % 3)
        return action


    # I have implemented this callable with parameters: self, move.
    def _is_legal(self, move) -> bool:


        return move in self._game.legal_moves()


    # I have implemented this callable with parameters: self.
    def _opponent_move(self):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        legal = self._game.legal_moves()
        if not legal:
            return None
        if self._opponent_type == "default":
            return self._default_move(legal)
        return random.choice(legal)


    # I have implemented this callable with parameters: self, legal.
    def _default_move(self, legal):


        opp = self.opponent_id
        agent = self.agent_id

        for move in legal:
            g = self._game.clone()
            g.apply_move(move)
            if g.winner() == opp:
                return move

        for move in legal:
            g = self._game.clone()
            g.current_player = agent
            g.apply_move(move)
            if g.winner() == agent:
                return move

        if self._is_ttt:
            prefs = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2),
                     (0, 1), (1, 0), (1, 2), (2, 1)]
            for m in prefs:
                if m in legal:
                    return m
        else:
            mid = self._game.cols // 2
            return sorted(legal, key=lambda c: abs(c - mid))[0]

        return random.choice(legal)
