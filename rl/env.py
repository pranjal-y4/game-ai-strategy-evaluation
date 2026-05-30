"""
rl/env.py
────────────────────────────────────────────────────────────────────────────
Headless game environments for RL training (no tkinter).

Both environments expose a standard gym-like interface:
    env.reset()                    → state (np.ndarray)
    env.step(action)               → (next_state, reward, done, info)
    env.get_legal_actions()        → list[int]
    env.encode_state()             → hashable tuple  (for Q-table)
    env.render()                   → prints ASCII board to stdout

State encoding  (always from the AGENT's perspective)
──────────────────────────────────────────────────────
    +1  = agent's own piece
     0  = empty cell
    -1  = opponent's piece
"""

from __future__ import annotations
import numpy as np
import random


class TicTacToeEnv:
    ROWS = 3
    COLS = 3
    N_ACTIONS = 9
    STATE_SIZE = 9

    def __init__(self, opponent: str = "random"):
        self.opponent_type = opponent
        self._board = np.zeros(9, dtype=np.int8)   # 0 empty, 1/2 players
        self.done = False

        self._role_alternation = False
        self._episode_count = 0

        self.agent_id = 1
        self.opponent_id = 2
        self.agent_starts = True

    def enable_role_alternation(self) -> None:
        self._role_alternation = True

    def reset(self, agent_starts: bool | None = None) -> np.ndarray:
        self._board = np.zeros(9, dtype=np.int8)
        self.done = False

        if agent_starts is None:
            if self._role_alternation:
                agent_starts = (self._episode_count % 2 == 0)
                self._episode_count += 1
            else:
                agent_starts = True

        self.agent_starts = bool(agent_starts)
        self.agent_id = 1 if self.agent_starts else 2
        self.opponent_id = 2 if self.agent_starts else 1

        if not self.agent_starts:
            opp_action = self._opponent_action()
            if opp_action >= 0:
                self._board[opp_action] = self.opponent_id
                winner = self._winner()
                if winner == self.opponent_id or self._draw():
                    self.done = True

        return self._state_from_agent()

    def step(self, action: int):
        if self.done:
            return self._state_from_agent(), 0.0, True, {}

        if action < 0 or action >= self.N_ACTIONS or self._board[action] != 0:
            self.done = True
            return self._state_from_agent(), -0.5, True, {"illegal": True}

        self._board[action] = self.agent_id
        winner = self._winner()
        if winner == self.agent_id:
            self.done = True
            return self._state_from_agent(), 1.0, True, {"winner": self.agent_id}
        if self._draw():
            self.done = True
            return self._state_from_agent(), 0.2, True, {"draw": True}

        opp_action = self._opponent_action()
        if opp_action >= 0:
            self._board[opp_action] = self.opponent_id
            winner = self._winner()
            if winner == self.opponent_id:
                self.done = True
                return self._state_from_agent(), -1.0, True, {"winner": self.opponent_id}
            if self._draw():
                self.done = True
                return self._state_from_agent(), 0.2, True, {"draw": True}

        return self._state_from_agent(), 0.0, False, {}

    def get_legal_actions(self) -> list[int]:
        return [i for i in range(9) if self._board[i] == 0]

    def encode_state(self) -> tuple:
        return tuple(self._state_from_agent().astype(int).tolist())

    def render(self) -> None:
        symbols = {0: ".", 1: "X", 2: "O"}
        b = self._board.reshape(3, 3)
        for row in b:
            print("  " + " ".join(symbols[int(c)] for c in row))
        print()

    def _state_from_agent(self) -> np.ndarray:
        s = np.zeros_like(self._board, dtype=np.float32)
        s[self._board == self.agent_id] = 1.0
        s[self._board == self.opponent_id] = -1.0
        return s

    def _opponent_action(self) -> int:
        legal = self.get_legal_actions()
        if not legal:
            return -1
        if self.opponent_type == "semi":
            return self._semi_move(legal)
        return random.choice(legal)

    def _semi_move(self, legal: list[int]) -> int:
        for a in legal:
            self._board[a] = self.opponent_id
            if self._winner() == self.opponent_id:
                self._board[a] = 0
                return a
            self._board[a] = 0

        for a in legal:
            self._board[a] = self.agent_id
            if self._winner() == self.agent_id:
                self._board[a] = 0
                return a
            self._board[a] = 0

        return random.choice(legal)

    def _winner(self) -> int:
        b = self._board.reshape(3, 3)
        for i in range(3):
            if b[i, 0] == b[i, 1] == b[i, 2] != 0:
                return int(b[i, 0])
            if b[0, i] == b[1, i] == b[2, i] != 0:
                return int(b[0, i])
        if b[0, 0] == b[1, 1] == b[2, 2] != 0:
            return int(b[0, 0])
        if b[0, 2] == b[1, 1] == b[2, 0] != 0:
            return int(b[0, 2])
        return 0

    def _draw(self) -> bool:
        return 0 not in self._board and self._winner() == 0

    @property
    def board(self) -> np.ndarray:
        return self._board.reshape(self.ROWS, self.COLS).copy()


class Connect4Env:
    def __init__(self, rows: int = 4, cols: int = 5, opponent: str = "random"):
        self.rows = rows
        self.cols = cols
        self.N_ACTIONS = cols
        self.STATE_SIZE = rows * cols
        self.opponent_type = opponent
        self._board = np.zeros((rows, cols), dtype=np.int8)
        self.done = False

        self._role_alternation = False
        self._episode_count = 0

        self.agent_id = 1
        self.opponent_id = 2
        self.agent_starts = True

    def enable_role_alternation(self) -> None:
        self._role_alternation = True

    def reset(self, agent_starts: bool | None = None) -> np.ndarray:
        self._board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.done = False

        if agent_starts is None:
            if self._role_alternation:
                agent_starts = (self._episode_count % 2 == 0)
                self._episode_count += 1
            else:
                agent_starts = True

        self.agent_starts = bool(agent_starts)
        self.agent_id = 1 if self.agent_starts else 2
        self.opponent_id = 2 if self.agent_starts else 1

        if not self.agent_starts:
            legal = self.get_legal_actions()
            if legal:
                opp_action = self._opponent_action(legal)
                self._drop(opp_action, self.opponent_id)
                winner = self._winner()
                if winner == self.opponent_id or self._draw():
                    self.done = True

        return self._state_from_agent()

    def step(self, action: int):
        if self.done:
            return self._state_from_agent(), 0.0, True, {}

        legal = self.get_legal_actions()
        if action not in legal:
            self.done = True
            return self._state_from_agent(), -0.5, True, {"illegal": True}

        self._drop(action, self.agent_id)
        winner = self._winner()
        if winner == self.agent_id:
            self.done = True
            return self._state_from_agent(), 1.0, True, {"winner": self.agent_id}
        if self._draw():
            self.done = True
            return self._state_from_agent(), 0.2, True, {"draw": True}

        legal_opp = self.get_legal_actions()
        if legal_opp:
            opp_action = self._opponent_action(legal_opp)
            self._drop(opp_action, self.opponent_id)
            winner = self._winner()
            if winner == self.opponent_id:
                self.done = True
                return self._state_from_agent(), -1.0, True, {"winner": self.opponent_id}
            if self._draw():
                self.done = True
                return self._state_from_agent(), 0.2, True, {"draw": True}

        return self._state_from_agent(), 0.0, False, {}

    def get_legal_actions(self) -> list[int]:
        return [c for c in range(self.cols) if self._board[0][c] == 0]

    def encode_state(self) -> tuple:
        return tuple(self._state_from_agent().astype(int).tolist())

    def render(self) -> None:
        symbols = {0: ".", 1: "R", 2: "Y"}
        for row in self._board:
            print("  " + " ".join(symbols[int(c)] for c in row))
        print("  " + " ".join(str(c) for c in range(self.cols)))
        print()

    def _drop(self, col: int, player: int) -> None:
        for r in range(self.rows - 1, -1, -1):
            if self._board[r][col] == 0:
                self._board[r][col] = player
                return

    def _state_from_agent(self) -> np.ndarray:
        s = np.zeros_like(self._board, dtype=np.float32)
        s[self._board == self.agent_id] = 1.0
        s[self._board == self.opponent_id] = -1.0
        return s.flatten()

    def _opponent_action(self, legal: list[int]) -> int:
        if self.opponent_type == "semi":
            return self._semi_move(legal)
        return random.choice(legal)

    def _semi_move(self, legal: list[int]) -> int:
        for c in legal:
            row = self._drop_and_get_row(c, self.opponent_id)
            if self._winner() == self.opponent_id:
                self._board[row][c] = 0
                return c
            self._board[row][c] = 0

        for c in legal:
            row = self._drop_and_get_row(c, self.agent_id)
            if self._winner() == self.agent_id:
                self._board[row][c] = 0
                return c
            self._board[row][c] = 0

        return random.choice(legal)

    def _drop_and_get_row(self, col: int, player: int) -> int:
        for r in range(self.rows - 1, -1, -1):
            if self._board[r][col] == 0:
                self._board[r][col] = player
                return r
        return -1

    def _winner(self) -> int:
        b = self._board
        r, c = self.rows, self.cols
        for row in range(r):
            for col in range(c):
                p = b[row][col]
                if p == 0:
                    continue
                if col + 3 < c and all(b[row][col + i] == p for i in range(4)):
                    return int(p)
                if row + 3 < r and all(b[row + i][col] == p for i in range(4)):
                    return int(p)
                if row + 3 < r and col + 3 < c and all(b[row + i][col + i] == p for i in range(4)):
                    return int(p)
                if row + 3 < r and col - 3 >= 0 and all(b[row + i][col - i] == p for i in range(4)):
                    return int(p)
        return 0

    def _draw(self) -> bool:
        return len(self.get_legal_actions()) == 0 and self._winner() == 0

    @property
    def board(self) -> np.ndarray:
        return self._board.copy()