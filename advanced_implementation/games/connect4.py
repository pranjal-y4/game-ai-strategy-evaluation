"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import numpy as np


class Connect4:


    game_type = "c4"


    # I have implemented this callable with parameters: self, rows, cols.
    def __init__(self, rows: int = 6, cols: int = 7):


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, rows, cols.
        """
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.current_player: int = 1
        self.move_count: int = 0
        self._winner: int | None = None
        self._terminal: bool = False
        self._last_row: int = -1
        self._last_col: int = -1


    # I have implemented this callable with parameters: self.
    def reset(self) -> "Connect4":


        self.board[:] = 0
        self.current_player = 1
        self.move_count = 0
        self._winner = None
        self._terminal = False
        self._last_row = -1
        self._last_col = -1
        return self


    # I have implemented this callable with parameters: self.
    def clone(self) -> "Connect4":


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        g = Connect4.__new__(Connect4)
        g.rows = self.rows
        g.cols = self.cols
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_count = self.move_count
        g._winner = self._winner
        g._terminal = self._terminal
        g._last_row = self._last_row
        g._last_col = self._last_col
        return g


    # I have implemented this callable with parameters: self.
    def legal_moves(self) -> list[int]:


        return [c for c in range(self.cols) if self.board[0, c] == 0]


    # I have implemented this callable with parameters: self, col.
    def get_drop_row(self, col: int) -> int:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, col.
        """
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return -1


    # I have implemented this callable with parameters: self, col.
    def apply_move(self, col: int) -> int:


        if col < 0 or col >= self.cols or self.board[0, col] != 0:
            raise ValueError(f"Illegal move: column {col}")
        row = self.get_drop_row(col)
        self.board[row, col] = self.current_player
        self.move_count += 1
        self._last_row = row
        self._last_col = col
        self._update_terminal(row, col)
        if not self._terminal:
            self.current_player = 3 - self.current_player
        return row


    # I have implemented this callable with parameters: self, row, col.
    def _update_terminal(self, row: int, col: int) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, row, col.
        """
        if self._check_win(row, col, self.current_player):
            self._winner = self.current_player
            self._terminal = True
        elif self.move_count == self.rows * self.cols:
            self._winner = 0
            self._terminal = True


    # I have implemented this callable with parameters: self, row, col, player.
    def _check_win(self, row: int, col: int, player: int) -> bool:


        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r += dr; c += dc

            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                count += 1
                r -= dr; c -= dc
            if count >= 4:
                return True
        return False


    # I have implemented this callable with parameters: self.
    def is_terminal(self) -> bool:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        return self._terminal


    # I have implemented this callable with parameters: self.
    def winner(self) -> int | None:


        return self._winner


    # I have implemented this callable with parameters: self, perspective_player.
    def encode_state(self, perspective_player: int | None = None) -> np.ndarray:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, perspective_player.
        """
        p = perspective_player if perspective_player is not None else self.current_player
        flat = self.board.flatten().astype(np.float32)
        return np.where(flat == p, 1.0, np.where(flat == 0, 0.0, -1.0))


    # I have implemented this callable with parameters: self, perspective_player.
    def encode_state_hashable(self, perspective_player: int | None = None) -> tuple:


        return tuple(self.encode_state(perspective_player).astype(np.int8).tolist())


    # I have implemented this callable with parameters: self.
    def render_text(self) -> str:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[int(self.board[r, c])] for c in range(self.cols))
                for r in range(self.rows)]
        rows.append(" ".join(str(c) for c in range(self.cols)))
        return "\n".join(rows)
