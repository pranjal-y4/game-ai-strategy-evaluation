"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import numpy as np


class TicTacToe:


    ROWS = 3
    COLS = 3
    game_type = "ttt"


    _PREF = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2),
             (0, 1), (1, 0), (1, 2), (2, 1)]


    # I have implemented this callable with parameters: self.
    def __init__(self):


        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player: int = 1
        self.move_count: int = 0
        self._winner: int | None = None
        self._terminal: bool = False


    # I have implemented this callable with parameters: self.
    def reset(self) -> "TicTacToe":


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        self.board[:] = 0
        self.current_player = 1
        self.move_count = 0
        self._winner = None
        self._terminal = False
        return self


    # I have implemented this callable with parameters: self.
    def clone(self) -> "TicTacToe":


        g = TicTacToe.__new__(TicTacToe)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_count = self.move_count
        g._winner = self._winner
        g._terminal = self._terminal
        return g


    # I have implemented this callable with parameters: self.
    def legal_moves(self) -> list[tuple[int, int]]:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        return [(r, c) for r in range(3) for c in range(3)
                if self.board[r, c] == 0]


    # I have implemented this callable with parameters: self, move.
    def apply_move(self, move: tuple[int, int]) -> "TicTacToe":


        r, c = move
        if self.board[r, c] != 0:
            raise ValueError(f"Illegal move {move}: cell occupied")
        self.board[r, c] = self.current_player
        self.move_count += 1
        self._update_terminal(r, c)
        if not self._terminal:
            self.current_player = 3 - self.current_player
        return self


    # I have implemented this callable with parameters: self, r, c.
    def _update_terminal(self, r: int, c: int) -> None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, r, c.
        """
        p = self.current_player
        b = self.board
        if (all(b[r, :] == p) or all(b[:, c] == p) or
                (r == c and all(b[i, i] == p for i in range(3))) or
                (r + c == 2 and all(b[i, 2 - i] == p for i in range(3)))):
            self._winner = p
            self._terminal = True
        elif self.move_count == 9:
            self._winner = 0
            self._terminal = True


    # I have implemented this callable with parameters: self.
    def is_terminal(self) -> bool:


        return self._terminal


    # I have implemented this callable with parameters: self.
    def winner(self) -> int | None:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self.
        """
        return self._winner


    # I have implemented this callable with parameters: self, perspective_player.
    def encode_state(self, perspective_player: int | None = None) -> np.ndarray:


        p = perspective_player if perspective_player is not None else self.current_player
        flat = self.board.flatten().astype(np.float32)
        return np.where(flat == p, 1.0, np.where(flat == 0, 0.0, -1.0))


    # I have implemented this callable with parameters: self, perspective_player.
    def encode_state_hashable(self, perspective_player: int | None = None) -> tuple:


        """
        I have implemented this function with a clearer note.
        Parameters used here: self, perspective_player.
        """
        return tuple(self.encode_state(perspective_player).astype(np.int8).tolist())


    # I have implemented this callable with parameters: self.
    def render_text(self) -> str:


        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[int(self.board[r, c])] for c in range(3))
                for r in range(3)]
        return "\n".join(rows)
