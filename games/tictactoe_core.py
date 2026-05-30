"""
games/tictactoe_core.py
Headless Tic-Tac-Toe game class for evaluation and training.
"""

import copy


class TicTacToe:
    """Headless Tic-Tac-Toe game."""

    def __init__(self):
        self.reset()

    def clone(self):
        """Return a deep copy of the game state."""
        return copy.deepcopy(self)

    def legal_moves(self):
        """Return list of legal moves as (row, col) tuples."""
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == 0]

    def apply_move(self, move):
        """Apply move (row, col) for current player."""
        r, c = move
        if self.board[r][c] != 0:
            raise ValueError("Invalid move")
        self.board[r][c] = self.current_player
        self.move_count += 1
        self.current_player = 3 - self.current_player

    def undo_move(self, move):
        """Undo the last move."""
        r, c = move
        self.board[r][c] = 0
        self.move_count -= 1
        self.current_player = 3 - self.current_player

    def is_terminal(self):
        """Return True if game is over."""
        return self.winner() != 0 or len(self.legal_moves()) == 0

    def winner(self):
        """Return winning player (1 or 2) or 0 if no winner."""
        b = self.board
        lines = [
            [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
            [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
            [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],
        ]
        for line in lines:
            vals = [b[r][c] for r, c in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return 0

    def encode_state(self):
        """Encode state for RL as flattened tuple."""
        return tuple(self.board[r][c] for r in range(3) for c in range(3))

    def reset(self):
        """Reset to initial state."""
        self.board = [[0] * 3 for _ in range(3)]
        self.current_player = 1
        self.move_count = 0

    def render_text(self):
        """Text representation for debugging."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        return '\n'.join(' '.join(symbols[cell] for cell in row) for row in self.board)