"""
games/connect4_core.py
Headless Connect 4 game class for evaluation and training.
"""

import copy


class Connect4:
    """Headless Connect 4 game."""

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def clone(self):
        """Return a deep copy of the game state."""
        return copy.deepcopy(self)

    def legal_moves(self):
        """Return list of legal moves as column indices."""
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def apply_move(self, move):
        """Apply move (column) for current player."""
        col = move
        row = self._get_drop_row(col)
        if row == -1:
            raise ValueError("Invalid move")
        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        self.current_player = 3 - self.current_player

    def undo_move(self, move):
        """Undo the last move."""
        col = move
        for r in range(self.rows):
            if self.board[r][col] != 0:
                self.board[r][col] = 0
                self.move_count -= 1
                self.current_player = 3 - self.current_player
                self.last_move = None
                return

    def is_terminal(self):
        """Return True if game is over."""
        return self.winner() != 0 or len(self.legal_moves()) == 0

    def winner(self):
        """Return winning player (1 or 2) or 0 if no winner."""
        if not self.last_move:
            return 0
        r, c = self.last_move
        player = self.board[r][c]
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                nr, nc = r + dr*i, c + dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                nr, nc = r - dr*i, c - dc*i
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            if count >= 4:
                return player
        return 0

    def encode_state(self):
        """Encode state for RL as flattened tuple."""
        return tuple(self.board[r][c] for r in range(self.rows) for c in range(self.cols))

    def reset(self):
        """Reset to initial state."""
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.current_player = 1
        self.move_count = 0
        self.last_move = None

    def render_text(self):
        """Text representation for debugging."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        return '\n'.join(' '.join(symbols[cell] for cell in row) for row in self.board)

    def _get_drop_row(self, col):
        """Get the row where a piece would drop in the given column."""
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return -1