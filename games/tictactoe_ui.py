"""
games/tictactoe_ui.py
GUI for Tic-Tac-Toe (optional demo mode).
"""

import tkinter as tk
from .tictactoe_core import TicTacToe


class TicTacToeUI(TicTacToe):
    """Tic-Tac-Toe with GUI."""

    def __init__(self):
        super().__init__()
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe")
        self.buttons = [[None]*3 for _ in range(3)]
        self._setup_ui()

    def _setup_ui(self):
        for r in range(3):
            for c in range(3):
                btn = tk.Button(self.window, text=' ', font=('Arial', 20), width=5, height=2,
                               command=lambda r=r, c=c: self._on_click(r, c))
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

    def _on_click(self, r, c):
        if self.board[r][c] == 0 and not self.is_terminal():
            self.apply_move((r, c))
            self._update_display()
            # For demo, could add AI move here

    def _update_display(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text=symbols[self.board[r][c]])

    def run(self):
        self.window.mainloop()