"""
games/connect4_ui.py
GUI for Connect 4 (optional demo mode).
"""

import tkinter as tk
from .connect4_core import Connect4


class Connect4UI(Connect4):
    """Connect 4 with GUI."""

    def __init__(self, rows=6, cols=7):
        super().__init__(rows, cols)
        self.window = tk.Tk()
        self.window.title("Connect 4")
        self.buttons = [None] * cols
        self.canvas = None
        self._setup_ui()

    def _setup_ui(self):
        # Column buttons
        for c in range(self.cols):
            btn = tk.Button(self.window, text=f'Col {c}', command=lambda c=c: self._on_click(c))
            btn.pack(side=tk.TOP)
            self.buttons[c] = btn

        # Canvas for board
        self.canvas = tk.Canvas(self.window, width=self.cols*50, height=self.rows*50)
        self.canvas.pack()
        self._draw_board()

    def _on_click(self, c):
        if c in self.legal_moves() and not self.is_terminal():
            self.apply_move(c)
            self._draw_board()
            # For demo, could add AI move here

    def _draw_board(self):
        self.canvas.delete("all")
        colors = {0: 'white', 1: 'red', 2: 'yellow'}
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c*50, r*50
                x2, y2 = x1+50, y1+50
                self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill=colors[self.board[r][c]])

    def run(self):
        self.window.mainloop()