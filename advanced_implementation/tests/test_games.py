import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
import numpy as np


def test_ttt_initial_state():
    g = TicTacToe()
    assert g.current_player == 1
    assert g.move_count == 0
    assert len(g.legal_moves()) == 9
    assert not g.is_terminal()
    assert g.winner() is None
    print("  [PASS] ttt_initial_state")


def test_ttt_win_detection_row():
    g = TicTacToe()
    for c in range(3):
        g.board[0, c] = 1
    g.board[0, 2] = 1

    g2 = TicTacToe()
    g2.apply_move((0, 0)); g2.apply_move((1, 0))
    g2.apply_move((0, 1)); g2.apply_move((1, 1))
    g2.apply_move((0, 2))
    assert g2.winner() == 1
    assert g2.is_terminal()
    print("  [PASS] ttt_win_detection_row")


def test_ttt_win_detection_col():
    g = TicTacToe()
    g.apply_move((0, 0)); g.apply_move((0, 1))
    g.apply_move((1, 0)); g.apply_move((1, 1))
    g.apply_move((2, 0))
    assert g.winner() == 1
    print("  [PASS] ttt_win_detection_col")


def test_ttt_win_detection_diagonal():
    g = TicTacToe()
    g.apply_move((0, 0)); g.apply_move((0, 1))
    g.apply_move((1, 1)); g.apply_move((0, 2))
    g.apply_move((2, 2))
    assert g.winner() == 1
    print("  [PASS] ttt_win_detection_diagonal")


def test_ttt_draw():
    g = TicTacToe()

    moves = [(0,0),(0,1),(0,2),(1,0),(1,1),(2,0),(1,2),(2,2),(2,1)]
    for m in moves:
        g.apply_move(m)
    assert g.is_terminal()
    assert g.winner() == 0
    print("  [PASS] ttt_draw")


def test_ttt_clone_independence():
    g = TicTacToe()
    g.apply_move((0, 0))
    g2 = g.clone()
    g2.apply_move((0, 1))
    assert g.board[0, 1] == 0
    print("  [PASS] ttt_clone_independence")


def test_ttt_no_illegal_moves():
    g = TicTacToe()
    g.apply_move((1, 1))
    try:
        g.apply_move((1, 1))
        print("  [FAIL] ttt_no_illegal_moves: no exception raised")
        return
    except ValueError:
        pass
    print("  [PASS] ttt_no_illegal_moves")


def test_ttt_state_encoding():
    g = TicTacToe()
    g.apply_move((0, 0))
    state = g.encode_state(perspective_player=1)
    assert state[0] == 1.0
    assert state.shape == (9,)
    state2 = g.encode_state(perspective_player=2)
    assert state2[0] == -1.0
    print("  [PASS] ttt_state_encoding")


def test_c4_gravity():
    g = Connect4(rows=4, cols=5)
    row = g.apply_move(2)
    assert row == 3
    row2 = g.apply_move(2)
    assert row2 == 2
    print("  [PASS] c4_gravity")


def test_c4_win_horizontal():
    g = Connect4(rows=6, cols=7)

    g.board[5, 0] = 1; g.board[5, 1] = 1; g.board[5, 2] = 1
    g.board[5, 3] = 1
    assert g._check_win(5, 3, 1)
    print("  [PASS] c4_win_horizontal")


def test_c4_win_vertical():
    g = Connect4(rows=6, cols=7)
    g.board[5, 3] = 1; g.board[4, 3] = 1
    g.board[3, 3] = 1; g.board[2, 3] = 1
    assert g._check_win(2, 3, 1)
    print("  [PASS] c4_win_vertical")


def test_c4_win_diagonal():
    g = Connect4(rows=6, cols=7)

    for i in range(4):
        g.board[5 - i, i] = 1
    assert g._check_win(5, 0, 1)
    print("  [PASS] c4_win_diagonal")


def test_c4_no_false_wins():
    g = Connect4(rows=6, cols=7)
    g.board[5, 0] = 1; g.board[5, 1] = 1; g.board[5, 2] = 1
    assert not g._check_win(5, 2, 1)
    print("  [PASS] c4_no_false_wins")


def test_c4_legal_moves_after_fill():
    g = Connect4(rows=2, cols=3)
    for _ in range(6):
        legal = g.legal_moves()
        if legal:
            g.apply_move(legal[0])
    assert g.legal_moves() == [] or g.is_terminal()
    print("  [PASS] c4_legal_moves_after_fill")


def test_c4_clone_independence():
    g = Connect4(rows=4, cols=5)
    g.apply_move(2)
    original_board = g.board.copy()
    g2 = g.clone()
    g2.apply_move(3)

    assert np.array_equal(g.board, original_board), "Clone modified original board"
    print("  [PASS] c4_clone_independence")


def test_c4_state_encoding_perspective():
    g = Connect4(rows=4, cols=5)
    g.apply_move(0)
    s1 = g.encode_state(perspective_player=1)
    s2 = g.encode_state(perspective_player=2)
    assert s1[g.cols * 3 + 0] == 1.0
    assert s2[g.cols * 3 + 0] == -1.0
    print("  [PASS] c4_state_encoding_perspective")


def test_c4_play_full_game():

    g = Connect4(rows=4, cols=5)
    import random as rng
    rng.seed(1)
    while not g.is_terminal():
        moves = g.legal_moves()
        g.apply_move(rng.choice(moves))
    assert g.winner() in [0, 1, 2]
    print("  [PASS] c4_play_full_game")


def test_c4_configurable_board():

    g = Connect4(rows=3, cols=4)
    assert g.rows == 3 and g.cols == 4
    assert len(g.legal_moves()) == 4
    print("  [PASS] c4_configurable_board")


if __name__ == "__main__":
    print("\n=== TicTacToe Tests ===")
    test_ttt_initial_state()
    test_ttt_win_detection_row()
    test_ttt_win_detection_col()
    test_ttt_win_detection_diagonal()
    test_ttt_draw()
    test_ttt_clone_independence()
    test_ttt_no_illegal_moves()
    test_ttt_state_encoding()

    print("\n=== Connect4 Tests ===")
    test_c4_gravity()
    test_c4_win_horizontal()
    test_c4_win_vertical()
    test_c4_win_diagonal()
    test_c4_no_false_wins()
    test_c4_legal_moves_after_fill()
    test_c4_clone_independence()
    test_c4_state_encoding_perspective()
    test_c4_play_full_game()
    test_c4_configurable_board()

    print("\nAll game tests passed.")
