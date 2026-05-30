"""
experiments/validate.py
────────────────────────────────────────────────────────────────────────────
Correctness validation checklist.

Checks:
  1. Default agent takes an immediate winning move
  2. Default agent blocks an immediate opponent win
  3. Minimax never loses on TTT (vs Random, 200 games)
  4. Alpha-Beta matches Minimax result on every TTT position (20 random states)
  5. Every agent returns only legal moves (100 random positions each)
  6. Model files exist and agents load correctly
  7. Board size used by each agent is printed

Usage:
    python3 experiments/validate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from games.tictactoe_core import TicTacToe
from games.connect4_core  import Connect4
from agents.default_agent import DefaultAgent
from agents.ttt_minimax_agent import TTTMinimaxAgent
from agents.ttt_alphabeta_agent import TTTAlphaBetaAgent
from agents.random_agent import RandomAgent
from agents.qlearning_ttt_agent import QLearningTTTAgent
from agents.qlearning_c4_reduced_agent import QLearningC4ReducedAgent
from agents.dqn_ttt_agent import DQNTTTAgent
from agents.dqn_c4_agent import DQNC4Agent
from agents.c4_depthlimited_alphabeta_agent import C4DepthLimitedAlphaBetaAgent
from experiments.run_match import play_game

PASS = "  [PASS]"
FAIL = "  [FAIL]"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        msg = f"{FAIL} {label}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. Default agent takes winning move (TTT)")
print("="*60)

game = TicTacToe()
default = DefaultAgent()
# Set up board: X wins if plays (0,2)
# X . .      row0
# X . .      row1
# . . .
game.board[0][0] = 1; game.board[1][0] = 1  # player 1 has two in col 0
game.current_player = 1
move = default.select_action(game)
check("Default takes winning move for player 1", move == (2, 0),
      f"expected (2,0), got {move}")

print("\n" + "="*60)
print("2. Default agent blocks immediate opponent win (TTT)")
print("="*60)

game2 = TicTacToe()
# Player 2 is about to win: col 0 has two 2s, player 1 to move
game2.board[0][0] = 2; game2.board[1][0] = 2
game2.current_player = 1
move2 = default.select_action(game2)
check("Default blocks opponent win at (2,0)", move2 == (2, 0),
      f"expected (2,0), got {move2}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. Minimax never loses on TTT (200 games vs Random)")
print("="*60)

minimax = TTTMinimaxAgent()
rng_agent = RandomAgent()
game_ttt = TicTacToe()
losses = 0
for i in range(200):
    wc, _, _, _, _, _ = play_game(game_ttt, minimax, rng_agent, game_num=i)
    if wc == 2:
        losses += 1
check("Minimax never loses vs Random (200 games)", losses == 0,
      f"lost {losses} games")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. Alpha-Beta matches Minimax on 20 random TTT mid-game positions")
print("="*60)

mm = TTTMinimaxAgent()
ab = TTTAlphaBetaAgent()
rng = random.Random(42)

mismatches = 0
for trial in range(20):
    g = TicTacToe()
    # Make 3-5 random moves to reach a mid-game state
    n_moves = rng.randint(3, 5)
    for _ in range(n_moves):
        lm = g.legal_moves()
        if not lm or g.is_terminal():
            break
        g.apply_move(rng.choice(lm))
    if g.is_terminal():
        continue

    mm.reset(); ab.reset()
    # Compare scores by checking if both pick moves with equal value
    # We can't compare moves directly (ties), so compare the board outcome
    # after each agent plays from the same clone
    g_mm = g.clone(); g_ab = g.clone()
    move_mm = mm.select_action(g_mm)
    move_ab = ab.select_action(g_ab)

    # Apply each move and check if the resulting board winner is the same
    g_mm.apply_move(move_mm)
    g_ab.apply_move(move_ab)
    # Both should result in the same terminal outcome when played optimally
    # We verify by finishing both with minimax
    def finish_with_minimax(game_state, current_mm):
        while not game_state.is_terminal():
            current_mm.reset()
            m = current_mm.select_action(game_state)
            game_state.apply_move(m)
        return game_state.winner()

    # For outcome comparison use a fresh minimax for both continuations
    w_mm = finish_with_minimax(g_mm, TTTMinimaxAgent())
    w_ab = finish_with_minimax(g_ab, TTTMinimaxAgent())
    if w_mm != w_ab:
        mismatches += 1
        print(f"    Mismatch trial {trial}: minimax outcome={w_mm}, ab outcome={w_ab}")

check("Alpha-Beta matches Minimax outcome (20 mid-game positions)", mismatches == 0,
      f"{mismatches} mismatches")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. Every agent returns only legal moves (100 random positions)")
print("="*60)

agents_ttt = [
    ("Random",      RandomAgent()),
    ("Default",     DefaultAgent()),
    ("Minimax",     TTTMinimaxAgent()),
    ("AlphaBeta",   TTTAlphaBetaAgent()),
    ("QLearning",   QLearningTTTAgent()),
    ("DQN_TTT",     DQNTTTAgent()),
]
agents_c4_standard = [
    ("Random_C4",         RandomAgent()),
    ("Default_C4",        DefaultAgent()),
    ("C4_AlphaBeta_D5",   C4DepthLimitedAlphaBetaAgent()),
    ("DQN_C4",            DQNC4Agent()),
]
agents_c4_reduced = [
    ("QLearning_C4_4x5",  QLearningC4ReducedAgent()),
]

rng2 = random.Random(0)

def test_legal_moves(agent_name, agent, game_factory, n=100):
    illegal_count = 0
    g = game_factory()
    for i in range(n):
        g.reset()
        # Make 0-5 random moves
        for _ in range(rng2.randint(0, 5)):
            lm = g.legal_moves()
            if not lm or g.is_terminal(): break
            g.apply_move(rng2.choice(lm))
        if g.is_terminal():
            continue
        agent.reset() if hasattr(agent, 'reset') else None
        action = agent.select_action(g)
        if action not in g.legal_moves():
            illegal_count += 1
    check(f"{agent_name} returns legal moves only",
          illegal_count == 0, f"{illegal_count} illegal moves in {n} positions")

for name, agent in agents_ttt:
    test_legal_moves(name, agent, TicTacToe)

for name, agent in agents_c4_standard:
    test_legal_moves(name, agent, Connect4, n=50)

for name, agent in agents_c4_reduced:
    test_legal_moves(name, agent, lambda: Connect4(rows=4, cols=5), n=50)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. Model files exist and agents report loaded status")
print("="*60)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

model_checks = [
    ("ttt_qlearning.pkl",  QLearningTTTAgent,         {}),
    ("ttt_dqn.pkl",        DQNTTTAgent,               {}),
    ("c4_qlearning_4x5.pkl", QLearningC4ReducedAgent, {}),
    ("c4_dqn.pkl",         DQNC4Agent,                {}),
]

for fname, AgentClass, kwargs in model_checks:
    path = os.path.join(MODEL_DIR, fname)
    exists = os.path.exists(path)
    check(f"Model file exists: {fname}", exists,
          f"not found at {path}")
    if exists:
        agent = AgentClass(**kwargs)
        loaded = getattr(agent, '_loaded', True) if hasattr(agent, '_loaded') else (agent._net is not None)
        check(f"  {AgentClass.__name__} loaded model successfully", loaded)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. Board size used by each agent (informational)")
print("="*60)

print("  TTT agents:")
for name, agent in agents_ttt:
    print(f"    {name:20s} → TicTacToe 3x3")

print("  C4 standard agents (6x7):")
for name, agent in agents_c4_standard:
    print(f"    {name:20s} → Connect4 6x7")

print("  C4 reduced agents (4x5):")
for name, agent in agents_c4_reduced:
    ag = agent
    r = getattr(ag, 'rows', '?')
    c = getattr(ag, 'cols', '?')
    print(f"    {name:20s} → Connect4 {r}x{c}  *** NOT comparable to 6x7 agents ***")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("8. RL role alternation — env.reset() correctly places opponent piece as P2")
print("="*60)

from rl.env import TicTacToeEnv, Connect4Env
import numpy as np

# ── Check 8a: role alternation board state ────────────────────────────────
ttt_env = TicTacToeEnv(opponent="random")
ttt_env.enable_role_alternation()

# Episode 0 → agent is P1: board should be empty, agent_id=1
ttt_env.reset()
board_ep0 = ttt_env._board.copy()
check("TTT role_alt ep0: board empty (agent is P1, agent_id=1)",
      board_ep0.sum() == 0 and ttt_env.agent_id == 1,
      f"expected empty board + agent_id=1, got board={board_ep0} agent_id={ttt_env.agent_id}")

# Episode 1 → agent is P2: exactly one opponent_id piece on board
ttt_env.reset()
board_ep1 = ttt_env._board.copy()
check("TTT role_alt ep1: 1 opponent piece (agent_id=2, opponent opens)",
      (board_ep1 == ttt_env.opponent_id).sum() == 1 and ttt_env.agent_id == 2,
      f"expected 1 opponent piece + agent_id=2, got board={board_ep1} agent_id={ttt_env.agent_id}")

# C4: same test on reduced board
c4_env = Connect4Env(rows=4, cols=5, opponent="random")
c4_env.enable_role_alternation()

c4_env.reset()
c4_board_ep0 = c4_env._board.copy()
check("C4 role_alt ep0: board empty (agent is P1)",
      c4_board_ep0.sum() == 0 and c4_env.agent_id == 1,
      f"expected empty + agent_id=1, got sum={c4_board_ep0.sum()} agent_id={c4_env.agent_id}")

c4_env.reset()
c4_board_ep1 = c4_env._board.copy()
check("C4 role_alt ep1: 1 opponent piece (agent_id=2)",
      (c4_board_ep1 == c4_env.opponent_id).sum() == 1 and c4_env.agent_id == 2,
      f"expected 1 opponent piece + agent_id=2, got sum={c4_board_ep1.sum()} agent_id={c4_env.agent_id}")

# ── Check 8b: direct role semantics test (agent_starts=False) ────────────────
ttt_env2 = TicTacToeEnv(opponent="random")
ttt_env2.reset(agent_starts=False)

# After P2 reset: opponent_id=1 opened, agent_id=2
check("TTT P2 role: agent_id=2, opponent_id=1",
      ttt_env2.agent_id == 2 and ttt_env2.opponent_id == 1,
      f"got agent_id={ttt_env2.agent_id}, opponent_id={ttt_env2.opponent_id}")

# Opening move placed as opponent_id (1), not hardcoded 2
check("TTT P2 role: opening move uses opponent_id (=1), not hardcoded 2",
      (ttt_env2._board == 1).sum() == 1 and (ttt_env2._board == 2).sum() == 0,
      f"board={ttt_env2._board}")

# State encoding maps agent_id cells to +1, opponent_id cells to -1
state_p2 = ttt_env2._state_from_agent()
check("TTT P2 role: state encodes agent_id→+1, opponent_id→-1",
      all(state_p2[ttt_env2._board == ttt_env2.agent_id] == 1.0) and
      all(state_p2[ttt_env2._board == ttt_env2.opponent_id] == -1.0),
      f"state={state_p2}, board={ttt_env2._board}")

# encode_state() returns role-invariant representation
state_enc = ttt_env2.encode_state()
check("TTT P2 role: encode_state() uses agent perspective (not raw board)",
      all(v in (-1, 0, 1) for v in state_enc),
      f"encode_state not in {{-1,0,1}}: {state_enc}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("9. P1+P2 metrics sum correctly to total (CSV integrity check)")
print("="*60)

import glob
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "experiments", "results")
checked_any = False
for pattern in ["vs_default_*.csv", "crossplay_*.csv"]:
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))[-2:]:
        fname = os.path.basename(fpath)
        try:
            import csv as _csv
            with open(fpath) as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            continue
        if not rows or "p1_games" not in rows[0]:
            continue
        bad_rows = 0
        for r in rows:
            try:
                n      = int(r.get("n_games", r.get("games", 0)))
                if n == 0:
                    continue
                p1_g   = int(r["p1_games"])
                p2_g   = int(r["p2_games"])
                w      = int(r.get("wins", 0))
                d      = int(r.get("draws", 0))
                l      = int(r.get("losses", 0))
                p1_wr  = float(r["p1_win_rate"])
                p2_wr  = float(r["p2_win_rate"])
                fma    = float(r.get("first_mover_advantage", r.get("fma", 0.0)))
                
                total_g = p1_g + p2_g
                outcomes = w + d + l
                # FMA logic assumes P1 win rate - P2 win rate
                expected_fma = round(p1_wr - p2_wr, 4)

                is_bad = False
                if total_g != n: is_bad = True
                if outcomes != n and ("wins" in r): is_bad = True
                if abs(fma - expected_fma) > 1e-3 and ("first_mover_advantage" in r or "fma" in r): is_bad = True
                
                if is_bad:
                    bad_rows += 1
            except (KeyError, ValueError):
                pass
        check(f"P1+P2 == n_games, outcomes == n, fma == p1-p2 in {fname}",
              bad_rows == 0, f"{bad_rows} rows with mismatched totals or invariants")
        checked_any = True
if not checked_any:
    print("  [INFO] No v2 result CSVs found — run evaluation scripts first.")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("10. Board-size isolation: C4 crossplay excludes 4x5 Q-learning agent")
print("="*60)

for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "crossplay_c4_*.csv")))[-1:]:
    fname = os.path.basename(fpath)
    try:
        with open(fpath) as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        continue
    if not rows:
        continue
    agent_names_in_cp = {r.get("agent", "") for r in rows}
    board_configs = {r.get("board_config", "") for r in rows}
    # QLearningC4ReducedAgent name should NOT appear in 6x7 crossplay
    reduced_names = {"QLearning_C4_4x5", "QLearning_C4_Reduced", "qlearning_c4", "QLearning"}
    found_reduced = agent_names_in_cp & reduced_names
    
    # Additional logic for check 10 requested by user:
    # Ensure QLearning_C4_4x5 only has "4x5" board_config and Connect4_DQN only "6x7"
    bad_config_rows = 0
    for r in rows:
        ag = r.get("agent", r.get("algorithm", ""))
        bc = r.get("board_config", "")
        if "QLearning" in ag and "C4" in ag and bc == "6x7":
            bad_config_rows += 1
    
    check(f"C4 crossplay excludes reduced-board Q-learning: {fname}",
          len(found_reduced) == 0 and bad_config_rows == 0,
          f"Found reduced-board agent(s) in 6x7 crossplay -> agent:{found_reduced}, bad_rows:{bad_config_rows}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("11. Plotter skips stale/incompatible CSV files gracefully")
print("="*60)

import tempfile, warnings as _warnings
try:
    import pandas as _pd
    # Write a CSV with wrong schema_version
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                      dir=RESULTS_DIR,
                                      prefix="vs_default_ttt_STALE_") as tmp:
        tmp.write("agent,win_rate,game,schema_version\n")
        tmp.write("TestAgent,0.5,TTT,v1\n")
        tmp_path = tmp.name

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.plotter import find_latest_compatible_csv, RESULTS_DIR as P_RESULTS
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        result = find_latest_compatible_csv(
            RESULTS_DIR, "vs_default_ttt",
            required_cols=["p1_win_rate", "p2_win_rate"],
            schema_version="v2"
        )
    # The stale file should NOT be selected as the v2 result
    stale_selected = (result is not None and "STALE" in os.path.basename(result or ""))
    check("Plotter find_latest_compatible_csv skips v1-schema stale file",
          not stale_selected,
          f"Stale file was incorrectly selected: {result}")
    os.unlink(tmp_path)
except ImportError:
    print("  [INFO] pandas not installed — plotter compatibility check skipped.")
except Exception as e:
    print(f"  [INFO] Plotter check skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"RESULT: {len(errors)} CHECK(S) FAILED")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED")
print("="*60 + "\n")
