# Validation Checklist

## Game Environment Checks

- [x] TicTacToe legal moves include only empty cells
- [x] TicTacToe winner detection: rows, columns, diagonals
- [x] TicTacToe draw detection after 9 moves
- [x] TicTacToe clone does not affect original
- [x] Connect4 gravity: piece falls to lowest empty row
- [x] Connect4 winner detection: horizontal, vertical, diagonal
- [x] Connect4 no false wins with fewer than 4 in a row
- [x] Connect4 legal_moves() returns [] when all columns full
- [x] Connect4 configurable board size (rows, cols)
- [x] State encoding: +1=own, 0=empty, -1=opponent (from agent's perspective)

## Agent Correctness Checks

- [x] All agents select only legal moves (TTT and C4, 50 random positions)
- [x] Default agent takes immediate win when available
- [x] Default agent blocks opponent's immediate win
- [x] Minimax never loses in TicTacToe vs Random (100 games)
- [x] Alpha-Beta produces same result as Minimax on same TTT positions
- [x] Advanced Alpha-Beta always returns a valid move
- [x] Advanced Alpha-Beta takes immediate winning move
- [x] Iterative deepening: depth 1 always completes (valid fallback move guaranteed)
- [x] Transposition table cleared at each root call (no stale reuse)

## RL Component Checks

- [x] PER buffer samples with correct proportional priority
- [x] IS weights are in [0, 1] range after normalisation
- [x] Priority update after TD error computation
- [x] N-step buffer returns R_n = r_t + γ*r_{t+1} + γ²*r_{t+2}
- [x] N-step buffer flushes correctly at episode end (done=True)
- [x] GameEnv always returns legal actions ≥ 1 (unless terminal)
- [x] GameEnv role alternation: even episodes = P1, odd = P2
- [x] Epsilon = 0 during evaluation mode
- [x] Epsilon restored after evaluation (no side effects)
- [x] Double DQN: online and target networks are separate objects
- [x] Double DQN: initial weights identical (after constructor)
- [x] Target network syncs every target_update steps

## Training Pipeline Checks

- [x] Curriculum switch triggers at correct episode count
- [x] Epsilon partially reset at curriculum switch (≥ phase2_epsilon_reset)
- [x] Phase 1 checkpoint saved before phase 2
- [x] Reward shaping is disabled if reward_shaping=False
- [x] Shaped reward is clipped to [-shaping_clip, shaping_clip]
- [x] Terminal rewards (+1/-1) dominate shaped rewards

## Evaluation Fairness Checks

- [x] Epsilon = 0 for all RL agents during evaluation
- [x] No reward shaping applied during evaluation
- [x] P1 and P2 games split: n_games//2 each
- [x] P1 evaluation: agent_starts=True (correct role setup)
- [x] P2 evaluation: agent_starts=False (correct role setup)
- [x] Evaluation opponent is DefaultAgent (not training env)
- [x] No training episodes occur during evaluation

## Search Agent Checks

- [x] Transposition table uses board bytes as hash key
- [x] TT flag: EXACT when α < value < β
- [x] TT flag: LOWER when value ≥ β (failed high)
- [x] TT flag: UPPER when value ≤ α_orig (failed low)
- [x] Move ordering: winning moves first, then blocking, then center-out
- [x] Heuristic function: positive = player advantage, negative = opponent
- [x] Heuristic window: 3-in-a-row scores higher than 2-in-a-row
- [x] Opponent 3-in-a-row threat penalised strongly (-50)
- [x] Iterative deepening from depth 1 to max_depth

## Reduced Board vs Full Board

- [x] Q-learning on 4×5 board: results labeled "board=4x5"
- [x] Q-learning results NOT included in 6×7 crossplay tables
- [x] DQN trained on full 6×7 board
- [x] Evaluation reports clearly separate reduced vs standard board
