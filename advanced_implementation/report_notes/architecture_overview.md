# Architecture Overview

## Folder Structure

```
advanced_implementation/
├── games/             Standalone game environments (no old code dependency)
├── agents/            Search agents (Minimax, Alpha-Beta, Advanced Alpha-Beta)
├── rl/                RL components (GameEnv, Q-learning, DQN, buffers)
├── training/          Training scripts with curriculum and reward shaping
├── evaluation/        Fair evaluation system (P1/P2 split, greedy, no shaping)
├── utils/             Seed, metrics, logger, plotter
├── configs/           Default hyperparameters
├── tests/             Unit tests and validation checklist
├── models/            Saved model weights (.pkl / .pt)
├── logs/              Training CSVs and evaluation CSVs
└── report_notes/      This documentation
```

## Key Design Decisions

### Independence from Original Project
All game environments are reimplemented from scratch. No imports from
the original `games/`, `agents/`, or `rl/` folders. This ensures the
advanced implementation can be understood, modified, and graded independently.

### Game Environments (games/)
- `TicTacToe`: 3×3 board, numpy int8, correct win/draw/terminal detection
- `Connect4`: configurable rows×cols, gravity-based placement, correct 4-in-a-row detection
- Both support: `reset()`, `clone()`, `legal_moves()`, `apply_move()`,
  `is_terminal()`, `winner()`, `encode_state()`, `render_text()`
- State encoding always from agent's perspective (+1=own, 0=empty, -1=opp)

### Search Agents (agents/)
- `MinimaxAgent`: full minimax baseline (no pruning) — correctness reference
- `AlphaBetaAgent`: standard alpha-beta with center-out / TTT preference ordering
- `AdvancedAlphaBetaC4Agent`: full advanced agent for Connect4 (see search_improvements.md)

### RL Agents (rl/)
- `AdvancedQLearning`: tabular Q-learning with n-step returns
- `AdvancedDQNAgent`: Double DQN + PER + n-step (PyTorch neural network)
- `GameEnv`: single-agent wrapper with role alternation and opponent modes
- `PrioritizedReplayBuffer`: sum-tree PER with IS weights
- `NStepBuffer`: n-step return accumulator

### Training (training/)
- Curriculum: phase 1 (random) → phase 2 (default) with partial epsilon reset
- Reward shaping: heuristic difference, clipped to [-0.1, 0.1], training only
- Role alternation: even episodes = agent is P1, odd = agent is P2
- Full logging: episode, epsilon, win rates, P1/P2 split, phase markers

### Evaluation (evaluation/)
- `evaluator.py`: core evaluation with correct P1/P2 split
- `run_evaluation.py`: full pipeline runner
- Fairness: greedy agents, no shaping, correct role setup

## Data Flow: Connect4 DQN Training

```
GameEnv.reset() → state
  ↓
Agent.choose_action(state, legal) → action (ε-greedy or greedy)
  ↓
GameEnv.step(action) → (next_state, raw_reward, done, info)
  ↓ [+ shaped_reward = clip(weight*(h(s')−h(s)), −0.1, 0.1)]
  ↓
NStepBuffer.push(s, a, r_shaped, s', done, legal') → ready transitions
  ↓
PrioritizedReplayBuffer.push(s_t, a_t, R_n, s_{t+n}, done_n, legal_n)
  ↓
_train_step():
  online(s) → q_pred (all actions)
  online(s') → greedy next action a*          ← Double DQN: online selects
  target(s') → Q_target(s', a*)               ← Double DQN: target evaluates
  y = R_n + γⁿ * Q_target(s', a*) * (1-done)
  loss = mean(IS_weights * (y - q_pred[a])²)
  Backprop + gradient clip + Adam step
  Update priorities: |y - q_pred[a]| + ε
```
