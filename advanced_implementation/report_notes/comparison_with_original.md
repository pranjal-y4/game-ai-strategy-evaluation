# Comparison with Original Project

## OLD → NEW Mapping Table

| Component | Status | Justification |
|-----------|--------|---------------|
| `games/tictactoe_core.py` | **Reimplemented** | Clean standalone; no UI dependency |
| `games/connect4_core.py` | **Reimplemented** | Added `_check_win()` using direction vectors; cleaner than nested loops |
| `agents/base_agent.py` | **Preserved logically** | Same ABC interface |
| `agents/random_agent.py` | **Preserved logically** | Correct and minimal |
| `agents/default_agent.py` | **Preserved logically** | Correct heuristic; same win>block>fallback |
| `agents/ttt_minimax_agent.py` | **Preserved logically** | Moved to `minimax_agent.py`; added max_depth param |
| `agents/ttt_alphabeta_agent.py` | **Preserved logically** | Moved to `alphabeta_agent.py` |
| `agents/c4_depthlimited_alphabeta_agent.py` | **Improved** → `advanced_alphabeta_c4.py` | Added TT, iterative deepening, winning-move-first ordering, balanced heuristic |
| `agents/c4_minimax_infeasible_agent.py` | **Dropped** | Demo-only agent not needed in advanced implementation |
| `agents/c4_alphabeta_infeasible_agent.py` | **Dropped** | Same reason |
| `rl/env.py` | **Replaced** → `rl/game_env.py` | Cleaner API; exposes heuristic for reward shaping |
| `rl/q_learning.py` | **Improved** → `rl/q_learning.py` | Added n-step returns (n=3); preserved core algorithm |
| `rl/dqn.py` | **Replaced** → `rl/dqn.py` | NumPy MLP replaced with PyTorch; added Double DQN, PER, n-step returns, cosine ε decay, gradient clipping |
| `rl/train_*.py` | **Improved** → `training/train_*.py` | Added curriculum, reward shaping, phase markers, checkpoint saving |
| `experiments/evaluate_*.py` | **Replaced** → `evaluation/run_evaluation.py` | Cleaner P1/P2 split; no training leakage; proper greedy evaluation |
| `experiments/validate.py` | **Replaced** → `tests/test_*.py` | Proper unit tests with assertions; covers more cases |
| `utils/seed.py` | **Preserved logically** | Same implementation |
| `utils/metrics.py` | **Improved** | Cleaner API with win/draw/loss tracking |
| `utils/serialization.py` | **Dropped** | Replaced by CSV DictWriter in training scripts |
| `utils/plotting.py` | **Improved** | Added training curve + heatmap plots |

## Quantitative Improvement Targets

| Metric | Original | Advanced |
|--------|----------|---------|
| C4 Search depth | 5 (fixed) | 8 (iterative deepening) |
| C4 Move ordering | center-out only | win-first + block + center-out |
| C4 Search TT | None | Yes (per-root-call) |
| DQN target computation | Standard DQN | Double DQN |
| DQN replay sampling | Uniform | PER (priority + IS weights) |
| DQN n-step | 1-step | 3-step |
| DQN ε decay | Linear | Cosine |
| Q-learning n-step | 1-step | 3-step |
| Curriculum training | Optional (--curriculum flag) | Built-in (always used) |
| Reward shaping | None | Heuristic difference (training only) |
| Gradient clipping | None (NumPy backprop) | nn.clip_grad_norm_(max=10) |
| P1/P2 split evaluation | Yes | Yes (cleaner implementation) |

## Key Preserved Logic (Conceptually Correct in Original)

1. **State perspective encoding** (+1=own, -1=opp) — correct and preserved
2. **Legal action masking** in both Q-learning and DQN — correct and preserved
3. **Role alternation** (even episodes = P1, odd = P2) — correct and preserved
4. **Draw reward = 0.0** (changed from original 0.2 draw reward to keep terminal rewards clean)
5. **Epsilon restored after evaluation** — correct and preserved
6. **Separate eval environments** from training environments — correct and preserved
7. **4×5 reduced board for tabular Connect4** — correct trade-off and preserved
8. **Win > Block > Fallback** in DefaultAgent — correct heuristic and preserved

## Why PyTorch for DQN?

The original implementation used a pure NumPy MLP (`_MLP` class with manual
Adam). This works but has several disadvantages:
- Manual backpropagation is error-prone
- Cannot leverage GPU acceleration
- Implementing Double DQN requires careful gradient management
- PER with importance-sampling weights requires per-sample loss weighting,
  which is cleaner with PyTorch's `loss.mean()` vs `(weights * loss).mean()`

PyTorch allows cleaner, more correct implementations of:
- Gradient clipping (`nn.utils.clip_grad_norm_`)
- Target network sync (`load_state_dict`)
- Per-sample IS weight correction

PyTorch is already a dependency in the original project (`requirements.txt`).
