# Advanced Implementation — TicTacToe & Connect4 AI

A standalone, high-performance AI implementation for TicTacToe and Connect4.
**All code lives inside `advanced_implementation/`. The original project is untouched.**

---

## Quick Start

```bash
# From the project root:
cd /path/to/Pranjal_AI_Assignment3

# 1. Run all validation tests
python advanced_implementation/tests/test_games.py
python advanced_implementation/tests/test_agents.py

# 2. Train TicTacToe agents
python advanced_implementation/training/train_ttt_qlearning.py
python advanced_implementation/training/train_ttt_dqn.py

# 3. Train Connect4 agents
python advanced_implementation/training/train_c4_qlearning_reduced.py    # 4×5 reduced board
python advanced_implementation/training/train_c4_dqn.py --reward_shaping # 6×7 full board

# 4. Evaluate all agents
python advanced_implementation/evaluation/run_evaluation.py --game both --n_games 200

# 5. Generate plots from training logs
python advanced_implementation/utils/plotting.py  # or call plot_training_curves() directly
```

---

## Structure

```
advanced_implementation/
├── games/
│   ├── tictactoe.py          TicTacToe environment (standalone)
│   └── connect4.py           Connect4 environment (configurable board)
│
├── agents/
│   ├── base_agent.py         Abstract base class
│   ├── random_agent.py       Uniform random baseline
│   ├── default_agent.py      Win > Block > Center heuristic
│   ├── minimax_agent.py      Full Minimax (correctness baseline)
│   ├── alphabeta_agent.py    Alpha-Beta pruning
│   └── advanced_alphabeta_c4.py  ★ Advanced: TT + ID + win-first ordering
│
├── rl/
│   ├── game_env.py           Gym-style RL wrapper with role alternation
│   ├── q_learning.py         ★ Tabular Q-learning with n-step returns
│   ├── dqn.py                ★ Double DQN + PER + n-step (PyTorch)
│   ├── replay_buffer.py      Uniform + Prioritized (PER) buffers
│   └── n_step_buffer.py      N-step return accumulator
│
├── training/
│   ├── train_ttt_qlearning.py    TTT Q-learning with curriculum
│   ├── train_c4_qlearning_reduced.py  C4 4×5 Q-learning with curriculum
│   ├── train_ttt_dqn.py           TTT DQN with curriculum
│   └── train_c4_dqn.py            ★ C4 DQN: Phase1(random)→Phase2(default)
│
├── evaluation/
│   ├── evaluator.py          Core evaluation: P1/P2 split, greedy, fair
│   └── run_evaluation.py     Full pipeline runner with CSV + plots
│
├── utils/
│   ├── seed.py               Reproducible seeding (random, numpy, torch)
│   ├── metrics.py            Win/draw/loss tracker
│   ├── logger.py             CSV training logger
│   └── plotting.py           Training curves + crossplay heatmaps
│
├── configs/
│   └── default_config.py     Default hyperparameters for all experiments
│
├── tests/
│   ├── test_games.py         Game environment unit tests
│   ├── test_agents.py        Agent + RL component tests
│   └── validation_checklist.md  Full validation checklist
│
├── models/                   Saved checkpoints (auto-created by training)
├── logs/                     Training CSVs and evaluation CSVs
│
└── report_notes/
    ├── architecture_overview.md
    ├── search_improvements.md
    ├── rl_improvements.md
    ├── evaluation_fairness.md
    ├── limitations.md
    └── comparison_with_original.md
```

---

## Key Improvements Over Original

| Feature | Original | Advanced |
|---------|----------|---------|
| C4 Alpha-Beta depth | 5 (fixed) | 8 (iterative deepening) |
| Move ordering | center-out | win-first + block + center-out |
| Transposition table | No | Yes (cleared per root call) |
| DQN architecture | NumPy MLP | PyTorch (Double DQN) |
| Replay sampling | Uniform | Prioritized (PER + IS weights) |
| N-step returns | 1-step | 3-step |
| Epsilon decay | Linear | Cosine |
| Curriculum training | Optional flag | Built-in (phase1→phase2) |
| Reward shaping | No | Heuristic difference (training only) |
| Gradient clipping | No | Yes (max_norm=10) |
| P1/P2 evaluation | Yes | Yes (cleaner, no hacks) |

---

## Training Details

### TicTacToe Q-Learning
```bash
python advanced_implementation/training/train_ttt_qlearning.py \
    --episodes 60000 --lr 0.15 --n_step 3 --curriculum_frac 0.6
```
- Phase 1 (60%): vs random opponent
- Phase 2 (40%): vs default opponent (epsilon reset to max(current, 0.2))
- Expected: ~90%+ win rate vs default after full training

### Connect4 Q-Learning (4×5 Reduced Board)
```bash
python advanced_implementation/training/train_c4_qlearning_reduced.py \
    --episodes 150000 --rows 4 --cols 5 --n_step 3
```
**Note:** Results are for 4×5 board ONLY. Not comparable to 6×7 agents.

### Connect4 DQN (6×7 Full Board)
```bash
python advanced_implementation/training/train_c4_dqn.py \
    --episodes 100000 --reward_shaping --curriculum_frac 0.6
```
- Phase 1 (60k eps): vs random opponent — learns basic game patterns
- Phase 2 (40k eps): vs default opponent — fine-tunes for evaluation target
- Checkpoint saved at phase 1 end: `models/c4_dqn_6x7_phase1.pt`
- Best model saved: `models/c4_dqn_6x7_best.pt`

**Honest note:** Phase 2 fine-tuning improves vs-default performance but
does not guarantee general optimality. See `report_notes/limitations.md`.

---

## Evaluation

```bash
python advanced_implementation/evaluation/run_evaluation.py \
    --game c4 --n_games 200 --ab_depth 8

# With a specific model:
python advanced_implementation/evaluation/run_evaluation.py \
    --game c4 --c4_dqn_model advanced_implementation/models/c4_dqn_6x7_best.pt
```

Outputs:
- CSV: `logs/vs_default_c4_<timestamp>.csv`
- CSV: `logs/crossplay_c4_<timestamp>.csv`
- Plot: `logs/plots/vs_default_c4_outcomes.png`

---

## Dependencies

```
numpy>=1.20.0
matplotlib>=3.5.0
torch>=1.12.0
pandas>=1.4.0  (optional, for plotting from CSV)
```

PyTorch is required only for the DQN agent. All other components use NumPy.
