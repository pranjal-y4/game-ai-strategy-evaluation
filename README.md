refer this video: https://youtu.be/I6cRKHLr4K0
# Game AI Strategy Evaluation

> Comparative evaluation of **Minimax**, **Alpha-Beta Pruning**, **Tabular Q-Learning**, and **Deep Q-Networks** on **Tic-Tac-Toe** and **Connect 4**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/DQN-Pure%20NumPy-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Not%20Required-lightgrey)
![Schema](https://img.shields.io/badge/CSV%20Schema-v2-purple)
![Mode](https://img.shields.io/badge/Runs-Headless-orange)

---

## Overview

This project compares classical search-based game-playing agents with reinforcement learning agents across two deterministic board games:

| Game | Board | Algorithms |
|---|---:|---|
| Tic-Tac-Toe | 3×3 | Minimax, Alpha-Beta, Tabular Q-Learning, DQN |
| Connect 4 | 6×7 | Depth-Limited Alpha-Beta, DQN |
| Connect 4 Reduced | 4×5 | Tabular Q-Learning |

All training and evaluation runs are **headless**, so no GUI is required.  
The GUI is only used for the optional demo:

```bash
python3 main.py --ui --game ttt
```

---

## Key Features

- Classical adversarial search using **Minimax** and **Alpha-Beta pruning**
- Reinforcement learning using **Tabular Q-Learning**
- Deep reinforcement learning using a **pure NumPy DQN**
- Role-alternated RL training so agents learn as both Player 1 and Player 2
- Cross-play tournaments and evaluation against a deterministic default opponent
- Connect 4 infeasibility demonstration for full-width search
- Multi-seed aggregation with mean, standard deviation, and 95% confidence intervals
- GitHub-friendly plots and timestamped experiment outputs
- Strict `schema_version="v2"` validation for result CSVs

---

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib pandas seaborn
```

No PyTorch is required.  
The DQN implementation is written in pure NumPy in:

```text
rl/dqn.py
```

---

## Quick Start

Run the full experiment pipeline:

```bash
bash run_experiments.sh
```

Run a quicker test pipeline:

```bash
bash run_experiments.sh --quick
```

| Command | Purpose | Approx. Runtime |
|---|---|---:|
| `bash run_experiments.sh` | Full training and evaluation pipeline | 3–4 hours |
| `bash run_experiments.sh --quick` | Lightweight verification run | 5–10 minutes |

---

## Project Structure

```text
.
├── agents/
│   ├── base_agent.py
│   ├── random_agent.py
│   ├── default_agent.py
│   ├── ttt_minimax_agent.py
│   ├── ttt_alphabeta_agent.py
│   ├── c4_depthlimited_alphabeta_agent.py
│   ├── qlearning_ttt_agent.py
│   ├── qlearning_c4_reduced_agent.py
│   ├── dqn_ttt_agent.py
│   └── dqn_c4_agent.py
│
├── games/
│   ├── tictactoe_core.py
│   ├── connect4_core.py
│   ├── tictactoe_ui.py
│   └── connect4_ui.py
│
├── rl/
│   ├── env.py
│   ├── q_learning.py
│   ├── dqn.py
│   ├── train_qlearning_ttt.py
│   ├── train_qlearning_c4_reduced.py
│   ├── train_dqn_ttt.py
│   └── train_dqn_c4.py
│
├── experiments/
│   ├── run_match.py
│   ├── evaluate_against_default.py
│   ├── evaluate_crossplay.py
│   ├── run_tournament.py
│   ├── connect4_infeasibility.py
│   ├── aggregate_results.py
│   ├── plotter.py
│   ├── validate.py
│   ├── results/
│   └── graphs/
│
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   ├── serialization.py
│   └── plotting.py
│
├── models/
├── main.py
├── run_experiments.sh
└── CHANGELOG.md
```

---

## Agents

### Tic-Tac-Toe

| Agent | Description |
|---|---|
| `random` | Random legal move |
| `default` | Heuristic opponent: win, block, smart fallback |
| `minimax` | Full Minimax with depth-sensitive scoring |
| `alphabeta` | Alpha-Beta pruning with the same scoring as Minimax |
| `qlearning_ttt` | Tabular Q-Learning over the full 3×3 state space |
| `dqn_ttt` | DQN with architecture `9 → 128 → 64 → 9` |

### Connect 4

| Agent | Board | Description |
|---|---:|---|
| `random` | 6×7 | Random legal move |
| `default` | 6×7 | Win, block, center-first fallback |
| `c4_alphabeta` | 6×7 | Depth-limited Alpha-Beta with heuristic evaluation |
| `qlearning_c4` | 4×5 | Tabular Q-Learning on reduced Connect 4 |
| `dqn_c4` | 6×7 | DQN trained on full Connect 4 |

---

## Default Opponent

The default opponent is deterministic and follows this priority:

```text
win → block → smart fallback
```

Fallback strategy:

| Game | Fallback |
|---|---|
| Tic-Tac-Toe | Center → corners → edges |
| Connect 4 | Center column → adjacent columns outward |

This makes the default opponent stronger and more stable than a random baseline.

---

## Reinforcement Learning Design

### Role Alternation

All RL training scripts call:

```python
env.enable_role_alternation()
```

This means:

| Episode Type | Agent Role |
|---|---|
| Even episodes | Player 1 |
| Odd episodes | Player 2 |

This prevents the agents from overfitting to one starting role.

The following role-conditioned metrics are logged at evaluation checkpoints:

```text
p1_win_rate
p2_win_rate
first_mover_advantage
```

---

### Opponent Types

RL agents can train against different opponent strengths:

| Opponent | Behaviour |
|---|---|
| `random` | Random legal moves |
| `semi` | Win → block → random |

Example curriculum-style training:

```bash
python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent random --seed 42
python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent semi   --seed 42
```

---

### Generalization Evaluation

At every evaluation checkpoint, RL agents are tested against both:

```text
random opponent
semi-intelligent opponent
```

The training CSVs include:

```text
eval_win_rate_random
eval_win_rate_semi
```

These columns show the generalization gap between easy and harder opponents.

---

## Connect 4 Infeasibility

Full Connect 4 search is computationally infeasible:

```text
3^42 ≈ 3 × 10^20 states
```

The following agents are used only for the infeasibility demonstration:

```text
c4_minimax_infeasible_agent.py
c4_alphabeta_infeasible_agent.py
```

Run the demo:

```bash
python3 experiments/connect4_infeasibility.py --time_budget 60 --seed 42
```

Full assignment run:

```bash
python3 experiments/connect4_infeasibility.py --time_budget 1800 --seed 42
```

For practical Connect 4 play, the project uses:

```text
c4_depthlimited_alphabeta_agent.py
```

This agent uses depth-limited Alpha-Beta search with a heuristic based on:

- Center-column control
- 4-in-a-row windows
- 3-in-a-row windows
- 2-in-a-row windows

---

## Reduced-Board Connect 4 for Q-Learning

Tabular Q-Learning is not practical on the full 6×7 Connect 4 board:

```text
Full board:     3^42 ≈ 3 × 10^20 states
Reduced board:  3^20 ≈ 3.5 × 10^9 theoretical states
```

The reduced 4×5 board is still large, but in practice only about `10^5–10^6` states are visited during training.

Important:

> Reduced-board Connect 4 Q-Learning results are plotted separately and must not be directly compared with full-board 6×7 agents.

CSV rows for this agent are labelled:

```text
board_config=4x5
```

---

## Running Experiments

### 1. Train RL Agents

#### Q-Learning on Tic-Tac-Toe

```bash
python3 rl/train_qlearning_ttt.py --episodes 50000 --seed 42
python3 rl/train_qlearning_ttt.py --episodes 50000 --opponent semi --seed 42
```

#### Q-Learning on Reduced Connect 4

```bash
python3 rl/train_qlearning_c4_reduced.py --episodes 100000 --rows 4 --cols 5 --seed 42
```

#### DQN on Tic-Tac-Toe

```bash
python3 rl/train_dqn_ttt.py --episodes 20000 --seed 42
```

#### DQN on Full Connect 4

```bash
python3 rl/train_dqn_c4.py --episodes 100000 --seed 42
```

Models are saved to:

```text
models/
```

Training metrics are saved to:

```text
experiments/results/
```

---

### 2. Evaluate Against Default Opponent

```bash
python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed 42
python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed 42
```

---

### 3. Run Cross-Play Round-Robin

```bash
python3 experiments/evaluate_crossplay.py --game ttt --games 200 --seed 42
python3 experiments/evaluate_crossplay.py --game c4  --games 100 --seed 42
```

---

### 4. Run Full Tournament

```bash
python3 experiments/run_tournament.py --games 200 --seed 42
python3 experiments/run_tournament.py --game ttt --games 500 --seed 42
```

---

### 5. Run Single Matchups

```bash
python3 experiments/run_match.py --game ttt --agent1 minimax   --agent2 default --games 50  --seed 42
python3 experiments/run_match.py --game ttt --agent1 alphabeta --agent2 random  --games 100 --seed 42
python3 experiments/run_match.py --game c4  --agent1 c4_alphabeta --agent2 default --games 50 --seed 42
```

Valid agent names:

```text
Tic-Tac-Toe: random, default, minimax, alphabeta, qlearning_ttt, dqn_ttt
Connect 4:   random, default, c4_alphabeta, qlearning_c4, dqn_c4
```

---

### 6. Generate Graphs

```bash
python3 experiments/plotter.py
```

This reads the latest schema-compatible v2 CSVs from:

```text
experiments/results/
```

And saves charts to:

```text
experiments/graphs/
```

---

### 7. Validate Correctness

```bash
python3 experiments/validate.py
```

Validation includes 11 checks, including:

- Default agent behaviour
- Minimax and Alpha-Beta agreement
- Legal-move compliance
- Model loading
- Role alternation
- P1/P2 metric consistency
- Board-size isolation
- Plotter schema safety

---

### 8. Aggregate Multi-Seed Results

```bash
python3 experiments/aggregate_results.py
python3 experiments/aggregate_results.py --game ttt
```

The aggregator produces mean, standard deviation, and 95% confidence intervals across compatible v2 CSVs.

---

## Multi-Seed Analysis

Recommended seeds: `42`, `123`, `999`

```bash
for SEED in 42 123 999; do
  python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed $SEED
  python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed $SEED
  python3 experiments/evaluate_crossplay.py       --game ttt --games 200 --seed $SEED
  python3 experiments/evaluate_crossplay.py       --game c4  --games 100 --seed $SEED
done

python3 experiments/aggregate_results.py
```

---

## Output Files

All output files are timestamped using:

```text
YYYYMMDD_HHMMSS
```

### Results

```text
experiments/results/
├── vs_default_ttt_<ts>.csv
├── vs_default_c4_<ts>.csv
├── crossplay_ttt_<ts>.csv
├── crossplay_c4_<ts>.csv
├── c4_infeasibility_<ts>.csv
├── rl_training_metrics_ttt_qlearning_<ts>.csv
├── rl_training_metrics_ttt_dqn_<ts>.csv
├── rl_training_metrics_c4_qlearning_<ts>.csv
├── rl_training_metrics_c4_dqn_<ts>.csv
├── aggregated_vs_default_<ts>.csv
└── aggregated_crossplay_<ts>.csv
```

### Graphs

```text
experiments/graphs/
├── ttt_vs_default_winrate.png
├── ttt_vs_default_outcomes.png
├── ttt_vs_default_role_stratified.png
├── c4_vs_default_winrate.png
├── c4_vs_default_outcomes.png
├── c4_vs_default_role_stratified.png
├── c4_reduced_vs_default_winrate.png
├── c4_reduced_vs_default_outcomes.png
├── ttt_crossplay_heatmap.png
├── c4_crossplay_heatmap.png
├── ttt_fma.png
├── c4_fma.png
├── overall_comparison_dashboard.png
├── speed_vs_quality.png
├── decision_time_comparison.png
├── ttt_qlearning_training_winrate.png
├── ttt_dqn_training_winrate.png
├── ttt_dqn_loss.png
├── c4_qlearning_training_winrate.png
├── c4_dqn_training_winrate.png
├── c4_dqn_loss.png
├── c4_minimax_infeasibility_nodes.png
└── c4_minimax_infeasibility_depth.png
```

---

## CSV Schema

All evaluation CSVs use:

```text
schema_version="v2"
```

Required columns include:

| Column |
|---|
| `schema_version` |
| `game` |
| `board_config` |
| `seed` |
| `agent` |
| `opponent` |
| `win_rate` |
| `draw_rate` |
| `loss_rate` |
| `p1_win_rate` |
| `p2_win_rate` |
| `first_mover_advantage` |
| `avg_agent_time_ms_per_move` |
| `avg_agent_nodes_per_move` |

The plotter and aggregator reject files that do not use:

```text
schema_version="v2"
```

---

## Evaluation Fairness Notes

- In Tic-Tac-Toe, `100%` draws against a competent default opponent indicate optimal or near-optimal play, not poor performance.
- High win rate against a random opponent does not necessarily mean strong performance against default or stronger agents.
- A policy with much higher `p1_win_rate` than `p2_win_rate` is not considered robust.
- Reduced-board Connect 4 results are approximations and are not directly comparable to full 6×7 Connect 4 results.
- The overall dashboards are descriptive comparison tools, not a single definitive ranking of all agents.

---

## Assignment Requirement Mapping

| Requirement | Implementation |
|---|---|
| Plain Minimax on Connect 4 | `c4_minimax_infeasible_agent.py` |
| Alpha-Beta on Connect 4 | `c4_alphabeta_infeasible_agent.py` |
| Practical Connect 4 Alpha-Beta | `c4_depthlimited_alphabeta_agent.py` |
| Tabular Q-Learning on Connect 4 | `qlearning_c4_reduced_agent.py` |
| Full Minimax on Tic-Tac-Toe | `ttt_minimax_agent.py` |
| Alpha-Beta on Tic-Tac-Toe | `ttt_alphabeta_agent.py` |
| Tabular Q-Learning on Tic-Tac-Toe | `qlearning_ttt_agent.py` |
| DQN on Tic-Tac-Toe | `dqn_ttt_agent.py` |
| DQN on Connect 4 | `dqn_c4_agent.py` |

---

## Optional GUI Demo

```bash
python3 main.py --ui --game ttt
python3 main.py --ui --game c4
```

The GUI is for demonstration only.  
Training and evaluation do not require a display.

---

## Notes

This project does not declare one universal winner because each algorithm operates under different computational constraints.

Instead, the evaluation focuses on trade-offs across:

| Metric | Meaning |
|---|---|
| Win rate vs default | Performance against a stable heuristic opponent |
| Cross-play win rate | Performance against other agents |
| First-mover advantage | Robustness across player roles |
| Decision time | Practical efficiency |
| Node count | Search complexity |

The goal is to compare how classical planning and reinforcement learning behave across small, solved games and larger, less tractable game spaces.

