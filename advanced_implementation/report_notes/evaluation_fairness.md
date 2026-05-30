# Evaluation Fairness

## Key Fairness Principles

### 1. Greedy Evaluation (ε = 0)
All RL agents (Q-learning, DQN) are evaluated with `epsilon = 0.0`.
This ensures the evaluation measures the learned policy, not random exploration.
Epsilon is restored after evaluation to avoid side effects on training.

**Implementation:** `_set_greedy(agent)` in `evaluation/evaluator.py`

### 2. No Reward Shaping During Evaluation
The evaluation loop uses `game.apply_move()` directly on game objects —
never through `GameEnv`. This means the shaped reward function in `GameEnv`
is never invoked during evaluation. Only true game outcomes (+1/-1/0) matter.

### 3. Correct P1/P2 Split
Each agent is evaluated for exactly `n_games//2` games as Player 1
and `n_games//2` games as Player 2.

**P1 games:** `game.reset()` → current_player = 1 → agent acts as P1  
**P2 games:** `game.reset()` → current_player = 1 (opponent first) → agent acts as P2

This is done by passing `agent_player=1` or `agent_player=2` to `_play_game()`,
which correctly routes game.current_player to the right agent at each turn.

**This is NOT a manual hack** — it correctly reflects the two roles a real
agent must play in competition.

### 4. No Training Leakage
Evaluation environments are created fresh each time. They share no state
with the training `GameEnv`. The Q-table/DQN weights are not updated
during evaluation.

### 5. Consistent Opponent
All agents are evaluated against the same `DefaultAgent` instance.
The default agent uses a deterministic heuristic (win > block > center-out),
so results are comparable across agents.

### 6. Deterministic Evaluation
Random seed is set before each evaluation run. For search agents (Alpha-Beta),
move selection is deterministic given the same board. For RL agents with ε=0,
`argmax` over Q-values is also deterministic.

## P1/P2 Reporting

For each agent, the following metrics are reported:
- `p1_win_rate`, `p1_draw_rate`, `p1_loss_rate`
- `p2_win_rate`, `p2_draw_rate`, `p2_loss_rate`
- `total_win_rate` = (p1_wins + p2_wins) / total_games
- `first_mover_advantage` = p1_win_rate - p2_win_rate

First mover advantage is expected to be positive for most strong agents
in Connect4 (P1 has theoretical advantage) but should be small for agents
that play well in both roles.

## What Evaluation Does NOT Measure

1. **Training curve quality** — only final greedy performance matters
2. **Generalisation** — evaluation vs `DefaultAgent` may not reflect
   performance vs truly optimal play
3. **Robustness** — a single opponent does not reveal all weaknesses

See `limitations.md` for a full discussion.
