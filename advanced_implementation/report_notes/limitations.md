# Limitations and Honest Assessment

## Tabular Q-Learning Limitations

### Full 6×7 Connect4 is Infeasible for Tabular RL
The theoretical state space of 6×7 Connect4 is ~4.5×10¹² positions.
Even the practically reachable states (~3×10⁷) far exceed what tabular
Q-learning can explore in a reasonable number of training episodes.

**What we do instead:** Train on a 4×5 reduced board (~10⁵–10⁶ visited states).
Results are clearly labeled `board=4x5` and never compared to 6×7 agents.
This is an explicit limitation of tabular RL for large games.

### N-Step Returns Improve Convergence, Not Optimality
N-step returns (n=3) accelerate credit assignment and reduce the number of
episodes needed for convergence. However, they do not change the limit to
which Q-learning converges given infinite training. The benefit is practical
speed of learning, not ultimate quality.

## DQN Limitations

### Opponent-Specific Fine-Tuning vs General Mastery
Phase 2 curriculum training fine-tunes the agent specifically against the
default opponent. This can yield strong performance in the evaluation
benchmark but does NOT imply the agent has learned optimal Connect4 strategy.

An agent fine-tuned against a weak opponent may lose to a stronger opponent
(e.g., Alpha-Beta depth 8) despite achieving high vs-default win rates.

**Honest claim:** "The agent is designed to achieve strong performance
against the default opponent, especially after curriculum phase 2."

**Not claimed:** "The agent plays Connect4 optimally."

### Function Approximation and Stability
DQN with function approximation can:
- Oscillate during training (addressed by target network + gradient clipping)
- Overestimate Q-values (addressed by Double DQN)
- Be sensitive to hyperparameters

The implemented improvements significantly stabilise training but do not
eliminate these fundamental challenges of off-policy Q-learning with
neural function approximation.

### Sample Efficiency
Even with PER and n-step returns, 100,000 episodes may not be sufficient
for a DQN to master full 6×7 Connect4 against a sophisticated opponent.
AlphaGo-style systems use millions of episodes. This implementation aims
for strong practical performance, not theoretical convergence guarantees.

## Search Agent Limitations

### Connect4 is Not Solved by Alpha-Beta at Depth 8
True optimal Connect4 play requires solving the game from the start
(proven in 1988: first player wins with perfect play). Alpha-Beta at
depth 8 approaches optimal play but is not guaranteed to be optimal
in all positions.

Deeper search (depth 12-15) with a good heuristic would be much stronger,
but computation time scales exponentially with depth.

### Transposition Table Memory
The TT can grow large for deep searches. The current implementation clears
it per root call, preventing unbounded memory growth but also losing
information across moves. A bounded TT with replacement policy would be
more memory-efficient for longer games.

## General Limitations

### Evaluation Opponent Mismatch
All evaluation results are vs the `DefaultAgent`. If the grading criterion
changes (e.g., vs a stronger or different opponent), results may change.

### Reproducibility vs Random Opponents
Even with fixed seeds, the default agent uses tie-breaking heuristics
that are deterministic but the random component in the random agent means
results can vary slightly across runs if seeds are not set correctly.

### Training Time
Full DQN training (100,000 episodes) takes ~30-60 minutes on CPU.
On GPU, this is faster. The implementation uses PyTorch which can leverage
GPU acceleration automatically.
