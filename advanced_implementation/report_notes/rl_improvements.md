# RL Improvements

## Tabular Q-Learning Improvements

### N-Step Returns (n=3)
**Original:** Standard 1-step TD update
```
Q(s_t, a_t) ← Q(s_t, a_t) + α [r_t + γ max Q(s_{t+1}) − Q(s_t, a_t)]
```

**Advanced:** N-step return (n=3)
```
R_n = r_t + γ*r_{t+1} + γ²*r_{t+2}
Q(s_t, a_t) ← Q(s_t, a_t) + α [R_n + γ³ max Q(s_{t+n}) − Q(s_t, a_t)]
```

**Why it helps:** In Connect4, rewards are sparse — only the terminal
step provides a non-zero reward. With 1-step TD, each update bootstraps
from a state with Q-value ≈ 0 for most of the game. N-step returns
propagate reward information faster, reducing the number of episodes
needed for convergence.

**For the 4×5 reduced board:** Average episode length ≈ 14–20 steps.
3-step returns bridge roughly 20% of the episode length, significantly
accelerating credit assignment.

## Advanced DQN Improvements

### 1. Double DQN (Prevents Maximization Bias)
**Original:** Standard DQN TD target
```
y = r + γ max_a Q_target(s', a)
```

**Advanced:** Double DQN
```
a* = argmax_a Q_online(s', a)   ← online network SELECTS
y = r + γ Q_target(s', a*)       ← target network EVALUATES
```

**Why it helps:** In standard DQN, `max_a Q_target(s', a)` overestimates
the true value when the Q-function has any noise (which it always does early
in training). Double DQN decorrelates action selection and evaluation,
reducing this systematic overestimation. This leads to more stable training
and less oscillation in Q-values.

**Implementation:** `rl/dqn.py lines 180–200`

### 2. Prioritized Experience Replay (PER)
**Original:** Uniform random sampling from circular buffer.

**Advanced:** Priority = |TD error| + ε, sampled proportional to priority.

**Sum-tree structure:** O(log N) sampling and O(log N) priority update.
Much more efficient than sorting the full buffer.

**Importance-sampling (IS) weights:** Corrects the sampling bias introduced
by non-uniform sampling:
```
w_i = (1/(N * P(i)))^β / max_w
```
β starts at 0.4 and anneals to 1.0 over training (full correction at end).

**Why it helps:** Rare but important transitions (e.g., the exact move that
lost the game) may never be replayed with uniform sampling. PER ensures
high-error transitions are replayed more frequently, accelerating learning.
The IS correction prevents these high-priority samples from dominating
gradient updates excessively.

### 3. N-Step Returns in DQN
Uses `NStepBuffer` to accumulate 3-step returns before pushing to PER:
```
R_3 = r_t + γ*r_{t+1} + γ²*r_{t+2}
TD target: R_3 + γ³ * Q_target(s_{t+3}, a*)
```

Combined with Double DQN and PER, this gives three complementary improvements:
- n-step: faster propagation of reward
- Double DQN: less bias in bootstrap target
- PER: more efficient use of experience buffer

### 4. Cosine Epsilon Decay
**Original:** Linear decay `ε ← ε - step`

**Advanced:** Cosine decay
```
ε = ε_min + 0.5*(ε_start - ε_min)*(1 + cos(π * t/T))
```

**Why it helps:** Cosine decay is smooth and naturally slows down decay
near the end of training (where fine-tuning is more important than exploration).
Linear decay can drop ε too quickly early on or too slowly near the end.

### 5. Curriculum Training
**Phase 1:** Train vs random opponent (60% of episodes)
- Agent learns basic gameplay, connect-4 patterns
- High exploration (ε starts at 1.0)

**Phase 2:** Fine-tune vs default opponent (40% of episodes)
- Agent specifically learns to beat the opponent it will be evaluated against
- Epsilon partially reset to max(current, 0.3) to encourage re-exploration

**Why it helps:** The default opponent uses non-trivial heuristics (win>block).
Phase 1 builds foundational game knowledge; Phase 2 specialises against the
evaluation target. This two-stage approach is more sample-efficient than
training directly against the default opponent from the start (which can be
slow to learn from a strong but non-optimal teacher).

### 6. Reward Shaping
During training only, an additional shaped reward is added:
```
shaped = clip(weight * (h(s') - h(s)), -clip, clip)
```
where `h(s)` is the board heuristic from the agent's perspective.

Terminal rewards (+1/-1) dominate (shaped reward clips to ±0.1).
Shaping is completely disabled during evaluation (greedy play uses
the actual game rewards only).

**Why it helps:** Sparse reward environments are difficult for RL because
most transitions have reward=0. Shaping provides a dense signal proportional
to the quality of each move, guiding early learning without distorting
the final reward structure.

### 7. Legal Action Masking
During both training and evaluation, illegal actions are assigned Q-value = −∞
before argmax. This ensures the agent never selects a full column in Connect4.

**Implementation:** `rl/dqn.py _greedy()` and `_train_step()` both apply masking.

### 8. Gradient Clipping
Gradients are clipped to max norm 10.0 before the Adam step. This prevents
rare large TD errors (especially early in training) from causing catastrophic
weight updates.
