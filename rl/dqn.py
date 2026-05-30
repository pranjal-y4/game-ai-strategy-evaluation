"""
rl/dqn.py
────────────────────────────────────────────────────────────────────────────
Deep Q-Network (DQN) agent – pure NumPy, zero extra dependencies.

Architecture
────────────
Online network  → used to select actions and compute Q(s,a) predictions.
Target network  → periodically copied from online net; provides stable
                  TD targets to prevent oscillation.

Neural network
──────────────
A small MLP: state_size → 128 → 64 → n_actions
Activation: ReLU for hidden layers, linear output.
Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8).
Loss: Mean Squared Error on the sampled mini-batch.

Experience replay
─────────────────
A circular replay buffer stores (s, a, r, s', done, legal') tuples.
Mini-batches are drawn uniformly at random.
Replay decouples correlated online steps and smooths the training signal.

Exploration policy
──────────────────
ε-greedy with linear decay from ε_start → ε_min over decay_steps.
Illegal actions are masked by setting their Q-values to −∞ before argmax.

Connect 4 note
──────────────
For the full 6×7 board the state has 42 features.  DQN handles this
naturally through function approximation – no board reduction needed.
"""

from __future__ import annotations

import numpy as np
import random
import pickle
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Numpy MLP with Adam optimiser
# ─────────────────────────────────────────────────────────────────────────────

class _MLP:
    """Small fully-connected network: state → Q-values."""

    def __init__(self, layer_sizes: list[int], lr: float = 1e-3):
        self.lr          = lr
        self.n_layers    = len(layer_sizes) - 1

        # He initialisation (good for ReLU)
        self.W = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            * np.sqrt(2.0 / layer_sizes[i])
            for i in range(self.n_layers)
        ]
        self.b = [np.zeros(layer_sizes[i + 1]) for i in range(self.n_layers)]

        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0   # Adam time-step

        self._cache_a: list  = []   # activations (pre-ReLU for hidden, raw out)
        self._cache_h: list  = []   # post-activation

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in_size) or (in_size,) → (batch, out_size)"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        self._cache_a = []
        self._cache_h = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = h @ W + b
            self._cache_a.append(a)
            h  = np.maximum(0, a) if i < self.n_layers - 1 else a  # ReLU / linear
            self._cache_h.append(h)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward, no caching side-effects needed at inference."""
        return self.forward(x)

    # ── backward (MSE loss assumed) ───────────────────────────────────────

    def backward_and_update(self, loss_grad: np.ndarray) -> float:
        """
        loss_grad: d_Loss / d_output  shape (batch, out_size).
        Returns the mean squared gradient norm (for logging).
        """
        batch = loss_grad.shape[0]
        delta = loss_grad  # gradient at the output layer

        grads_W = []
        grads_b = []

        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                # Backprop through ReLU
                delta = delta * (self._cache_a[i] > 0).astype(np.float32)
            dW = self._cache_h[i].T @ delta / batch
            db = delta.mean(axis=0)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            delta = delta @ self.W[i].T

        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(self.n_layers):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * grads_W[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * grads_W[i] ** 2
            mW_hat = self.mW[i] / (1 - b1 ** self.t)
            vW_hat = self.vW[i] / (1 - b2 ** self.t)
            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * grads_b[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * grads_b[i] ** 2
            mb_hat = self.mb[i] / (1 - b1 ** self.t)
            vb_hat = self.vb[i] / (1 - b2 ** self.t)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return float(np.mean(loss_grad ** 2))

    # ── weight copy ───────────────────────────────────────────────────────

    def get_weights(self) -> list[tuple]:
        return [(w.copy(), b.copy()) for w, b in zip(self.W, self.b)]

    def set_weights(self, weights: list[tuple]) -> None:
        for i, (w, b) in enumerate(weights):
            self.W[i] = w.copy()
            self.b[i] = b.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal):
        self.buffer.append((state, action, reward, next_state, done, next_legal))

    def sample(self, n: int):
        batch = random.sample(self.buffer, n)
        s, a, r, ns, d, nl = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
            nl,   # list of legal-action lists (variable length)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
#  DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN Agent with experience replay and a target network.

    Parameters
    ----------
    env             : game environment (TicTacToeEnv or Connect4Env)
    hidden          : list of hidden-layer sizes, e.g. [128, 64]
    lr              : Adam learning rate (default 1e-3)
    gamma           : discount factor γ (default 0.95)
    epsilon_start   : initial exploration rate (default 1.0)
    epsilon_min     : floor (default 0.05)
    decay_steps     : linear decay over this many steps (default 15 000)
    batch_size      : replay mini-batch size (default 64)
    buffer_size     : replay buffer capacity (default 20 000)
    target_update   : copy online → target every N steps (default 200)
    """

    def __init__(
        self,
        env,
        hidden:         list  = None,
        lr:             float = 1e-3,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_min:    float = 0.05,
        decay_steps:    int   = 15_000,
        batch_size:     int   = 64,
        buffer_size:    int   = 20_000,
        target_update:  int   = 200,
    ):
        self.env          = env
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_min  = epsilon_min
        self.epsilon_step = (epsilon_start - epsilon_min) / max(decay_steps, 1)
        self.batch_size   = batch_size
        self.target_update = target_update

        hidden = hidden or [128, 64]
        layers = [env.STATE_SIZE] + hidden + [env.N_ACTIONS]

        self.online = _MLP(layers, lr=lr)
        self.target = _MLP(layers, lr=lr)
        self.target.set_weights(self.online.get_weights())

        self.buffer    = _ReplayBuffer(buffer_size)
        self.step_count = 0

        # History
        self.episode_rewards: list[float] = []
        self.losses:          list[float] = []
        self.win_rates:       list[float] = []

    # ══════════════════════════════════════════════════════════════════════
    #  Action selection
    # ══════════════════════════════════════════════════════════════════════

    def choose_action(self, state: np.ndarray, legal_actions: list[int]) -> int:
        """
        ε-greedy with action masking.

        Exploration : random choice among legal_actions.
        Exploitation: forward-pass, mask illegal Q-values to −∞, argmax.
        """
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return self._greedy(state, legal_actions)

    def _greedy(self, state: np.ndarray, legal_actions: list[int]) -> int:
        q_vals = self.online.forward(state).flatten()
        mask   = np.full(self.env.N_ACTIONS, -np.inf)
        mask[legal_actions] = q_vals[legal_actions]
        return int(np.argmax(mask))

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

    # ══════════════════════════════════════════════════════════════════════
    #  Training step
    # ══════════════════════════════════════════════════════════════════════

    def _train_step(self) -> Optional[float]:
        """Sample a mini-batch, compute TD targets, backprop. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, done, next_legal = self.buffer.sample(self.batch_size)

        # Predicted Q-values for all actions
        q_pred = self.online.forward(s)   # (batch, n_actions)

        # Target Q-values from target network (action-masked)
        q_next = self.target.forward(ns)  # (batch, n_actions)

        targets = q_pred.copy()

        for i in range(self.batch_size):
            if done[i]:
                td_target = r[i]
            else:
                nl = next_legal[i]
                if nl:
                    masked = np.full(self.env.N_ACTIONS, -np.inf)
                    masked[nl] = q_next[i][nl]
                    td_target = r[i] + self.gamma * np.max(masked[nl])
                else:
                    td_target = r[i]
            targets[i, a[i]] = td_target

        # MSE loss gradient: 2*(pred - target) / batch
        loss_grad = 2.0 * (q_pred - targets) / self.batch_size
        loss = self.online.backward_and_update(loss_grad)
        return loss

    # ══════════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════════

    def train(
        self,
        n_episodes:    int = 5_000,
        eval_every:    int = 250,
        eval_episodes: int = 100,
        callback=None,
        stop_flag=None,
    ) -> dict:
        """
        Training Loop
        ─────────────
        Each episode: agent interacts with env, stores transitions, trains.
        Target network syncs every target_update steps.

        Returns
        -------
        dict with episode_rewards, losses, win_rates
        """
        for ep in range(1, n_episodes + 1):
            if stop_flag and stop_flag.is_set():
                break

            state = self.env.reset()
            total_r = 0.0

            while True:
                legal  = self.env.get_legal_actions()
                action = self.choose_action(state, legal)

                next_state, reward, done, _ = self.env.step(action)
                next_legal = self.env.get_legal_actions()

                self.buffer.push(state, action, reward, next_state, done, next_legal)
                loss = self._train_step()
                if loss is not None:
                    self.losses.append(loss)

                self.step_count += 1
                self._decay_epsilon()

                if self.step_count % self.target_update == 0:
                    self.target.set_weights(self.online.get_weights())

                total_r += reward
                state    = next_state

                if done:
                    break

            self.episode_rewards.append(total_r)

            if ep % eval_every == 0:
                wr = self.evaluate(eval_episodes)
                self.win_rates.append(wr)
                if callback:
                    callback(ep, wr, self.epsilon)

        return {
            "episode_rewards": self.episode_rewards,
            "losses":          self.losses,
            "win_rates":       self.win_rates,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  Evaluation mode
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self, n_episodes: int = 100) -> float:
        """
        Evaluation Mode
        ───────────────
        Greedy play (ε = 0, no exploration) for n_episodes.
        Returns fraction of episodes where agent won (reward > 0).
        """
        saved = self.epsilon
        self.epsilon = 0.0
        wins = 0

        for _ in range(n_episodes):
            state = self.env.reset()
            while True:
                legal  = self.env.get_legal_actions()
                action = self._greedy(state, legal)
                state, reward, done, _ = self.env.step(action)
                if done:
                    if reward >= 1.0:  # win=+1.0; draw=+0.2 must NOT count as win
                        wins += 1
                    break

        self.epsilon = saved
        return wins / n_episodes

    # ══════════════════════════════════════════════════════════════════════
    #  Inference  (used by DQNOpponent in opponents.py)
    # ══════════════════════════════════════════════════════════════════════

    def predict(self, state: np.ndarray, legal_actions: list[int]) -> int:
        """Greedy action at play-time (no exploration)."""
        return self._greedy(state, legal_actions)

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self, path: str) -> None:
        data = {
            "online_weights": self.online.get_weights(),
            "target_weights": self.target.get_weights(),
            "epsilon":        self.epsilon,
            "step_count":     self.step_count,
            "episode_rewards": self.episode_rewards,
            "losses":         self.losses,
            "win_rates":      self.win_rates,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.online.set_weights(data["online_weights"])
        self.target.set_weights(data["target_weights"])
        self.epsilon          = data.get("epsilon", self.epsilon_min)
        self.step_count       = data.get("step_count", 0)
        self.episode_rewards  = data.get("episode_rewards", [])
        self.losses           = data.get("losses", [])
        self.win_rates        = data.get("win_rates", [])

    def info(self) -> dict:
        return {
            "agent":            "DQN",
            "network_layers":   [self.env.STATE_SIZE] + [128, 64] + [self.env.N_ACTIONS],
            "epsilon":          self.epsilon,
            "buffer_size":      len(self.buffer),
            "step_count":       self.step_count,
            "episodes_trained": len(self.episode_rewards),
            "avg_loss_recent":  float(np.mean(self.losses[-100:])) if self.losses else 0.0,
        }
