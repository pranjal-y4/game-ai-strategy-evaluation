"""
rl/q_learning.py
────────────────────────────────────────────────────────────────────────────
Tabular Q-learning agent.

Algorithm
─────────
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]

Unless the transition is terminal, in which case:
Q(s, a) ← Q(s, a) + α · [ r − Q(s, a) ]

State encoding
──────────────
States are encoded as hashable tuples (via env.encode_state()) and stored
in a Python defaultdict, giving a sparse table that only allocates entries
for states actually visited during training.

Exploration policy
──────────────────
ε-greedy:  with probability ε pick a random legal action,
           otherwise pick argmax_a Q(s, a) over legal actions only.
ε decays exponentially after each episode until ε_min is reached.

Action masking
──────────────
Illegal actions are never selected: argmax is computed only over the set
of legal actions returned by env.get_legal_actions().
"""

from __future__ import annotations

import random
import pickle
from collections import defaultdict
from typing import Optional


class TabularQLearning:
    """
    Tabular Q-learning agent compatible with TicTacToeEnv and Connect4Env.

    Parameters
    ----------
    env             : game environment (TicTacToeEnv or Connect4Env)
    lr              : learning rate α   (default 0.1)
    gamma           : discount factor γ (default 0.95)
    epsilon         : initial exploration rate (default 1.0)
    epsilon_decay   : multiplicative decay per episode (default 0.995)
    epsilon_min     : floor for epsilon (default 0.05)
    """

    def __init__(
        self,
        env,
        lr:            float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min:   float = 0.05,
    ):
        self.env           = env
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Sparse Q-table: state_tuple → dict{action: q_value}
        self.q_table: dict = defaultdict(lambda: defaultdict(float))

        # Training history
        self.episode_rewards: list[float] = []
        self.win_rates:       list[float] = []

    # ══════════════════════════════════════════════════════════════════════
    #  Core Q-learning methods
    # ══════════════════════════════════════════════════════════════════════

    def encode(self) -> tuple:
        """Return the current board as a hashable Q-table key."""
        return self.env.encode_state()

    def get_q(self, state: tuple, action: int) -> float:
        return self.q_table[state][action]

    def best_q(self, state: tuple, legal_actions: list[int]) -> float:
        """Max Q-value over legal actions (action masking)."""
        if not legal_actions:
            return 0.0
        return max(self.q_table[state][a] for a in legal_actions)

    def update(
        self,
        state:        tuple,
        action:       int,
        reward:       float,
        next_state:   tuple,
        done:         bool,
        next_legal:   list[int],
    ) -> None:
        """Apply the Q-learning update rule."""
        q_current = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.best_q(next_state, next_legal)

        self.q_table[state][action] = q_current + self.lr * (target - q_current)

    # ── exploration policy ────────────────────────────────────────────────

    def choose_action(self, state: tuple, legal_actions: list[int]) -> int:
        """
        ε-greedy action selection over legal actions only.

        With probability ε → random legal action  (exploration)
        With probability 1-ε → argmax Q(s,a)      (exploitation)
        """
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Greedy pick among legal actions
        return max(legal_actions, key=lambda a: self.q_table[state][a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ══════════════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════════

    def train(
        self,
        n_episodes:      int = 10_000,
        eval_every:      int = 500,
        eval_episodes:   int = 200,
        callback=None,            # called with (episode, win_rate, epsilon)
        stop_flag=None,           # threading.Event; training stops when set
    ) -> dict:
        """
        Training Loop
        ─────────────
        Runs n_episodes of self-play (agent vs built-in env opponent).
        After each eval_every episodes, evaluates win rate over eval_episodes
        greedy games.

        Returns
        -------
        dict with "episode_rewards", "win_rates", "q_table_size"
        """
        recent = []

        for ep in range(1, n_episodes + 1):
            if stop_flag and stop_flag.is_set():
                break

            state_arr = self.env.reset()
            state     = self.env.encode_state()
            total_r   = 0.0

            while True:
                legal = self.env.get_legal_actions()
                action = self.choose_action(state, legal)

                _, reward, done, _ = self.env.step(action)
                next_state  = self.env.encode_state()
                next_legal  = self.env.get_legal_actions()

                self.update(state, action, reward, next_state, done, next_legal)

                total_r += reward
                state    = next_state

                if done:
                    break

            self.decay_epsilon()
            self.episode_rewards.append(total_r)
            recent.append(total_r)

            if ep % eval_every == 0:
                wr = self.evaluate(eval_episodes)
                self.win_rates.append(wr)
                if callback:
                    callback(ep, wr, self.epsilon)

        return {
            "episode_rewards": self.episode_rewards,
            "win_rates":       self.win_rates,
            "q_table_size":    sum(len(v) for v in self.q_table.values()),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  Evaluation mode  (greedy, ε = 0)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(self, n_episodes: int = 200) -> float:
        """
        Evaluation Mode
        ───────────────
        Runs n_episodes with ε = 0 (purely greedy).
        Returns win rate (fraction of episodes where agent scored reward > 0).
        """
        saved_eps = self.epsilon
        self.epsilon = 0.0
        wins = 0

        for _ in range(n_episodes):
            self.env.reset()
            state = self.env.encode_state()
            while True:
                legal = self.env.get_legal_actions()
                action = self.choose_action(state, legal)
                _, reward, done, _ = self.env.step(action)
                state = self.env.encode_state()
                if done:
                    if reward >= 1.0:  # win=+1.0; draw=+0.2 must NOT count as win
                        wins += 1
                    break

        self.epsilon = saved_eps
        return wins / n_episodes

    # ══════════════════════════════════════════════════════════════════════
    #  Inference  (used by QLearningOpponent in opponents.py)
    # ══════════════════════════════════════════════════════════════════════

    def predict(self, state: tuple, legal_actions: list[int]) -> int:
        """Greedy action (no exploration). Used at play-time."""
        if not legal_actions:
            return -1
        return max(legal_actions, key=lambda a: self.q_table[state][a])

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self, path: str) -> None:
        data = {
            "q_table":        dict((k, dict(v)) for k, v in self.q_table.items()),
            "epsilon":        self.epsilon,
            "episode_rewards": self.episode_rewards,
            "win_rates":      self.win_rates,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data["q_table"].items():
            for a, q in actions.items():
                self.q_table[state][a] = q
        self.epsilon          = data.get("epsilon", self.epsilon_min)
        self.episode_rewards  = data.get("episode_rewards", [])
        self.win_rates        = data.get("win_rates", [])

    def info(self) -> dict:
        return {
            "agent":         "TabularQLearning",
            "lr":            self.lr,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "q_table_states": len(self.q_table),
            "q_table_entries": sum(len(v) for v in self.q_table.values()),
            "episodes_trained": len(self.episode_rewards),
        }
