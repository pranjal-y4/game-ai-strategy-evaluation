"""
agents/qlearning_ttt_agent.py
Wrapper that loads a trained TabularQLearning model and exposes the
standard BaseAgent interface for headless evaluation on TicTacToe.

State-encoding contract
───────────────────────
The Q-table was trained with TicTacToeEnv.encode_state() which encodes
states from the AGENT's perspective (matches rl/env.py _state_from_agent):
    +1  = own piece          0  = empty cell         -1  = opponent's piece
Q-table keys are tuples of (+1|0|-1) integers.

At inference time we convert the raw game board (1|2|0) to the same
perspective encoding regardless of which player number we are assigned.

Action encoding
───────────────
TicTacToe.legal_moves() returns (row, col) tuples.
The Q-table uses flat indices: flat = row * 3 + col.
We convert both ways.
"""

import os
import random
import pickle
from collections import defaultdict
from .base_agent import BaseAgent

# Absolute path to the default model
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "ttt_qlearning.pkl"
)


class QLearningTTTAgent(BaseAgent):
    """Tabular Q-learning agent for Tic-Tac-Toe (inference only)."""

    def __init__(self, model_path: str = None):
        self.model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        # q_table[state_tuple][flat_action] → float
        self.q_table: dict = defaultdict(lambda: defaultdict(float))
        self._loaded = False
        self._try_load()

    @property
    def name(self) -> str:
        return "QLearning_TTT"

    def reset(self):
        pass

    def select_action(self, game, training: bool = False):
        """
        Greedy action selection with perspective normalisation.
        Falls back to a random legal move if no model is loaded.
        """
        me = game.current_player          # 1 or 2
        opp = 3 - me                      # the other player id
        raw_state = game.encode_state()   # tuple of (1|2|0) raw board values

        # Convert to agent-perspective encoding matching rl/env.py training:
        # own piece = +1, empty = 0, opponent = -1
        state = tuple(1 if x == me else (-1 if x == opp else 0) for x in raw_state)

        legal_moves = game.legal_moves()  # list of (row, col) tuples
        legal_flat  = [r * 3 + c for r, c in legal_moves]

        if not self._loaded or not legal_flat:
            return random.choice(legal_moves) if legal_moves else None

        # Greedy pick over legal actions
        best_flat = max(legal_flat, key=lambda a: self.q_table[state][a])
        return (best_flat // 3, best_flat % 3)

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load(self):
        """Load the Q-table from disk if the model file exists."""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            raw = data.get("q_table", {})
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in raw.items():
                for a, q in actions.items():
                    self.q_table[state][a] = q
            self._loaded = True
        except Exception as e:
            print(f"[QLearningTTTAgent] Warning: could not load model: {e}")
