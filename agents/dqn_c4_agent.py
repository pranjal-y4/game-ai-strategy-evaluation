"""
agents/dqn_c4_agent.py
Wrapper that loads a trained DQN model for the full 6×7 Connect 4 board
and exposes the standard BaseAgent interface.

State encoding
──────────────
DQN was trained with Connect4Env(rows=6, cols=7) using the encoding:
    +1.0  = agent's own piece
     0.0  = empty cell
    -1.0  = opponent's piece
(numpy float32 array of length 42, flattened row-major)

When this agent plays as player 2 we still encode MY pieces as +1 and
the opponent's pieces as -1.

Actions
───────
Column indices 0-6. Connect4.legal_moves() already returns column ints.
No conversion needed.
"""

import os
import random
import numpy as np
from .base_agent import BaseAgent

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "c4_dqn.pkl"
)


class DQNC4Agent(BaseAgent):
    """DQN agent for full 6×7 Connect 4 (inference only)."""

    def __init__(self, model_path: str = None, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        self._net = None
        self._n_actions = cols
        self._try_load()

    @property
    def name(self) -> str:
        return "DQN_C4"

    def reset(self):
        pass

    def select_action(self, game, training: bool = False):
        """
        Greedy action using the loaded DQN online network.
        Falls back to random if dimensions don't match or no model was loaded.
        """
        if game.rows != self.rows or game.cols != self.cols:
            legal = game.legal_moves()
            return random.choice(legal) if legal else None

        me = game.current_player
        legal = game.legal_moves()   # column indices

        if not legal:
            return None

        if self._net is None:
            return random.choice(legal)

        # Build state vector from agent's perspective
        raw = game.encode_state()  # tuple of (1|2|0), length rows*cols
        state = np.array(
            [1.0 if x == me else (-1.0 if x != 0 else 0.0) for x in raw],
            dtype=np.float32
        )

        q_vals = self._net.forward(state).flatten()
        mask = np.full(self._n_actions, -np.inf)
        mask[legal] = q_vals[legal]
        return int(np.argmax(mask))

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            import pickle
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            from rl.dqn import _MLP
            weights = data["online_weights"]
            layer_sizes = [weights[0][0].shape[0]]
            for w, _ in weights:
                layer_sizes.append(w.shape[1])
            net = _MLP(layer_sizes, lr=1e-3)
            net.set_weights(weights)
            self._net = net
            self._n_actions = layer_sizes[-1]
        except Exception as e:
            print(f"[DQNC4Agent] Warning: could not load model: {e}")
