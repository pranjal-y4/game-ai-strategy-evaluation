"""
agents/dqn_ttt_agent.py
Wrapper that loads a trained DQN model and exposes the BaseAgent interface
for headless Tic-Tac-Toe evaluation.

State encoding
──────────────
DQN was trained with TicTacToeEnv which encodes the board as:
    +1.0  = agent's own piece
     0.0  = empty cell
    -1.0  = opponent's piece
(numpy float32 array of length 9, flattened row-major)

When this agent plays as player 2 we still encode MY pieces as +1 and
the opponent's pieces as -1 before passing to the network.

Action encoding
───────────────
DQN actions are flat indices 0-8.
TicTacToe.legal_moves() returns (row, col) tuples.
We convert: flat = row * 3 + col   and back: (flat // 3, flat % 3).
"""

import os
import random
import numpy as np
from .base_agent import BaseAgent

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "ttt_dqn.pkl"
)


class DQNTTTAgent(BaseAgent):
    """DQN agent for Tic-Tac-Toe (inference only)."""

    def __init__(self, model_path: str = None):
        self.model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        self._net = None   # online network (_MLP instance)
        self._n_actions = 9
        self._try_load()

    @property
    def name(self) -> str:
        return "DQN_TTT"

    def reset(self):
        pass

    def select_action(self, game, training: bool = False):
        """
        Greedy action using the loaded DQN online network.
        Falls back to random if no model was loaded.
        """
        me = game.current_player
        legal_moves = game.legal_moves()   # list of (row, col)
        legal_flat  = [r * 3 + c for r, c in legal_moves]

        if not legal_flat:
            return None

        if self._net is None:
            return random.choice(legal_moves)

        # Build state vector: +1 own, -1 opp, 0 empty
        raw = game.encode_state()  # tuple of (1|2|0)
        state = np.array(
            [1.0 if x == me else (-1.0 if x != 0 else 0.0) for x in raw],
            dtype=np.float32
        )

        q_vals = self._net.forward(state).flatten()
        # Mask illegal actions to -inf
        mask = np.full(self._n_actions, -np.inf)
        mask[legal_flat] = q_vals[legal_flat]
        best_flat = int(np.argmax(mask))
        return (best_flat // 3, best_flat % 3)

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            import pickle
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            # Reconstruct the online network from saved weights
            from rl.dqn import _MLP
            weights = data["online_weights"]
            # Infer layer sizes from weight shapes
            layer_sizes = [weights[0][0].shape[0]]
            for w, _ in weights:
                layer_sizes.append(w.shape[1])
            net = _MLP(layer_sizes, lr=1e-3)
            net.set_weights(weights)
            self._net = net
            self._n_actions = layer_sizes[-1]
        except Exception as e:
            print(f"[DQNTTTAgent] Warning: could not load model: {e}")
