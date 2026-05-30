"""
agents/qlearning_c4_reduced_agent.py
Wrapper for the tabular Q-learning model trained on a reduced Connect 4 board.

IMPORTANT: This agent only works on the REDUCED board (default 4×5).
Do NOT use it against Connect4(rows=6, cols=7) agents; results are NOT comparable.
See README.txt for the justification of the reduced-board choice.

State-encoding contract
───────────────────────
The Q-table was trained with Connect4Env.encode_state() which encodes
states from the AGENT's perspective (matches rl/env.py _state_from_agent):
    +1  = own piece          0  = empty cell         -1  = opponent's piece
Q-table keys are tuples of (+1|0|-1) integers.

At inference time we convert the raw game board (1|2|0) to the same
perspective encoding regardless of which player number we are assigned.

Actions
───────
Connect4.legal_moves() returns column indices (ints) — same as Q-table actions.
No conversion needed.
"""

import os
import random
import pickle
from collections import defaultdict
from .base_agent import BaseAgent

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "c4_qlearning_4x5.pkl"
)


class QLearningC4ReducedAgent(BaseAgent):
    """Tabular Q-learning agent for the reduced 4×5 Connect 4 board."""

    def __init__(self, model_path: str = None, rows: int = 4, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.model_path = os.path.abspath(model_path or _DEFAULT_MODEL)
        self.q_table: dict = defaultdict(lambda: defaultdict(float))
        self._loaded = False
        self._try_load()

    @property
    def name(self) -> str:
        return f"QLearning_C4_{self.rows}x{self.cols}"

    def reset(self):
        pass

    def select_action(self, game, training: bool = False):
        """
        Greedy action on the reduced board.
        Falls back to a random legal move if board dimensions don't match
        or no model is loaded.
        """
        if game.rows != self.rows or game.cols != self.cols:
            # Dimension mismatch — random fallback (should not happen in correct experiments)
            legal = game.legal_moves()
            return random.choice(legal) if legal else None

        me = game.current_player
        opp = 3 - me                      # the other player id
        raw_state = game.encode_state()   # tuple of (1|2|0) raw board values

        # Convert to agent-perspective encoding matching rl/env.py training:
        # own piece = +1, empty = 0, opponent = -1
        state = tuple(1 if x == me else (-1 if x == opp else 0) for x in raw_state)

        legal = game.legal_moves()  # list of column integers

        if not self._loaded or not legal:
            return random.choice(legal) if legal else None

        return max(legal, key=lambda a: self.q_table[state][a])

    # ── persistence ──────────────────────────────────────────────────────────

    def _try_load(self):
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
            print(f"[QLearningC4ReducedAgent] Warning: could not load model: {e}")
