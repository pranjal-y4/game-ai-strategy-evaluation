from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


THIS_FILE = Path(__file__).resolve()
ADV_ROOT = THIS_FILE.parents[1]
PROJ_ROOT = THIS_FILE.parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))
if str(ADV_ROOT) not in sys.path:

    sys.path.insert(0, str(ADV_ROOT))

from agents.base_agent import BaseAgent
from agents.default_agent import DefaultAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.advanced_alphabeta_c4 import AdvancedAlphaBetaC4Agent
from evaluation.evaluator import evaluate_agent, _set_greedy
from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from rl.game_env import GameEnv
from rl.q_learning import AdvancedQLearning


class QLModelAdapter(BaseAgent):


    def __init__(self, model_path: str, game_cls, game_kwargs: dict | None, label: str):
        self._name = label
        self._is_ttt = game_cls == TicTacToe
        env = GameEnv(game_cls, game_kwargs=game_kwargs or {}, opponent="random")
        self._q = AdvancedQLearning(env)
        self._q.load(model_path)
        self._q.epsilon = 0.0

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, game, training: bool = False):
        p = game.current_player
        state = game.encode_state_hashable(perspective_player=p)
        if self._is_ttt:
            legal_flat = [r * 3 + c for (r, c) in game.legal_moves()]
            action = self._q.predict(state, legal_flat)
            return (action // 3, action % 3)
        legal_cols = game.legal_moves()
        return self._q.predict(state, legal_cols)


class DQNModelAdapter(BaseAgent):


    def __init__(self, model_path: str, game_cls, game_kwargs: dict | None, label: str):

        from rl.dqn import AdvancedDQNAgent

        self._name = label
        self._is_ttt = game_cls == TicTacToe
        hidden = [128, 64] if self._is_ttt else [256, 128]

        env = GameEnv(game_cls, game_kwargs=game_kwargs or {}, opponent="random")
        self._dqn = AdvancedDQNAgent(env, hidden=hidden)
        self._dqn.load(model_path)
        self._dqn.epsilon = 0.0

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, game, training: bool = False):
        p = game.current_player
        state = game.encode_state(perspective_player=p)
        if self._is_ttt:
            legal = [r * 3 + c for (r, c) in game.legal_moves()]
            action = self._dqn._greedy(state, legal)
            return (action // 3, action % 3)
        legal = game.legal_moves()
        return self._dqn._greedy(state, legal)


@dataclass
class AgentSpec:
    key: str
    label: str
    game_key: str
    source: str
    factory: callable


@dataclass
class SessionStats:
    games: int = 0
    human_wins: int = 0
    ai_wins: int = 0
    draws: int = 0
    human_as_p1_games: int = 0
    human_as_p1_wins: int = 0
    human_as_p1_losses: int = 0
    human_as_p2_games: int = 0
    human_as_p2_wins: int = 0
    human_as_p2_losses: int = 0
    human_move_times: list[float] = field(default_factory=list)
    ai_move_times: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.games = 0
        self.human_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.human_as_p1_games = 0
        self.human_as_p1_wins = 0
        self.human_as_p1_losses = 0
        self.human_as_p2_games = 0
        self.human_as_p2_wins = 0
        self.human_as_p2_losses = 0
        self.human_move_times.clear()
        self.ai_move_times.clear()

    def record_game(self, winner: int, human_player: int) -> None:
        self.games += 1
        if human_player == 1:
            self.human_as_p1_games += 1
        else:
            self.human_as_p2_games += 1

        if winner == 0:
            self.draws += 1
        elif winner == human_player:
            self.human_wins += 1
            if human_player == 1:
                self.human_as_p1_wins += 1
            else:
                self.human_as_p2_wins += 1
        else:
            self.ai_wins += 1
            if human_player == 1:
                self.human_as_p1_losses += 1
            else:
                self.human_as_p2_losses += 1

    @staticmethod
    def _rate(a: int, b: int) -> float:
        return float(a / b) if b > 0 else 0.0

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _probe_torch_runtime(py_executable: str) -> tuple[bool, str]:


    code = "import torch; print(torch.__version__)"
    try:
        proc = subprocess.run(
            [py_executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
            check=False,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "unknown"
            return True, f"torch={version}"
        msg = proc.stderr.strip() or proc.stdout.strip() or "torch import failed"
        return False, msg.splitlines()[-1]
    except Exception as exc:
        return False, str(exc)


class GameArenaApp(tk.Tk):
    BG = "#f3f6fb"
    CARD = "#ffffff"
    ACCENT = "#2f6fed"
    ACCENT_SOFT = "#e8efff"
    TEXT = "#1f2a44"
    MUTED = "#667085"
    BORDER = "#dbe3f0"
    WIN = "#16a34a"
    LOSE = "#dc2626"
    DRAW = "#a16207"

    def __init__(self):
        super().__init__()
        self.title("Model Arena - TicTacToe & Connect4")
        self.geometry("1440x900")
        self.minsize(1220, 760)
        self.configure(bg=self.BG)

        self.model_dir = ADV_ROOT / "models"

        self._setup_style()

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arena-worker")
        self.agent_lock = threading.Lock()
        self.ai_future: Future | None = None
        self.eval_future: Future | None = None
        self.baseline_future: Future | None = None

        self.active_game_token = 0
        self.game = None
        self.game_key = "ttt"
        self.ai_agent: BaseAgent | None = None
        self.human_player = 1
        self.ai_player = 2
        self.move_index = 0
        self.game_over = False
        self.pending_ai_token = -1
        self.turn_started_ts = 0.0
        self.last_move_text = "None"
        self.status_text = "Ready"
        self.quick_eval_result = "Not run yet"
        self.session_key = None
        self.baseline_cache: dict[tuple[str, int], dict] = {}
        self.pending_baseline_key: tuple[str, int] | None = None
        self.pending_baseline_spec_key: str | None = None

        self.session_stats = SessionStats()


        self.torch_ok, self.torch_reason = _probe_torch_runtime(sys.executable)
        self.agent_specs: dict[str, list[AgentSpec]] = {"ttt": [], "c4": []}
        self._build_agent_specs()

        self._init_state_vars()
        self._build_layout()
        self._sync_agent_options()
        self.start_new_game()

        self.protocol("WM_DELETE_WINDOW", self._on_close)


    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Card.TFrame", background=self.CARD, relief="flat")
        style.configure("App.TLabel", background=self.BG, foreground=self.TEXT, font=("Avenir Next", 11))
        style.configure("CardTitle.TLabel", background=self.CARD, foreground=self.TEXT, font=("Avenir Next", 14, "bold"))
        style.configure("CardValue.TLabel", background=self.CARD, foreground=self.TEXT, font=("Avenir Next", 11))
        style.configure("Muted.TLabel", background=self.CARD, foreground=self.MUTED, font=("Avenir Next", 10))
        style.configure("Accent.TButton", font=("Avenir Next", 10, "bold"), padding=(12, 8))
        style.map("Accent.TButton", background=[("active", self.ACCENT)], foreground=[("active", "white")])
        style.configure("TCombobox", padding=6)
        style.configure("TRadiobutton", background=self.CARD, foreground=self.TEXT, font=("Avenir Next", 10))
        style.configure("TSpinbox", padding=5)

    def _init_state_vars(self) -> None:
        self.selected_game = tk.StringVar(value="TicTacToe")
        self.selected_agent = tk.StringVar(value="")
        self.human_side = tk.StringVar(value="Player 1 (First)")
        self.quick_eval_games = tk.IntVar(value=30)

        self.turn_var = tk.StringVar(value="Turn: -")
        self.result_var = tk.StringVar(value="Result: -")
        self.moves_var = tk.StringVar(value="Moves: 0")
        self.last_move_var = tk.StringVar(value="Last move: None")
        self.status_var = tk.StringVar(value="Ready")

        self.session_games_var = tk.StringVar(value="0")
        self.session_human_var = tk.StringVar(value="0.0%")
        self.session_ai_var = tk.StringVar(value="0.0%")
        self.session_draw_var = tk.StringVar(value="0.0%")
        self.session_p1_var = tk.StringVar(value="0.0%")
        self.session_p2_var = tk.StringVar(value="0.0%")
        self.session_fair_human_var = tk.StringVar(value="Need both sides")
        self.session_fair_ai_var = tk.StringVar(value="Need both sides")
        self.session_fair_draw_var = tk.StringVar(value="Need both sides")
        self.session_balance_var = tk.StringVar(value="P1:0 | P2:0")
        self.session_human_time_var = tk.StringVar(value="0 ms")
        self.session_ai_time_var = tk.StringVar(value="0 ms")

        self.baseline_wr_var = tk.StringVar(value="Pending")
        self.baseline_draw_var = tk.StringVar(value="Pending")
        self.baseline_p1_var = tk.StringVar(value="Pending")
        self.baseline_p2_var = tk.StringVar(value="Pending")
        self.baseline_time_var = tk.StringVar(value="Pending")
        self.quick_eval_var = tk.StringVar(value="Not run yet")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Card.TFrame")
        root.pack(fill="both", expand=True, padx=18, pady=18)

        root.grid_columnconfigure(0, weight=0, minsize=320)
        root.grid_columnconfigure(1, weight=1, minsize=640)
        root.grid_columnconfigure(2, weight=0, minsize=360)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)

        header = tk.Frame(root, bg=self.BG)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 14))

        title = tk.Label(
            header,
            text="Model Arena",
            bg=self.BG,
            fg=self.TEXT,
            font=("Avenir Next", 26, "bold"),
        )
        subtitle = tk.Label(
            header,
            text="Play against saved agents and watch live performance metrics update in real time.",
            bg=self.BG,
            fg=self.MUTED,
            font=("Avenir Next", 11),
        )
        title.pack(anchor="w")
        subtitle.pack(anchor="w", pady=(2, 0))

        self.left_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        self.left_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self._build_left_controls(self.left_card)

        self.center_card = ttk.Frame(root, style="Card.TFrame", padding=14)
        self.center_card.grid(row=1, column=1, sticky="nsew", padx=(0, 12))
        self._build_center_board(self.center_card)

        self.right_card = ttk.Frame(root, style="Card.TFrame", padding=16)
        self.right_card.grid(row=1, column=2, sticky="nsew")
        self._build_right_metrics(self.right_card)

        self.log_card = ttk.Frame(root, style="Card.TFrame", padding=12)
        self.log_card.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self._build_log_area(self.log_card)

    def _build_left_controls(self, parent) -> None:
        ttk.Label(parent, text="Match Setup", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        ttk.Label(parent, text="Game", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(16, 4))
        game_combo = ttk.Combobox(
            parent,
            state="readonly",
            textvariable=self.selected_game,
            values=["TicTacToe", "Connect4"],
            width=28,
        )
        game_combo.grid(row=2, column=0, sticky="ew")
        game_combo.bind("<<ComboboxSelected>>", self._on_game_changed)

        ttk.Label(parent, text="Opponent", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(16, 4))
        self.agent_combo = ttk.Combobox(
            parent,
            state="readonly",
            textvariable=self.selected_agent,
            values=[],
            width=28,
        )
        self.agent_combo.grid(row=4, column=0, sticky="ew")
        self.agent_combo.bind("<<ComboboxSelected>>", self._on_agent_changed)

        ttk.Label(parent, text="You play as", style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(16, 6))
        ttk.Radiobutton(
            parent,
            text="Player 1 (First)",
            variable=self.human_side,
            value="Player 1 (First)",
        ).grid(row=6, column=0, sticky="w")
        ttk.Radiobutton(
            parent,
            text="Player 2 (Second)",
            variable=self.human_side,
            value="Player 2 (Second)",
        ).grid(row=7, column=0, sticky="w", pady=(2, 10))

        btn_row = tk.Frame(parent, bg=self.CARD)
        btn_row.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        ttk.Button(btn_row, text="New Game", command=self.start_new_game).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btn_row, text="Reset Session", command=self.reset_session).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=9, column=0, sticky="ew", pady=16)

        ttk.Label(parent, text="Quick Evaluation", style="CardTitle.TLabel").grid(row=10, column=0, sticky="w")
        ttk.Label(parent, text="Games vs Default opponent", style="Muted.TLabel").grid(row=11, column=0, sticky="w", pady=(12, 4))
        spin = tk.Spinbox(
            parent,
            from_=10,
            to=500,
            increment=10,
            textvariable=self.quick_eval_games,
            width=8,
            bd=1,
            relief="solid",
            highlightthickness=0,
        )
        spin.grid(row=12, column=0, sticky="w")
        ttk.Button(parent, text="Run Quick Eval", command=self.run_quick_eval).grid(row=13, column=0, sticky="ew", pady=(10, 0))

        torch_note = "DQN models: enabled" if self.torch_ok else f"DQN models: disabled ({self.torch_reason})"
        tk.Label(parent, text=torch_note, bg=self.CARD, fg=self.MUTED, font=("Avenir Next", 9)).grid(
            row=14, column=0, sticky="w", pady=(14, 0)
        )

    def _build_center_board(self, parent) -> None:
        top = tk.Frame(parent, bg=self.CARD)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Live Board", style="CardTitle.TLabel").pack(side="left")
        self.status_chip = tk.Label(
            top,
            textvariable=self.status_var,
            bg=self.ACCENT_SOFT,
            fg=self.ACCENT,
            font=("Avenir Next", 10, "bold"),
            padx=10,
            pady=4,
        )
        self.status_chip.pack(side="right")

        self.canvas_frame = tk.Frame(parent, bg=self.CARD)
        self.canvas_frame.pack(fill="both", expand=True)

        self.board_canvas = tk.Canvas(
            self.canvas_frame,
            bg=self.CARD,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.board_canvas.pack(fill="both", expand=True)
        self.board_canvas.bind("<Button-1>", self._on_canvas_click)
        self.board_canvas.bind("<Motion>", self._on_canvas_motion)
        self.board_canvas.bind("<Leave>", self._on_canvas_leave)

    def _build_right_metrics(self, parent) -> None:
        ttk.Label(parent, text="Live Metrics", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        game_box = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        game_box.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        self._metric_row(game_box, "Turn", self.turn_var, 0)
        self._metric_row(game_box, "Result", self.result_var, 1)
        self._metric_row(game_box, "Moves", self.moves_var, 2)
        self._metric_row(game_box, "Last move", self.last_move_var, 3)

        session_title = ttk.Label(parent, text="Session vs Current Opponent", style="CardTitle.TLabel")
        session_title.grid(row=2, column=0, sticky="w", pady=(8, 6))
        session_box = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        session_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._metric_row(session_box, "Games", self.session_games_var, 0)
        self._metric_row(session_box, "Your win rate", self.session_human_var, 1)
        self._metric_row(session_box, "Model win rate", self.session_ai_var, 2)
        self._metric_row(session_box, "Draw rate", self.session_draw_var, 3)
        self._metric_row(session_box, "Your WR as P1", self.session_p1_var, 4)
        self._metric_row(session_box, "Your WR as P2", self.session_p2_var, 5)
        self._metric_row(session_box, "Fair WR (you)", self.session_fair_human_var, 6)
        self._metric_row(session_box, "Fair WR (model)", self.session_fair_ai_var, 7)
        self._metric_row(session_box, "Fair draw rate", self.session_fair_draw_var, 8)
        self._metric_row(session_box, "Role balance", self.session_balance_var, 9)
        self._metric_row(session_box, "Avg your move", self.session_human_time_var, 10)
        self._metric_row(session_box, "Avg model move", self.session_ai_time_var, 11)

        baseline_title = ttk.Label(parent, text="Live Baseline (Fair Eval vs Default)", style="CardTitle.TLabel")
        baseline_title.grid(row=4, column=0, sticky="w", pady=(8, 6))
        baseline_box = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        baseline_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self._metric_row(baseline_box, "Win rate vs Default", self.baseline_wr_var, 0)
        self._metric_row(baseline_box, "Draw rate vs Default", self.baseline_draw_var, 1)
        self._metric_row(baseline_box, "P1 win rate", self.baseline_p1_var, 2)
        self._metric_row(baseline_box, "P2 win rate", self.baseline_p2_var, 3)
        self._metric_row(baseline_box, "Avg move time", self.baseline_time_var, 4)

        quick_title = ttk.Label(parent, text="Quick Eval Result", style="CardTitle.TLabel")
        quick_title.grid(row=6, column=0, sticky="w", pady=(8, 6))
        quick_box = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        quick_box.grid(row=7, column=0, sticky="ew")
        tk.Label(
            quick_box,
            textvariable=self.quick_eval_var,
            justify="left",
            wraplength=320,
            bg=self.CARD,
            fg=self.TEXT,
            font=("Avenir Next", 10),
            padx=10,
            pady=10,
            anchor="w",
        ).pack(fill="both", expand=True)

    def _metric_row(self, parent, label: str, value_var: tk.StringVar, row: int) -> None:
        tk.Label(parent, text=label, bg=self.CARD, fg=self.MUTED, font=("Avenir Next", 10)).grid(
            row=row, column=0, sticky="w", padx=10, pady=4
        )
        tk.Label(parent, textvariable=value_var, bg=self.CARD, fg=self.TEXT, font=("Avenir Next", 10, "bold")).grid(
            row=row, column=1, sticky="e", padx=10, pady=4
        )
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=0)

    def _build_log_area(self, parent) -> None:
        header = tk.Frame(parent, bg=self.CARD)
        header.pack(fill="x")
        ttk.Label(header, text="Move Log", style="CardTitle.TLabel").pack(side="left")

        self.log_text = tk.Text(
            parent,
            height=7,
            bg="#fbfdff",
            fg=self.TEXT,
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            wrap="word",
            font=("Menlo", 10),
        )
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))
        self.log_text.configure(state="disabled")


    def _build_agent_specs(self) -> None:
        self.agent_specs = {"ttt": [], "c4": []}

        def add_spec(spec: AgentSpec) -> None:
            self.agent_specs[spec.game_key].append(spec)


        add_spec(AgentSpec("ttt_random", "Random", "ttt", "built-in", lambda: RandomAgent()))
        add_spec(AgentSpec("ttt_default", "Default", "ttt", "built-in", lambda: DefaultAgent()))
        add_spec(AgentSpec("ttt_minimax5", "Minimax(d=5)", "ttt", "built-in", lambda: MinimaxAgent(max_depth=5)))
        add_spec(AgentSpec("ttt_alphabeta5", "AlphaBeta(d=5)", "ttt", "built-in", lambda: AlphaBetaAgent(max_depth=5)))

        add_spec(AgentSpec("c4_random", "Random", "c4", "built-in", lambda: RandomAgent()))
        add_spec(AgentSpec("c4_default", "Default", "c4", "built-in", lambda: DefaultAgent()))
        add_spec(AgentSpec("c4_alphabeta5", "AlphaBeta(d=5)", "c4", "built-in", lambda: AlphaBetaAgent(max_depth=5)))
        add_spec(
            AgentSpec(
                "c4_advab5",
                "AdvAB_C4(d=5)",
                "c4",
                "built-in",
                lambda: AdvancedAlphaBetaC4Agent(max_depth=5),
            )
        )


        ttt_ql = self.model_dir / "ttt_qlearning_default.pkl"
        if ttt_ql.exists():
            add_spec(
                AgentSpec(
                    "ttt_ql_model",
                    "QL_TTT (saved)",
                    "ttt",
                    ttt_ql.name,
                    lambda p=str(ttt_ql): QLModelAdapter(p, TicTacToe, None, "QL_TTT"),
                )
            )

        c4_ql = self.model_dir / "c4_qlearning_6x7_default.pkl"
        if c4_ql.exists():
            add_spec(
                AgentSpec(
                    "c4_ql_model",
                    "QL_C4 (saved 6x7)",
                    "c4",
                    c4_ql.name,
                    lambda p=str(c4_ql): QLModelAdapter(p, Connect4, {"rows": 6, "cols": 7}, "QL_C4"),
                )
            )


        if self.torch_ok:
            ttt_dqn = self.model_dir / "ttt_dqn_default.pt"
            if ttt_dqn.exists():
                add_spec(
                    AgentSpec(
                        "ttt_dqn_model",
                        "DQN_TTT (saved)",
                        "ttt",
                        ttt_dqn.name,
                        lambda p=str(ttt_dqn): DQNModelAdapter(p, TicTacToe, None, "DQN_TTT"),
                    )
                )

            c4_dqn = self.model_dir / "c4_dqn_6x7_default.pt"
            if c4_dqn.exists():
                add_spec(
                    AgentSpec(
                        "c4_dqn_model",
                        "DQN_C4 (saved 6x7)",
                        "c4",
                        c4_dqn.name,
                        lambda p=str(c4_dqn): DQNModelAdapter(p, Connect4, {"rows": 6, "cols": 7}, "DQN_C4"),
                    )
                )

    def _baseline_cache_key(self, spec: AgentSpec, n_games: int) -> tuple[str, int]:
        return (spec.key, int(n_games))

    def _request_baseline_eval(self, n_games: int = 80) -> None:

        spec = self._get_selected_spec()
        if spec is None:
            self._set_baseline_pending("N/A")
            return

        key = self._baseline_cache_key(spec, n_games)
        if key in self.baseline_cache:
            self._apply_baseline_result(self.baseline_cache[key])
            return

        if self.baseline_future and not self.baseline_future.done():

            return

        self.pending_baseline_key = key
        self.pending_baseline_spec_key = spec.key
        self._set_baseline_pending(f"Running {n_games} games...")
        self.baseline_future = self.executor.submit(self._quick_eval_worker, spec, n_games)
        self.after(90, self._poll_baseline_future)

    def _poll_baseline_future(self) -> None:
        if self.baseline_future is None:
            return
        if not self.baseline_future.done():
            self.after(90, self._poll_baseline_future)
            return

        try:
            result = self.baseline_future.result()
        except Exception as exc:
            self._set_baseline_pending("Eval failed")
            self.log_line(f"Baseline eval failed: {exc}")
            return

        key = self.pending_baseline_key
        if key is not None:
            self.baseline_cache[key] = result

        active_spec = self._get_selected_spec()
        if active_spec and active_spec.key == self.pending_baseline_spec_key:
            self._apply_baseline_result(result)
            self.log_line(
                f"Baseline ready ({result['agent']}): WR {100*result['total_win_rate']:.1f}%, "
                f"Draw {100*result['total_draw_rate']:.1f}%"
            )

        self.pending_baseline_key = None
        self.pending_baseline_spec_key = None

    def _set_baseline_pending(self, text: str) -> None:
        self.baseline_wr_var.set(text)
        self.baseline_draw_var.set(text)
        self.baseline_p1_var.set(text)
        self.baseline_p2_var.set(text)
        self.baseline_time_var.set(text)

    def _apply_baseline_result(self, row: dict) -> None:
        self.baseline_wr_var.set(f"{100*_safe_float(row.get('total_win_rate', '0')):.1f}%")
        self.baseline_draw_var.set(f"{100*_safe_float(row.get('total_draw_rate', '0')):.1f}%")
        self.baseline_p1_var.set(f"{100*_safe_float(row.get('p1_win_rate', '0')):.1f}%")
        self.baseline_p2_var.set(f"{100*_safe_float(row.get('p2_win_rate', '0')):.1f}%")
        self.baseline_time_var.set(f"{_safe_float(row.get('avg_agent_time_ms_per_move', '0')):.2f} ms")


    def _on_game_changed(self, _event=None) -> None:
        self._sync_agent_options()
        self.start_new_game()

    def _on_agent_changed(self, _event=None) -> None:
        self._request_baseline_eval()
        self.start_new_game()

    def _sync_agent_options(self) -> None:
        self.game_key = "ttt" if self.selected_game.get() == "TicTacToe" else "c4"
        labels = [spec.label for spec in self.agent_specs[self.game_key]]
        self.agent_combo["values"] = labels
        if not labels:
            self.selected_agent.set("")
            return
        if self.selected_agent.get() not in labels:
            self.selected_agent.set(labels[0])
        self._request_baseline_eval()

    def _on_close(self) -> None:
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.destroy()


    def _get_selected_spec(self) -> AgentSpec | None:
        label = self.selected_agent.get().strip()
        for spec in self.agent_specs[self.game_key]:
            if spec.label == label:
                return spec
        return None

    def start_new_game(self) -> None:
        spec = self._get_selected_spec()
        if spec is None:
            messagebox.showerror("No Opponent", "No agent is available for this game.")
            return


        new_session_key = (self.game_key, spec.label)
        if self.session_key != new_session_key:
            self.session_stats.reset()
            self.session_key = new_session_key

        self.active_game_token += 1
        self.pending_ai_token = -1
        self.game_over = False
        self.move_index = 0
        self.last_move_text = "None"
        self.quick_eval_var.set("Not run yet")

        try:
            self.ai_agent = spec.factory()
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            messagebox.showerror("Agent Load Failed", f"Could not load '{spec.label}'.\n\n{exc}\n\n{tb}")
            return

        if self.game_key == "ttt":
            self.game = TicTacToe()
        else:
            self.game = Connect4(rows=6, cols=7)

        if self.human_side.get().startswith("Player 1"):
            self.human_player = 1
            self.ai_player = 2
        else:
            self.human_player = 2
            self.ai_player = 1

        self.game.reset()
        if self.ai_agent:
            self.ai_agent.reset()
        self.log_text_clear()
        self.log_line(f"New game started: {self.selected_game.get()} vs {spec.label}")
        self.log_line(f"You are Player {self.human_player}.")

        self.result_var.set("Result: Ongoing")
        self.moves_var.set("Moves: 0")
        self.last_move_var.set("Last move: None")

        self._refresh_turn_text()
        self._draw_board()
        self._update_session_panel()
        self._request_baseline_eval()

        if self.game.current_player == self.ai_player:
            self._queue_ai_turn()
        else:
            self.turn_started_ts = time.perf_counter()
            self.status_var.set("Your turn")

    def reset_session(self) -> None:
        self.session_stats.reset()
        self._update_session_panel()
        self.log_line("Session metrics reset.")


    def _draw_board(self) -> None:
        self.board_canvas.delete("all")
        if self.game_key == "ttt":
            self._draw_ttt_board()
        else:
            self._draw_c4_board()

    def _draw_ttt_board(self) -> None:
        width = self.board_canvas.winfo_width() or 700
        height = self.board_canvas.winfo_height() or 700
        size = min(width, height) - 80
        size = max(360, size)
        x0 = (width - size) / 2
        y0 = (height - size) / 2
        cell = size / 3


        self.board_canvas.create_rectangle(
            x0 - 12, y0 - 12, x0 + size + 12, y0 + size + 12,
            fill="#f8fbff", outline=self.BORDER, width=2
        )

        for i in range(4):
            self.board_canvas.create_line(x0 + i * cell, y0, x0 + i * cell, y0 + size, fill="#c6d3eb", width=3)
            self.board_canvas.create_line(x0, y0 + i * cell, x0 + size, y0 + i * cell, fill="#c6d3eb", width=3)

        b = self.game.board
        for r in range(3):
            for c in range(3):
                cx = x0 + c * cell + cell / 2
                cy = y0 + r * cell + cell / 2
                v = int(b[r, c])
                if v == 1:
                    self._draw_x(cx, cy, cell * 0.32, "#1d4ed8")
                elif v == 2:
                    self._draw_o(cx, cy, cell * 0.34, "#f59e0b")

        self._ttt_geom = (x0, y0, cell, size)

    def _draw_x(self, cx: float, cy: float, r: float, color: str) -> None:
        self.board_canvas.create_line(cx - r, cy - r, cx + r, cy + r, fill=color, width=8, capstyle="round")
        self.board_canvas.create_line(cx - r, cy + r, cx + r, cy - r, fill=color, width=8, capstyle="round")

    def _draw_o(self, cx: float, cy: float, r: float, color: str) -> None:
        self.board_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=8)

    def _draw_c4_board(self, hover_col: int | None = None) -> None:
        width = self.board_canvas.winfo_width() or 840
        height = self.board_canvas.winfo_height() or 720
        rows, cols = self.game.rows, self.game.cols

        cell = min((width - 120) / cols, (height - 130) / rows)
        cell = max(68, min(96, cell))
        board_w = cell * cols
        board_h = cell * rows
        x0 = (width - board_w) / 2
        y0 = (height - board_h) / 2 + 10


        if hover_col is not None and hover_col in self.game.legal_moves() and self.game.current_player == self.human_player and not self.game_over:
            hx = x0 + hover_col * cell
            self.board_canvas.create_rectangle(hx, y0 - 36, hx + cell, y0 + board_h + 4, fill="#eaf2ff", outline="")

        self.board_canvas.create_rectangle(
            x0 - 14, y0 - 14, x0 + board_w + 14, y0 + board_h + 14,
            fill="#eff5ff", outline=self.BORDER, width=2
        )
        self.board_canvas.create_rectangle(
            x0, y0, x0 + board_w, y0 + board_h,
            fill="#2f6fed", outline="#255ed3", width=2
        )

        b = self.game.board
        for r in range(rows):
            for c in range(cols):
                px0 = x0 + c * cell + 8
                py0 = y0 + r * cell + 8
                px1 = px0 + cell - 16
                py1 = py0 + cell - 16
                v = int(b[r, c])
                if v == 0:
                    fill = "#f8fbff"
                    outline = "#d9e5fb"
                elif v == 1:
                    fill = "#ef4444"
                    outline = "#dc2626"
                else:
                    fill = "#facc15"
                    outline = "#eab308"
                self.board_canvas.create_oval(px0, py0, px1, py1, fill=fill, outline=outline, width=2)


        for c in range(cols):
            cx = x0 + c * cell + cell / 2
            self.board_canvas.create_text(cx, y0 - 18, text=str(c + 1), fill=self.MUTED, font=("Avenir Next", 10, "bold"))

        self._c4_geom = (x0, y0, cell, rows, cols)


    def _on_canvas_motion(self, event) -> None:
        if self.game_key != "c4" or self.game is None:
            return
        if self.game_over or self.game.current_player != self.human_player:
            return
        col = self._event_to_c4_col(event.x, event.y)
        self._draw_c4_board(hover_col=col if col is not None else None)

    def _on_canvas_leave(self, _event) -> None:
        if self.game_key == "c4":
            self._draw_board()

    def _on_canvas_click(self, event) -> None:
        if self.game is None or self.game_over:
            return
        if self.game.current_player != self.human_player:
            return

        move = None
        if self.game_key == "ttt":
            move = self._event_to_ttt_move(event.x, event.y)
        else:
            move = self._event_to_c4_col(event.x, event.y)

        if move is None:
            return
        if move not in self.game.legal_moves():
            return

        elapsed_ms = (time.perf_counter() - self.turn_started_ts) * 1000.0 if self.turn_started_ts else 0.0
        self.session_stats.human_move_times.append(elapsed_ms)
        self._apply_move(move, actor="human", elapsed_ms=elapsed_ms)

    def _event_to_ttt_move(self, x: float, y: float):
        if not hasattr(self, "_ttt_geom"):
            return None
        x0, y0, cell, size = self._ttt_geom
        if not (x0 <= x <= x0 + size and y0 <= y <= y0 + size):
            return None
        c = int((x - x0) // cell)
        r = int((y - y0) // cell)
        if 0 <= r < 3 and 0 <= c < 3:
            return (r, c)
        return None

    def _event_to_c4_col(self, x: float, y: float):
        if not hasattr(self, "_c4_geom"):
            return None
        x0, y0, cell, rows, cols = self._c4_geom
        if not (x0 <= x <= x0 + cell * cols and y0 - 40 <= y <= y0 + cell * rows):
            return None
        c = int((x - x0) // cell)
        if 0 <= c < cols:
            return c
        return None

    def _queue_ai_turn(self) -> None:
        if self.game_over or self.ai_agent is None or self.game is None:
            return

        self.status_var.set("Model thinking...")
        self.pending_ai_token = self.active_game_token
        g_clone = self.game.clone()
        turn_index = self.move_index
        self.ai_future = self.executor.submit(self._compute_ai_move_safe, g_clone, self.ai_agent, turn_index)
        self.after(40, self._poll_ai_future)

    def _compute_ai_move_safe(self, game_clone, agent, turn_index: int):
        t0 = time.perf_counter()
        with self.agent_lock:
            agent.reset()
            move = agent.select_action(game_clone, training=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return turn_index, move, dt_ms

    def _poll_ai_future(self) -> None:
        if self.ai_future is None:
            return
        if not self.ai_future.done():
            self.after(40, self._poll_ai_future)
            return

        try:
            _, move, dt_ms = self.ai_future.result()
        except Exception as exc:
            self.status_var.set("AI error")
            self.log_line(f"AI error: {exc}")
            return

        if self.pending_ai_token != self.active_game_token:
            return
        if self.game is None or self.game_over:
            return
        if move not in self.game.legal_moves():
            self.status_var.set("AI produced illegal move")
            self.log_line(f"AI illegal move ignored: {move}")
            return

        self.session_stats.ai_move_times.append(dt_ms)
        self._apply_move(move, actor="ai", elapsed_ms=dt_ms)

    def _apply_move(self, move, actor: str, elapsed_ms: float) -> None:
        if self.game is None:
            return
        player = self.game.current_player
        self.game.apply_move(move)
        self.move_index += 1

        move_txt = self._format_move(move)
        who = "You" if actor == "human" else "Model"
        self.last_move_text = f"{who} (P{player}) -> {move_txt}"
        self.last_move_var.set(f"Last move: {self.last_move_text}")
        self.moves_var.set(f"Moves: {self.move_index}")

        ms = f"{elapsed_ms:.1f}ms"
        self.log_line(f"{who:<6} P{player} moved {move_txt:<8} ({ms})")

        self._draw_board()

        if self.game.is_terminal():
            self._finish_game()
            return

        self._refresh_turn_text()
        if self.game.current_player == self.human_player:
            self.status_var.set("Your turn")
            self.turn_started_ts = time.perf_counter()
        else:
            self._queue_ai_turn()

    def _finish_game(self) -> None:
        if self.game is None:
            return
        self.game_over = True
        winner = self.game.winner()
        self.session_stats.record_game(winner=winner, human_player=self.human_player)
        self._update_session_panel()

        if winner == 0:
            msg = "Draw"
            color = self.DRAW
        elif winner == self.human_player:
            msg = "You win"
            color = self.WIN
        else:
            msg = "Model wins"
            color = self.LOSE

        self.result_var.set(f"Result: {msg}")
        self.status_var.set("Game complete")
        self.log_line(f"Game finished -> {msg}")


        self.status_chip.configure(fg=color)
        self.after(1100, lambda: self.status_chip.configure(fg=self.ACCENT))


    def _refresh_turn_text(self) -> None:
        if self.game is None:
            self.turn_var.set("Turn: -")
            return
        if self.game.current_player == self.human_player:
            self.turn_var.set("Turn: You")
        else:
            self.turn_var.set("Turn: Model")

    def _update_session_panel(self) -> None:
        s = self.session_stats
        self.session_games_var.set(str(s.games))
        self.session_human_var.set(f"{100*s._rate(s.human_wins, s.games):.1f}%")
        self.session_ai_var.set(f"{100*s._rate(s.ai_wins, s.games):.1f}%")
        self.session_draw_var.set(f"{100*s._rate(s.draws, s.games):.1f}%")
        p1_wr_h = s._rate(s.human_as_p1_wins, s.human_as_p1_games)
        p2_wr_h = s._rate(s.human_as_p2_wins, s.human_as_p2_games)
        self.session_p1_var.set(f"{100*p1_wr_h:.1f}%")
        self.session_p2_var.set(f"{100*p2_wr_h:.1f}%")


        if s.human_as_p1_games > 0 and s.human_as_p2_games > 0:

            model_as_p2_wr = s._rate(s.human_as_p1_losses, s.human_as_p1_games)
            model_as_p1_wr = s._rate(s.human_as_p2_losses, s.human_as_p2_games)

            draw_as_p1 = max(0.0, 1.0 - p1_wr_h - model_as_p2_wr)
            draw_as_p2 = max(0.0, 1.0 - p2_wr_h - model_as_p1_wr)

            fair_human = 0.5 * (p1_wr_h + p2_wr_h)
            fair_ai = 0.5 * (model_as_p1_wr + model_as_p2_wr)
            fair_draw = 0.5 * (draw_as_p1 + draw_as_p2)

            self.session_fair_human_var.set(f"{100*fair_human:.1f}%")
            self.session_fair_ai_var.set(f"{100*fair_ai:.1f}%")
            self.session_fair_draw_var.set(f"{100*fair_draw:.1f}%")
        else:
            self.session_fair_human_var.set("Need both sides")
            self.session_fair_ai_var.set("Need both sides")
            self.session_fair_draw_var.set("Need both sides")

        self.session_balance_var.set(f"P1:{s.human_as_p1_games} | P2:{s.human_as_p2_games}")
        self.session_human_time_var.set(f"{s._mean(s.human_move_times):.1f} ms")
        self.session_ai_time_var.set(f"{s._mean(s.ai_move_times):.1f} ms")


    def run_quick_eval(self) -> None:
        if self.eval_future and not self.eval_future.done():
            messagebox.showinfo("Quick Eval", "A quick evaluation is already running.")
            return

        spec = self._get_selected_spec()
        if spec is None:
            return
        n_games = max(10, int(self.quick_eval_games.get()))
        self.quick_eval_var.set(f"Running {n_games} games...")
        self.log_line(f"Quick eval started for {spec.label} ({n_games} games vs Default).")

        self.eval_future = self.executor.submit(self._quick_eval_worker, spec, n_games)
        self.after(90, self._poll_eval_future)

    def _quick_eval_worker(self, spec: AgentSpec, n_games: int) -> dict:
        agent = spec.factory()
        _set_greedy(agent)
        if spec.game_key == "ttt":
            result = evaluate_agent(agent, TicTacToe, n_games=n_games, seed=42)
        else:
            result = evaluate_agent(agent, Connect4, game_kwargs={"rows": 6, "cols": 7}, n_games=n_games, seed=42)
        return result

    def _poll_eval_future(self) -> None:
        if self.eval_future is None:
            return
        if not self.eval_future.done():
            self.after(90, self._poll_eval_future)
            return

        try:
            r = self.eval_future.result()
        except Exception as exc:
            self.quick_eval_var.set(f"Quick eval failed: {exc}")
            self.log_line(f"Quick eval failed: {exc}")
            return

        summary = (
            f"{r['agent']} vs {r['opponent']}\n"
            f"WR: {100*r['total_win_rate']:.1f}% | Draw: {100*r['total_draw_rate']:.1f}%\n"
            f"P1 WR: {100*r['p1_win_rate']:.1f}% | P2 WR: {100*r['p2_win_rate']:.1f}%\n"
            f"Avg move time: {r['avg_agent_time_ms_per_move']:.2f} ms"
        )
        self.quick_eval_var.set(summary)
        self.log_line(f"Quick eval done -> WR {100*r['total_win_rate']:.1f}%, Draw {100*r['total_draw_rate']:.1f}%")


    def log_line(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log_text_clear(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


    def _format_move(self, move) -> str:
        if self.game_key == "ttt":
            r, c = move
            return f"({r+1},{c+1})"
        return f"col {int(move)+1}"


def main() -> None:
    app = GameArenaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
