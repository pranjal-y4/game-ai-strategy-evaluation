"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
import sys, os, csv, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.advanced_alphabeta_c4 import AdvancedAlphaBetaC4Agent
from evaluation.evaluator import evaluate_agent, crossplay, _set_greedy
from utils.seed import set_seed

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False


# I have implemented this callable with parameters: args.
def _build_ttt_agents(args) -> list:


    agents = [
        RandomAgent(),
        DefaultAgent(),
        MinimaxAgent(max_depth=20),
        AlphaBetaAgent(max_depth=20),
    ]


    try:
        from rl.q_learning import AdvancedQLearning
        from rl.game_env import GameEnv
        from agents.base_agent import BaseAgent


        class _QAgent(BaseAgent):


            # I have implemented this callable with parameters: self, path, name.
            def __init__(self, path, name="QL_TTT"):


                """
                I have implemented this function with a clearer note.
                Parameters used here: self, path, name.
                """
                self._name = name
                self.epsilon = 0.0
                env = GameEnv(TicTacToe, opponent="random")
                self._q = AdvancedQLearning(env)
                self._q.load(path)
                self._q.epsilon = 0.0

            @property


            # I have implemented this callable with parameters: self.
            def name(self):


                return self._name


            # I have implemented this callable with parameters: self, game, training.
            def select_action(self, game, training=False):


                """
                I have implemented this function with a clearer note.
                Parameters used here: self, game, training.
                """
                state = game.encode_state_hashable(
                    perspective_player=game.current_player)
                legal = [r * 3 + c for (r, c) in game.legal_moves()]
                action = self._q.predict(state, legal)
                return (action // 3, action % 3)

        model_path = args.ttt_ql_model or os.path.join(
            os.path.dirname(__file__), "..", "models", "ttt_qlearning_best.pkl")
        if os.path.exists(model_path):
            agents.append(_QAgent(model_path, "QL_TTT"))
            print(f"  Loaded TTT Q-learning from {model_path}")
    except Exception as e:
        print(f"  [WARN] Could not load TTT Q-learning: {e}")


    try:
        import torch
        from rl.dqn import AdvancedDQNAgent
        from rl.game_env import GameEnv
        from agents.base_agent import BaseAgent


        class _DQNAgent(BaseAgent):


            # I have implemented this callable with parameters: self, path, name.
            def __init__(self, path, name="DQN_TTT"):


                self._name = name
                self.epsilon = 0.0
                env = GameEnv(TicTacToe, opponent="random")
                self._dqn = AdvancedDQNAgent(env, hidden=[128, 64])
                self._dqn.load(path)
                self._dqn.epsilon = 0.0

            @property


            # I have implemented this callable with parameters: self.
            def name(self):


                """
                I have implemented this function with a clearer note.
                Parameters used here: self.
                """
                return self._name


            # I have implemented this callable with parameters: self, game, training.
            def select_action(self, game, training=False):


                state = game.encode_state(perspective_player=game.current_player)
                legal = [r * 3 + c for (r, c) in game.legal_moves()]
                action = self._dqn._greedy(state, legal)
                return (action // 3, action % 3)

        model_path = args.ttt_dqn_model or os.path.join(
            os.path.dirname(__file__), "..", "models", "ttt_dqn_best.pt")
        if os.path.exists(model_path):
            agents.append(_DQNAgent(model_path, "DQN_TTT"))
            print(f"  Loaded TTT DQN from {model_path}")
    except Exception as e:
        print(f"  [WARN] Could not load TTT DQN: {e}")

    return agents


# I have implemented this callable with parameters: args.
def _build_c4_agents(args) -> list:


    """
    I have implemented this function with a clearer note.
    Parameters used here: args.
    """
    agents = [
        RandomAgent(),
        DefaultAgent(),
        AlphaBetaAgent(max_depth=5),
        AdvancedAlphaBetaC4Agent(max_depth=args.ab_depth,
                                 time_budget=args.ab_time),
    ]


    try:
        import torch
        from rl.dqn import AdvancedDQNAgent
        from rl.game_env import GameEnv
        from agents.base_agent import BaseAgent


        class _DQNC4Agent(BaseAgent):


            # I have implemented this callable with parameters: self, path, rows, cols, name.
            def __init__(self, path, rows=6, cols=7, name="DQN_C4"):


                self._name = name
                self.epsilon = 0.0
                self._rows = rows
                self._cols = cols
                env = GameEnv(Connect4,
                              game_kwargs={"rows": rows, "cols": cols},
                              opponent="random")
                self._dqn = AdvancedDQNAgent(env, hidden=[256, 128])
                self._dqn.load(path)
                self._dqn.epsilon = 0.0

            @property


            # I have implemented this callable with parameters: self.
            def name(self):


                """
                I have implemented this function with a clearer note.
                Parameters used here: self.
                """
                return self._name


            # I have implemented this callable with parameters: self, game, training.
            def select_action(self, game, training=False):


                if game.rows != self._rows or game.cols != self._cols:
                    return game.legal_moves()[0]
                state = game.encode_state(perspective_player=game.current_player)
                legal = game.legal_moves()
                return self._dqn._greedy(state, legal)

        model_path = args.c4_dqn_model or os.path.join(
            os.path.dirname(__file__), "..", "models", "c4_dqn_6x7_best.pt")
        if os.path.exists(model_path):
            agents.append(_DQNC4Agent(model_path, name="DQN_C4"))
            print(f"  Loaded C4 DQN from {model_path}")
    except Exception as e:
        print(f"  [WARN] Could not load C4 DQN: {e}")

    return agents


# I have implemented this callable with parameters: agents, game_cls, game_kwargs, n_games, label, out_dir.
def run_vs_default(agents, game_cls, game_kwargs, n_games, label, out_dir):


    """
    I have implemented this function with a clearer note.
    Parameters used here: agents, game_cls, game_kwargs, n_games, label, out_dir.
    """
    default = DefaultAgent()
    rows = []
    print(f"\n{'='*60}")
    print(f"  {label} — All agents vs Default ({n_games} games each)")
    print(f"{'='*60}")
    for agent in agents:
        _set_greedy(agent)
        r = evaluate_agent(agent, game_cls, game_kwargs,
                           opponent=default, n_games=n_games)
        rows.append(r)
        print(f"  {r['agent']:30s} | wr={r['total_win_rate']:.3f} "
              f"| p1={r['p1_win_rate']:.3f} | p2={r['p2_win_rate']:.3f} "
              f"| fma={r['first_mover_advantage']:+.3f} "
              f"| t={r['avg_agent_time_ms_per_move']:.2f}ms/move")


    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f"vs_default_{label.lower()}_{ts}.csv")
    if rows:
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV saved: {fname}")
    return rows


# I have implemented this callable with parameters: agents, game_cls, game_kwargs, n_games, label, out_dir.
def run_crossplay(agents, game_cls, game_kwargs, n_games, label, out_dir):


    print(f"\n{'='*60}")
    print(f"  {label} — Round-robin crossplay ({n_games} games/pair)")
    print(f"{'='*60}")
    rows = crossplay(agents, game_cls, game_kwargs, n_games=n_games)
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f"crossplay_{label.lower()}_{ts}.csv")
    if rows:
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV saved: {fname}")


    agent_names = list(dict.fromkeys([r["agent"] for r in rows]))
    print(f"\n  Win rate matrix (row=agent, col=opponent):")
    header = f"{'':30s}" + "".join(f"{n[:12]:>14s}" for n in agent_names)
    print("  " + header)
    for a in agent_names:
        row_s = f"{a[:30]:30s}"
        for b in agent_names:
            if a == b:
                row_s += f"{'—':>14s}"
            else:
                match = [r for r in rows if r["agent"] == a and r["opponent"] == b]
                row_s += f"{match[0]['win_rate']:>14.3f}" if match else f"{'?':>14s}"
        print("  " + row_s)
    return rows


# I have implemented this callable with parameters: rows, label, out_dir.
def plot_vs_default(rows, label, out_dir):


    """
    I have implemented this function with a clearer note.
    Parameters used here: rows, label, out_dir.
    """
    if not _PLOT or not rows:
        return
    agents = [r["agent"] for r in rows]
    win_r = [r["total_win_rate"] for r in rows]
    draw_r = [r["total_draw_rate"] for r in rows]
    loss_r = [r["total_loss_rate"] for r in rows]
    x = np.arange(len(agents))
    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.5), 5))
    ax.bar(x, win_r, label="Win", color="steelblue")
    ax.bar(x, draw_r, bottom=win_r, label="Draw", color="gold")
    ax.bar(x, loss_r,
           bottom=[w + d for w, d in zip(win_r, draw_r)],
           label="Loss", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(f"{label} — vs Default Opponent")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"vs_default_{label.lower()}_outcomes.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved: {fname}")


# I have implemented this callable with parameters: args.
def main(args):


    set_seed(args.seed)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "plots")

    if args.game in ("ttt", "both"):
        print("\n" + "="*60)
        print("  Evaluating TicTacToe agents")
        print("="*60)
        ttt_agents = _build_ttt_agents(args)
        vd = run_vs_default(ttt_agents, TicTacToe, {}, args.n_games, "TTT", out_dir)
        plot_vs_default(vd, "TTT", plot_dir)
        if len(ttt_agents) > 1:
            run_crossplay(ttt_agents, TicTacToe, {}, args.n_games // 2, "TTT", out_dir)

    if args.game in ("c4", "both"):
        print("\n" + "="*60)
        print("  Evaluating Connect4 agents (6×7)")
        print("="*60)
        c4_agents = _build_c4_agents(args)
        vd = run_vs_default(c4_agents, Connect4, {"rows": 6, "cols": 7},
                            args.n_games, "C4", out_dir)
        plot_vs_default(vd, "C4", plot_dir)
        if len(c4_agents) > 1:
            run_crossplay(c4_agents, Connect4, {"rows": 6, "cols": 7},
                          args.n_games // 2, "C4", out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["ttt", "c4", "both"], default="both")
    p.add_argument("--n_games", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ab_depth", type=int, default=8,
                   help="Depth for AdvancedAlphaBeta agent")
    p.add_argument("--ab_time", type=float, default=None,
                   help="Time budget per move for AdvancedAlphaBeta (seconds)")
    p.add_argument("--ttt_ql_model", type=str, default=None)
    p.add_argument("--ttt_dqn_model", type=str, default=None)
    p.add_argument("--c4_dqn_model", type=str, default=None)
    args = p.parse_args()
    main(args)
