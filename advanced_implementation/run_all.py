from __future__ import annotations
import sys, os, argparse, time, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.advanced_alphabeta_c4 import AdvancedAlphaBetaC4Agent
from rl.game_env import GameEnv
from rl.q_learning import AdvancedQLearning
from evaluation.evaluator import evaluate_agent, crossplay, _set_greedy
from utils.seed import set_seed

try:
    import torch
    from rl.dqn import AdvancedDQNAgent
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib not available — graphs will be skipped")


MODEL_DIR = os.path.join(ROOT, "models")
LOG_DIR   = os.path.join(ROOT, "logs")
GRAPH_DIR = os.path.join(LOG_DIR, "graphs")
for d in [MODEL_DIR, LOG_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

TS = time.strftime("%Y%m%d_%H%M%S")


def train_ttt_qlearning(episodes, curriculum_frac, seed, n_step=3):

    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"  [1/4] TTT Q-Learning | episodes={episodes} | seed={seed}")
    print(f"        Curriculum: {int(episodes*curriculum_frac)} random → "
          f"{episodes - int(episodes*curriculum_frac)} default")
    print(f"{'─'*60}")

    train_env = GameEnv(TicTacToe, opponent="random")
    train_env.enable_role_alternation()

    agent = AdvancedQLearning(
        train_env, lr=0.15, gamma=0.95,
        epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
        n_step=n_step,
    )

    curriculum_switch = int(episodes * curriculum_frac)
    phase = "phase1_random"
    csv_rows = []
    reward_window = []

    for ep in range(1, episodes + 1):
        if ep == curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.epsilon = max(agent.epsilon, 0.2)
            print(f"  ★ Curriculum switch → default at ep {ep}")

        state_arr = train_env.reset()
        state = train_env.encode_state()
        total_r = 0.0
        ep_trans = []

        while True:
            legal = train_env.get_legal_actions()
            action = agent.choose_action(state, legal)
            _, reward, done, _ = train_env.step(action)
            ns = train_env.encode_state()
            nl = train_env.get_legal_actions()
            ep_trans.append((state, action, reward, ns, done, nl))
            total_r += reward; state = ns
            if done: break

        T = len(ep_trans)
        for t in range(T):
            s_t, a_t = ep_trans[t][0], ep_trans[t][1]
            G, actual_n = 0.0, 0
            for i in range(n_step):
                if t + i >= T: break
                G += (0.95**i) * ep_trans[t+i][2]
                actual_n = i + 1
                if ep_trans[t+i][4]: break
            idx_n = t + actual_n - 1
            ns_n, done_n, ln = ep_trans[idx_n][3], ep_trans[idx_n][4], ep_trans[idx_n][5]
            agent.update(s_t, a_t, G, ns_n, done_n, ln, 0.95**actual_n)

        agent.decay_epsilon()
        agent.episode_rewards.append(total_r)
        reward_window.append(total_r)

        eval_every = max(1, episodes // 30)
        if ep % eval_every == 0:
            wr = agent.evaluate(200)
            agent.win_rates.append(wr)
            sm = float(np.mean(reward_window[-200:])) if reward_window else 0.0
            csv_rows.append({
                "episode": ep, "win_rate": round(wr, 4),
                "epsilon": round(agent.epsilon, 4),
                "q_states": len(agent.q_table), "phase": phase,
                "smoothed_reward": round(sm, 4),
                "p1_win_rate": round(wr, 4),
            })
            print(f"  ep={ep:6d} | ε={agent.epsilon:.3f} | wr={wr:.3f} "
                  f"| states={len(agent.q_table)} | {phase}")

    path = os.path.join(MODEL_DIR, "ttt_qlearning_default.pkl")
    agent.save(path)
    _save_csv(csv_rows, os.path.join(LOG_DIR, f"train_ttt_ql_{TS}.csv"))
    print(f"  ✓ Saved: {path}")
    return agent, csv_rows


def train_c4_qlearning(episodes, curriculum_frac, seed, rows=6, cols=7, n_step=3):

    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"  [2/4] C4 Q-Learning {rows}×{cols} | episodes={episodes} | seed={seed}")
    print(f"{'─'*60}")

    train_env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                        opponent="random")
    train_env.enable_role_alternation()

    agent = AdvancedQLearning(
        train_env, lr=0.1, gamma=0.95,
        epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.05,
        n_step=n_step,
    )

    curriculum_switch = int(episodes * curriculum_frac)
    phase = "phase1_random"
    csv_rows = []
    reward_window = []

    for ep in range(1, episodes + 1):
        if ep == curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.epsilon = max(agent.epsilon, 0.25)
            print(f"  ★ Curriculum switch → default at ep {ep}")

        state_arr = train_env.reset()
        state = train_env.encode_state()
        total_r = 0.0
        ep_trans = []

        while True:
            legal = train_env.get_legal_actions()
            action = agent.choose_action(state, legal)
            _, reward, done, _ = train_env.step(action)
            ns = train_env.encode_state()
            nl = train_env.get_legal_actions()
            ep_trans.append((state, action, reward, ns, done, nl))
            total_r += reward; state = ns
            if done: break

        T = len(ep_trans)
        for t in range(T):
            s_t, a_t = ep_trans[t][0], ep_trans[t][1]
            G, actual_n = 0.0, 0
            for i in range(n_step):
                if t + i >= T: break
                G += (0.95**i) * ep_trans[t+i][2]
                actual_n = i + 1
                if ep_trans[t+i][4]: break
            idx_n = t + actual_n - 1
            ns_n, done_n, ln = ep_trans[idx_n][3], ep_trans[idx_n][4], ep_trans[idx_n][5]
            agent.update(s_t, a_t, G, ns_n, done_n, ln, 0.95**actual_n)

        agent.decay_epsilon()
        agent.episode_rewards.append(total_r)
        reward_window.append(total_r)

        eval_every = max(1, episodes // 30)
        if ep % eval_every == 0:
            wr = agent.evaluate(200)
            agent.win_rates.append(wr)
            sm = float(np.mean(reward_window[-200:])) if reward_window else 0.0
            csv_rows.append({
                "episode": ep, "win_rate": round(wr, 4),
                "epsilon": round(agent.epsilon, 4),
                "q_states": len(agent.q_table), "phase": phase,
                "smoothed_reward": round(sm, 4), "board": f"{rows}x{cols}",
            })
            print(f"  ep={ep:6d} | ε={agent.epsilon:.3f} | wr={wr:.3f} "
                  f"| states={len(agent.q_table)} | {phase}")

    path = os.path.join(MODEL_DIR, f"c4_qlearning_{rows}x{cols}_default.pkl")
    agent.save(path)
    _save_csv(csv_rows, os.path.join(LOG_DIR, f"train_c4_ql_{rows}x{cols}_{TS}.csv"))
    print(f"  ✓ Saved: {path}  [NOTE: {rows}×{cols} board only]")
    return agent, csv_rows


def train_ttt_dqn(episodes, curriculum_frac, seed):

    if not HAS_TORCH:
        print("\n  [3/4] TTT DQN — SKIPPED (PyTorch not available)")
        return None, []

    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"  [3/4] TTT DQN | episodes={episodes} | seed={seed}")
    print(f"{'─'*60}")

    train_env = GameEnv(TicTacToe, opponent="random")
    train_env.enable_role_alternation()

    agent = AdvancedDQNAgent(
        train_env, hidden=[128, 64], lr=5e-4, gamma=0.95,
        epsilon_start=1.0, epsilon_min=0.05,
        decay_steps=int(episodes * 0.7),
        batch_size=64, buffer_size=20_000,
        target_update=200, n_step=3, use_per=True,
    )

    curriculum_switch = int(episodes * curriculum_frac)
    phase = "phase1_random"
    csv_rows = []

    def callback(ep, wr, eps, avg_loss):
        nonlocal phase
        if ep >= curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.set_epsilon(max(eps, 0.2))
            print(f"  ★ Curriculum switch → default at ep {ep}")
        sm = (float(np.mean(agent.episode_rewards[-200:]))
              if len(agent.episode_rewards) >= 200 else 0.0)
        csv_rows.append({
            "episode": ep, "win_rate": round(wr, 4),
            "epsilon": round(eps, 4), "avg_loss": round(avg_loss, 6),
            "phase": phase, "smoothed_reward": round(sm, 4),
        })
        print(f"  ep={ep:6d} | ε={eps:.3f} | wr={wr:.3f} "
              f"| loss={avg_loss:.4f} | {phase}")

    agent.train(
        n_episodes=episodes,
        eval_every=max(1, episodes // 20),
        eval_episodes=200,
        reward_shaping=False,
        callback=callback,
    )

    path = os.path.join(MODEL_DIR, "ttt_dqn_default.pt")
    agent.save(path)
    _save_csv(csv_rows, os.path.join(LOG_DIR, f"train_ttt_dqn_{TS}.csv"))
    print(f"  ✓ Saved: {path}")
    return agent, csv_rows


def train_c4_dqn(episodes, curriculum_frac, seed):

    if not HAS_TORCH:
        print("\n  [4/4] C4 DQN — SKIPPED (PyTorch not available)")
        return None, []

    set_seed(seed)
    print(f"\n{'─'*60}")
    print(f"  [4/4] C4 DQN 6×7 | episodes={episodes} | seed={seed}")
    print(f"        Reward shaping: ON | Double DQN + PER + n-step(3)")
    print(f"{'─'*60}")

    train_env = GameEnv(Connect4, game_kwargs={"rows": 6, "cols": 7},
                        opponent="random")
    train_env.enable_role_alternation()

    agent = AdvancedDQNAgent(
        train_env, hidden=[256, 128], lr=5e-4, gamma=0.99,
        epsilon_start=1.0, epsilon_min=0.05,
        decay_steps=int(episodes * 0.65),
        batch_size=64, buffer_size=50_000,
        target_update=500, n_step=3,
        per_alpha=0.6, per_beta_start=0.4,
        per_beta_steps=episodes * 8,
        use_per=True, grad_clip=10.0,
    )

    curriculum_switch = int(episodes * curriculum_frac)
    phase = "phase1_random"
    csv_rows = []


    print(f"  Phase 1: vs Random (0 → {curriculum_switch} eps)")
    def cb1(ep, wr, eps, avg_loss):
        sm = (float(np.mean(agent.episode_rewards[-200:]))
              if len(agent.episode_rewards) >= 200 else 0.0)
        csv_rows.append({
            "episode": ep, "win_rate": round(wr, 4),
            "epsilon": round(eps, 4), "avg_loss": round(avg_loss, 6),
            "phase": "phase1_random", "smoothed_reward": round(sm, 4),
        })
        print(f"  ep={ep:6d} | ε={eps:.3f} | wr={wr:.3f} "
              f"| loss={avg_loss:.4f} | phase1_random")

    agent.train(
        n_episodes=curriculum_switch,
        eval_every=max(1, curriculum_switch // 10),
        eval_episodes=100,
        reward_shaping=True, shaping_weight=0.01, shaping_clip=0.1,
        callback=cb1,
    )

    agent.save(os.path.join(MODEL_DIR, "c4_dqn_phase1.pt"))


    print(f"\n  Phase 2: vs Default ({curriculum_switch} → {episodes} eps)")
    train_env._opponent_type = "default"
    new_eps = max(agent.epsilon, 0.3)
    agent.set_epsilon(new_eps)
    print(f"  ★ Epsilon reset to {new_eps:.3f}")

    def cb2(ep, wr, eps, avg_loss):
        sm = (float(np.mean(agent.episode_rewards[-200:]))
              if len(agent.episode_rewards) >= 200 else 0.0)
        csv_rows.append({
            "episode": curriculum_switch + ep, "win_rate": round(wr, 4),
            "epsilon": round(eps, 4), "avg_loss": round(avg_loss, 6),
            "phase": "phase2_default", "smoothed_reward": round(sm, 4),
        })
        print(f"  ep={curriculum_switch+ep:6d} | ε={eps:.3f} | wr={wr:.3f} "
              f"| loss={avg_loss:.4f} | phase2_default")

    agent.train(
        n_episodes=episodes - curriculum_switch,
        eval_every=max(1, (episodes - curriculum_switch) // 10),
        eval_episodes=100,
        reward_shaping=True, shaping_weight=0.01, shaping_clip=0.1,
        callback=cb2,
    )

    path = os.path.join(MODEL_DIR, "c4_dqn_6x7_default.pt")
    agent.save(path)
    _save_csv(csv_rows, os.path.join(LOG_DIR, f"train_c4_dqn_{TS}.csv"))
    print(f"  ✓ Saved: {path}")
    return agent, csv_rows


def build_rl_agent_from_model(model_type, model_path, game_cls, game_kwargs=None):

    from agents.base_agent import BaseAgent
    gkw = game_kwargs or {}

    if model_type == "ql":
        class _QLAgent(BaseAgent):
            def __init__(self, path, name, gc, gkw):
                self._name = name
                self.epsilon = 0.0
                env = GameEnv(gc, game_kwargs=gkw, opponent="random")
                self._q = AdvancedQLearning(env)
                self._q.load(path)
                self._q.epsilon = 0.0
                self._is_ttt = (gc == TicTacToe)

            @property
            def name(self): return self._name

            def select_action(self, game, training=False):
                p = game.current_player
                state = game.encode_state_hashable(perspective_player=p)
                if self._is_ttt:
                    legal_flat = [r * 3 + c for (r, c) in game.legal_moves()]
                    action = self._q.predict(state, legal_flat)
                    return (action // 3, action % 3)
                else:
                    legal = game.legal_moves()
                    return self._q.predict(state, legal)

        return _QLAgent(model_path,
                        f"QL_{'TTT' if game_cls==TicTacToe else 'C4'}",
                        game_cls, gkw)

    elif model_type == "dqn" and HAS_TORCH:
        class _DQNAgent(BaseAgent):
            def __init__(self, path, name, gc, gkw, hidden):
                self._name = name
                self.epsilon = 0.0
                env = GameEnv(gc, game_kwargs=gkw, opponent="random")
                self._dqn = AdvancedDQNAgent(env, hidden=hidden)
                self._dqn.load(path)
                self._dqn.epsilon = 0.0
                self._is_ttt = (gc == TicTacToe)

            @property
            def name(self): return self._name

            def select_action(self, game, training=False):
                p = game.current_player
                state = game.encode_state(perspective_player=p)
                if self._is_ttt:
                    legal = [r * 3 + c for (r, c) in game.legal_moves()]
                    action = self._dqn._greedy(state, legal)
                    return (action // 3, action % 3)
                else:
                    legal = game.legal_moves()
                    return self._dqn._greedy(state, legal)

        hidden = [128, 64] if game_cls == TicTacToe else [256, 128]
        return _DQNAgent(model_path,
                         f"DQN_{'TTT' if game_cls==TicTacToe else 'C4'}",
                         game_cls, gkw, hidden)
    return None


def collect_nodes_stats(agents_info, game_cls, game_kwargs, n_games=50, seed=42):


    import time as _time
    np.random.seed(seed)
    gkw = game_kwargs or {}
    results = []

    for agent, label in agents_info:
        nodes_per_move = []
        time_per_move  = []
        wins = 0
        opp = DefaultAgent()

        for gn in range(n_games):
            game = game_cls(**gkw)
            game.reset()
            agent.reset()
            agent_player = 1 if gn % 2 == 0 else 2
            move_count = 0

            while not game.is_terminal():
                if game.current_player == agent_player:
                    agent.reset()
                    t0 = _time.time()
                    move = agent.select_action(game, training=False)
                    dt = _time.time() - t0
                    nodes = getattr(agent, "nodes_expanded", 0) + \
                            getattr(agent, "nodes_searched", 0)
                    nodes_per_move.append(nodes)
                    time_per_move.append(dt * 1000)
                    move_count += 1
                else:
                    move = opp.select_action(game, training=False)
                game.apply_move(move)

            w = game.winner()
            if w == agent_player:
                wins += 1

        results.append({
            "agent": label,
            "avg_nodes_per_move": round(float(np.mean(nodes_per_move)) if nodes_per_move else 0, 1),
            "max_nodes_per_move": int(max(nodes_per_move)) if nodes_per_move else 0,
            "avg_time_ms": round(float(np.mean(time_per_move)) if time_per_move else 0, 2),
            "win_rate": round(wins / n_games, 4),
        })
        print(f"  {label:35s} | nodes/move={results[-1]['avg_nodes_per_move']:>10,.0f} "
              f"| t={results[-1]['avg_time_ms']:6.1f}ms | wr={results[-1]['win_rate']:.3f}")

    return results


def run_evaluation(agents_ttt, agents_c4, n_games, seed):

    set_seed(seed)
    default = DefaultAgent()

    print(f"\n{'═'*60}")
    print(f"  EVALUATION — {n_games} games per agent")
    print(f"{'═'*60}")


    print("\n  TicTacToe — All agents vs Default:")
    ttt_vd = []
    for agent in agents_ttt:
        _set_greedy(agent)
        r = evaluate_agent(agent, TicTacToe, n_games=n_games, seed=seed)
        ttt_vd.append(r)
        print(f"  {r['agent']:30s} | wr={r['total_win_rate']:.3f} "
              f"| p1={r['p1_win_rate']:.3f} | p2={r['p2_win_rate']:.3f} "
              f"| draw={r['total_draw_rate']:.3f} | t={r['avg_agent_time_ms_per_move']:.2f}ms")
    _save_csv(ttt_vd, os.path.join(LOG_DIR, f"eval_ttt_vs_default_{TS}.csv"))


    print("\n  Connect4 — All agents vs Default:")
    c4_vd = []
    for agent in agents_c4:
        _set_greedy(agent)
        r = evaluate_agent(agent, Connect4,
                           game_kwargs={"rows": 6, "cols": 7},
                           n_games=n_games, seed=seed)
        c4_vd.append(r)
        print(f"  {r['agent']:30s} | wr={r['total_win_rate']:.3f} "
              f"| p1={r['p1_win_rate']:.3f} | p2={r['p2_win_rate']:.3f} "
              f"| draw={r['total_draw_rate']:.3f} | t={r['avg_agent_time_ms_per_move']:.2f}ms")
    _save_csv(c4_vd, os.path.join(LOG_DIR, f"eval_c4_vs_default_{TS}.csv"))


    print("\n  TicTacToe — Cross-play:")
    ttt_cp = crossplay(agents_ttt, TicTacToe, n_games=max(20, n_games // 5), seed=seed)
    _save_csv(ttt_cp, os.path.join(LOG_DIR, f"eval_ttt_crossplay_{TS}.csv"))
    print(f"  ({len(ttt_cp)} matchups saved)")


    print("\n  Connect4 — Cross-play:")
    c4_cp = crossplay(agents_c4, Connect4,
                      game_kwargs={"rows": 6, "cols": 7},
                      n_games=max(20, n_games // 5), seed=seed)
    _save_csv(c4_cp, os.path.join(LOG_DIR, f"eval_c4_crossplay_{TS}.csv"))
    print(f"  ({len(c4_cp)} matchups saved)")

    return ttt_vd, c4_vd, ttt_cp, c4_cp


COLORS = {
    "win":   "#2ecc71",   "draw":  "#f1c40f",   "loss":  "#e74c3c",
    "p1":    "#3498db",   "p2":    "#e67e22",
    "phase1":"#95a5a6",   "phase2":"#9b59b6",
    "nodes": "#1abc9c",   "time":  "#e74c3c",
}

def smooth(arr, w=10):
    if len(arr) < w: return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')


def plot_learning_curves(ql_rows, dqn_rows, label, save_path):

    if not HAS_PLOT: return
    has_dqn = bool(dqn_rows)
    ncols = 3 if has_dqn else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    fig.suptitle(f"{label} — Training Curves (trained on Default opponent)",
                 fontsize=13, fontweight="bold")

    def _get_phase_split(rows):
        for i, r in enumerate(rows):
            if r.get("phase") == "phase2_default":
                return rows[i]["episode"]
        return None


    ax = axes[0]
    if ql_rows:
        eps_ql = [r["episode"] for r in ql_rows]
        wr_ql  = [r["win_rate"] for r in ql_rows]
        ax.plot(eps_ql, wr_ql, color="#3498db", linewidth=1.5,
                alpha=0.5, label="Q-Learning (raw)")
        if len(wr_ql) > 4:
            sw = smooth(wr_ql, min(5, len(wr_ql)//3))
            ax.plot(eps_ql[len(eps_ql)-len(sw):], sw,
                    color="#2980b9", linewidth=2.2, label="Q-Learning (smooth)")
        split = _get_phase_split(ql_rows)
        if split:
            ax.axvline(split, color=COLORS["phase2"], linestyle="--",
                       alpha=0.7, label="Curriculum switch")
    if dqn_rows:
        eps_dqn = [r["episode"] for r in dqn_rows]
        wr_dqn  = [r["win_rate"] for r in dqn_rows]
        ax.plot(eps_dqn, wr_dqn, color="#e74c3c", linewidth=1.5,
                alpha=0.5, label="DQN (raw)")
        if len(wr_dqn) > 4:
            sw = smooth(wr_dqn, min(5, len(wr_dqn)//3))
            ax.plot(eps_dqn[len(eps_dqn)-len(sw):], sw,
                    color="#c0392b", linewidth=2.2, label="DQN (smooth)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate vs Default Opponent")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.3)


    ax2 = axes[1]
    if ql_rows:
        eps_vals = [r["epsilon"] for r in ql_rows]
        ax2.plot([r["episode"] for r in ql_rows], eps_vals,
                 color="#3498db", linewidth=2, label="Q-Learning ε")
        split = _get_phase_split(ql_rows)
        if split:
            ax2.axvline(split, color=COLORS["phase2"], linestyle="--", alpha=0.7)
    if dqn_rows:
        eps_vals2 = [r["epsilon"] for r in dqn_rows]
        ax2.plot([r["episode"] for r in dqn_rows], eps_vals2,
                 color="#e74c3c", linewidth=2, label="DQN ε")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Epsilon (ε)")
    ax2.set_title("Exploration Rate Decay")
    ax2.set_ylim(-0.02, 1.05); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)


    if has_dqn:
        ax3 = axes[2]
        loss_vals = [r["avg_loss"] for r in dqn_rows if r.get("avg_loss", 0) > 0]
        eps_loss  = [r["episode"] for r in dqn_rows if r.get("avg_loss", 0) > 0]
        if loss_vals:
            ax3.plot(eps_loss, loss_vals, color="#e67e22", linewidth=1.5, alpha=0.6)
            if len(loss_vals) > 5:
                sl = smooth(loss_vals, min(5, len(loss_vals)//3))
                ax3.plot(eps_loss[len(eps_loss)-len(sl):], sl,
                         color="#d35400", linewidth=2.2, label="Smoothed loss")
        ax3.set_xlabel("Episode"); ax3.set_ylabel("TD Loss")
        ax3.set_title("DQN Training Loss")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_winrate_vs_default(results_list, title, save_path):

    if not HAS_PLOT or not results_list: return
    agents  = [r["agent"] for r in results_list]
    win_r   = [r["total_win_rate"]  for r in results_list]
    draw_r  = [r["total_draw_rate"] for r in results_list]
    loss_r  = [r["total_loss_rate"] for r in results_list]

    x = np.arange(len(agents))
    fig, ax = plt.subplots(figsize=(max(8, len(agents)*1.8), 5))
    bars_w = ax.bar(x, win_r,  label="Win",  color=COLORS["win"],  edgecolor="white")
    bars_d = ax.bar(x, draw_r, label="Draw", color=COLORS["draw"], edgecolor="white",
                    bottom=win_r)
    bars_l = ax.bar(x, loss_r, label="Loss", color=COLORS["loss"], edgecolor="white",
                    bottom=[w+d for w,d in zip(win_r, draw_r)])


    for i, (w, d, l) in enumerate(zip(win_r, draw_r, loss_r)):
        if w > 0.04: ax.text(i, w/2, f"{w:.2f}", ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white")
        if d > 0.04: ax.text(i, w+d/2, f"{d:.2f}", ha="center", va="center",
                              fontsize=8, fontweight="bold")
        if l > 0.04: ax.text(i, w+d+l/2, f"{l:.2f}", ha="center", va="center",
                              fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(x); ax.set_xticklabels(agents, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_p1p2_comparison(results_list, title, save_path):

    if not HAS_PLOT or not results_list: return
    agents = [r["agent"] for r in results_list]
    p1_wr  = [r["p1_win_rate"] for r in results_list]
    p2_wr  = [r["p2_win_rate"] for r in results_list]
    fma    = [r["first_mover_advantage"] for r in results_list]

    x = np.arange(len(agents))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(agents)*2.2), 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")


    ax1.bar(x - w/2, p1_wr, w, label="P1 Win Rate", color=COLORS["p1"], edgecolor="white")
    ax1.bar(x + w/2, p2_wr, w, label="P2 Win Rate", color=COLORS["p2"], edgecolor="white")
    for i in range(len(agents)):
        ax1.text(i-w/2, p1_wr[i]+0.01, f"{p1_wr[i]:.2f}", ha="center", fontsize=7)
        ax1.text(i+w/2, p2_wr[i]+0.01, f"{p2_wr[i]:.2f}", ha="center", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Win Rate"); ax1.set_ylim(0, 1.12)
    ax1.set_title("P1 vs P2 Win Rate"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)


    colors_fma = [COLORS["win"] if f >= 0 else COLORS["loss"] for f in fma]
    ax2.bar(x, fma, color=colors_fma, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    for i, f in enumerate(fma):
        ax2.text(i, f + (0.01 if f >= 0 else -0.03), f"{f:+.2f}",
                 ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("P1 WR − P2 WR"); ax2.set_title("First Mover Advantage")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_crossplay_heatmap(rows, agent_names, title, save_path):

    if not HAS_PLOT or not rows: return
    n = len(agent_names)
    mat = np.full((n, n), np.nan)
    idx = {a: i for i, a in enumerate(agent_names)}
    for r in rows:
        i, j = idx.get(r["agent"], -1), idx.get(r["opponent"], -1)
        if i >= 0 and j >= 0:
            mat[i, j] = r["win_rate"]

    fig, ax = plt.subplots(figsize=(max(7, n*1.2), max(5, n)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Win Rate", fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(agent_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(agent_names, fontsize=8)
    ax.set_xlabel("Opponent", fontsize=10); ax.set_ylabel("Agent", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                color = "white" if (mat[i,j] < 0.25 or mat[i,j] > 0.75) else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")
            elif i != j:
                ax.text(j, i, "—", ha="center", va="center",
                        color="grey", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_nodes_expanded(node_stats_ttt, node_stats_c4, save_path):

    if not HAS_PLOT: return
    has_ttt = bool(node_stats_ttt)
    has_c4  = bool(node_stats_c4)
    if not has_ttt and not has_c4: return

    ncols = 2 if (has_ttt and has_c4) else 1
    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 8))
    if ncols == 1: axes = axes.reshape(-1, 1)
    fig.suptitle("Search Agent — Nodes Expanded & Time per Move",
                 fontsize=13, fontweight="bold")

    for col, (stats, label) in enumerate(
            [(node_stats_ttt, "TicTacToe"), (node_stats_c4, "Connect4")]
    ):
        if not stats:
            continue
        names   = [s["agent"] for s in stats]
        nodes   = [s["avg_nodes_per_move"] for s in stats]
        times   = [s["avg_time_ms"] for s in stats]
        x = np.arange(len(names))


        ax1 = axes[0, col]
        bars = ax1.bar(x, nodes, color="#1abc9c", edgecolor="white")
        ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax1.set_ylabel("Avg Nodes per Move"); ax1.set_yscale("symlog")
        ax1.set_title(f"{label} — Avg Nodes Expanded/Move")
        ax1.grid(axis="y", alpha=0.3)
        for bar, n in zip(bars, nodes):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.1 if n > 0 else 0.5,
                     f"{n:,.0f}", ha="center", fontsize=8)


        ax2 = axes[1, col]
        bars2 = ax2.bar(x, times, color="#e74c3c", edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Avg Time per Move (ms)"); ax2.set_yscale("symlog")
        ax2.set_title(f"{label} — Move Time (ms)")
        ax2.grid(axis="y", alpha=0.3)
        for bar, t in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.1 if t > 0 else 0.1,
                     f"{t:.1f}ms", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_speed_vs_quality(node_stats_ttt, node_stats_c4, save_path):

    if not HAS_PLOT: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Speed vs Quality (Search Agents)", fontsize=12, fontweight="bold")

    for ax, (stats, title) in zip(axes, [
        (node_stats_ttt, "TicTacToe"), (node_stats_c4, "Connect4")
    ]):
        if not stats:
            ax.set_visible(False)
            continue
        times = [s["avg_time_ms"] for s in stats]
        wrs   = [s["win_rate"] for s in stats]
        names = [s["agent"] for s in stats]
        sc = ax.scatter(times, wrs, s=120, c=range(len(names)),
                        cmap="tab10", zorder=3)
        for i, nm in enumerate(names):
            ax.annotate(nm, (times[i], wrs[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Avg Time per Move (ms)"); ax.set_ylabel("Win Rate vs Default")
        ax.set_title(title); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if max(times) > 100:
            ax.set_xscale("symlog")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_overall_summary(ttt_vd, c4_vd, save_path):

    if not HAS_PLOT or not ttt_vd: return
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Overall Agent Performance Summary\n(Trained & Evaluated vs Default Opponent)",
                 fontsize=14, fontweight="bold")

    def _stacked(ax, results, title):
        names = [r["agent"] for r in results]
        wr = [r["total_win_rate"] for r in results]
        dr = [r["total_draw_rate"] for r in results]
        lr = [r["total_loss_rate"] for r in results]
        x = np.arange(len(names))
        ax.bar(x, wr, label="Win",  color=COLORS["win"],  edgecolor="w")
        ax.bar(x, dr, label="Draw", color=COLORS["draw"], edgecolor="w", bottom=wr)
        ax.bar(x, lr, label="Loss", color=COLORS["loss"], edgecolor="w",
               bottom=[w+d for w,d in zip(wr,dr)])
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Rate")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.2)

    def _fma(ax, results, title):
        names = [r["agent"] for r in results]
        fma   = [r["first_mover_advantage"] for r in results]
        colors = [COLORS["win"] if f >= 0 else COLORS["loss"] for f in fma]
        x = np.arange(len(names))
        ax.bar(x, fma, color=colors, edgecolor="w")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("P1 WR − P2 WR"); ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    _stacked(fig.add_subplot(gs[0, 0]), ttt_vd, "TicTacToe — Win/Draw/Loss vs Default")
    if c4_vd:
        _stacked(fig.add_subplot(gs[0, 1]), c4_vd, "Connect4 — Win/Draw/Loss vs Default")
    _fma(fig.add_subplot(gs[1, 0]), ttt_vd, "TicTacToe — First Mover Advantage")
    if c4_vd:
        _fma(fig.add_subplot(gs[1, 1]), c4_vd, "Connect4 — First Mover Advantage")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_rl_phase_analysis(ql_rows, dqn_rows, label, save_path):

    if not HAS_PLOT: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{label} — Phase Analysis (Random warmup → Default fine-tune)",
                 fontsize=12, fontweight="bold")

    for ax, (rows, name, color) in zip(axes, [
        (ql_rows, "Q-Learning", "#3498db"),
        (dqn_rows, "DQN", "#e74c3c"),
    ]):
        if not rows:
            ax.set_visible(False)
            continue
        p1 = [r for r in rows if r.get("phase") == "phase1_random"]
        p2 = [r for r in rows if r.get("phase") == "phase2_default"]

        if p1:
            ax.plot([r["episode"] for r in p1],
                    [r["win_rate"] for r in p1],
                    color=COLORS["phase1"], linewidth=2, label="Phase 1: vs Random")
        if p2:
            ax.plot([r["episode"] for r in p2],
                    [r["win_rate"] for r in p2],
                    color=COLORS["phase2"], linewidth=2, label="Phase 2: vs Default")

        split = next((r["episode"] for r in rows if r.get("phase") == "phase2_default"), None)
        if split:
            ax.axvline(split, color="black", linestyle=":", alpha=0.6)
            ax.text(split, 0.02, " Switch", color="black", fontsize=8, va="bottom")

        ax.fill_between([r["episode"] for r in p1],
                        [r["win_rate"] for r in p1],
                        alpha=0.1, color=COLORS["phase1"])
        ax.fill_between([r["episode"] for r in p2],
                        [r["win_rate"] for r in p2],
                        alpha=0.1, color=COLORS["phase2"])

        ax.set_xlabel("Episode"); ax.set_ylabel("Win Rate")
        ax.set_title(f"{name} — Phase Comparison")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def plot_game_length(ttt_vd, c4_vd, save_path):

    if not HAS_PLOT: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Average Game Length by Agent", fontsize=12, fontweight="bold")
    max_ttt = 9
    max_c4  = 42

    for ax, (results, title, max_moves) in zip(axes, [
        (ttt_vd, "TicTacToe", max_ttt),
        (c4_vd,  "Connect4",  max_c4),
    ]):
        if not results:
            ax.set_visible(False)
            continue
        names  = [r["agent"] for r in results]
        lengths= [r["avg_game_length"] for r in results]
        times  = [r["avg_agent_time_ms_per_move"] for r in results]
        x = np.arange(len(names))

        ax2 = ax.twinx()
        ax.bar(x, lengths, color="#3498db", alpha=0.7, label="Avg game length")
        ax2.plot(x, times, "o-", color="#e74c3c", linewidth=2,
                 markersize=6, label="Time/move (ms)")
        ax.axhline(max_moves, color="grey", linestyle="--", alpha=0.5,
                   label=f"Max ({max_moves})")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Avg Game Length", color="#3498db")
        ax2.set_ylabel("Time per Move (ms)", color="#e74c3c")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max_moves * 1.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph: {save_path}")


def _save_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train all agents on Default opponent and generate all graphs.")
    parser.add_argument("--quick", action="store_true",
        help="Reduced episodes (fast demo): TTT-QL=8k, C4-QL=20k, DQN=5k/10k")
    parser.add_argument("--full",  action="store_true",
        help="Full training: TTT-QL=60k, C4-QL=150k, TTT-DQN=20k, C4-DQN=100k")
    parser.add_argument("--eval_games", type=int, default=200,
        help="Games per evaluation match (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ab_depth", type=int, default=5,
        help="Alpha-Beta search depth for evaluation (default: 5)")
    parser.add_argument("--skip_train", action="store_true",
        help="Skip training, only run evaluation + graphs on existing models")
    args = parser.parse_args()


    if args.quick:
        eps_ttt_ql, eps_c4_ql  = 8_000,  20_000
        eps_ttt_dqn, eps_c4_dqn = 5_000, 10_000
        curriculum_frac = 0.35
    elif args.full:
        eps_ttt_ql, eps_c4_ql  = 60_000, 150_000
        eps_ttt_dqn, eps_c4_dqn = 20_000, 100_000
        curriculum_frac = 0.35
    else:
        eps_ttt_ql, eps_c4_ql  = 20_000,  60_000
        eps_ttt_dqn, eps_c4_dqn = 8_000,  30_000
        curriculum_frac = 0.35

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     ADVANCED IMPLEMENTATION — FULL EXPERIMENT RUNNER            ║
║     Train ALL agents on Default opponent, then evaluate + plot  ║
╠══════════════════════════════════════════════════════════════════╣
║  TTT Q-Learning  : {eps_ttt_ql:>6,} episodes                        ║
║  C4  Q-Learning  : {eps_c4_ql:>6,} episodes  (6×7 board)            ║
║  TTT DQN         : {eps_ttt_dqn:>6,} episodes  {'(PyTorch available)' if HAS_TORCH else '(SKIPPED — no PyTorch)   '}  ║
║  C4  DQN         : {eps_c4_dqn:>6,} episodes  {'(PyTorch available)' if HAS_TORCH else '(SKIPPED — no PyTorch)   '}  ║
║  Curriculum      : {int(curriculum_frac*100):>2}% random warmup → {100-int(curriculum_frac*100):>2}% default         ║
║  Eval games      : {args.eval_games:>4} per matchup                          ║
║  Alpha-Beta depth: {args.ab_depth:>2}                                        ║
╚══════════════════════════════════════════════════════════════════╝
""")

    t_start = time.time()
    ql_ttt_rows = ql_c4_rows = dqn_ttt_rows = dqn_c4_rows = []
    ql_ttt_agent = ql_c4_agent = dqn_ttt_agent = dqn_c4_agent = None


    if not args.skip_train:
        print("\n" + "═"*60)
        print("  PHASE 1 — TRAINING")
        print("═"*60)

        ql_ttt_agent, ql_ttt_rows = train_ttt_qlearning(
            eps_ttt_ql, curriculum_frac, args.seed)
        ql_c4_agent, ql_c4_rows   = train_c4_qlearning(
            eps_c4_ql, curriculum_frac, args.seed)
        dqn_ttt_agent, dqn_ttt_rows = train_ttt_dqn(
            eps_ttt_dqn, curriculum_frac, args.seed)
        dqn_c4_agent, dqn_c4_rows   = train_c4_dqn(
            eps_c4_dqn, curriculum_frac, args.seed)
    else:
        print("\n  [Skipping training — loading existing models]")

        p = os.path.join(MODEL_DIR, "ttt_qlearning_default.pkl")
        if os.path.exists(p):
            env = GameEnv(TicTacToe, opponent="random")
            ql_ttt_agent = AdvancedQLearning(env)
            ql_ttt_agent.load(p)
            print(f"  Loaded: {p}")
        p2 = os.path.join(MODEL_DIR, "c4_qlearning_6x7_default.pkl")
        if os.path.exists(p2):
            env2 = GameEnv(Connect4, game_kwargs={"rows":6,"cols":7}, opponent="random")
            ql_c4_agent = AdvancedQLearning(env2)
            ql_c4_agent.load(p2)
            print(f"  Loaded: {p2}")


    print("\n" + "═"*60)
    print("  PHASE 2 — BUILDING AGENT ROSTERS")
    print("═"*60)


    ttt_agents = [
        RandomAgent(),
        DefaultAgent(),
        MinimaxAgent(max_depth=args.ab_depth),
        AlphaBetaAgent(max_depth=args.ab_depth),
    ]

    ql_ttt_path = os.path.join(MODEL_DIR, "ttt_qlearning_default.pkl")
    if os.path.exists(ql_ttt_path):
        a = build_rl_agent_from_model("ql", ql_ttt_path, TicTacToe)
        if a: ttt_agents.append(a)

    dqn_ttt_path = os.path.join(MODEL_DIR, "ttt_dqn_default.pt")
    if os.path.exists(dqn_ttt_path) and HAS_TORCH:
        a = build_rl_agent_from_model("dqn", dqn_ttt_path, TicTacToe)
        if a: ttt_agents.append(a)

    print(f"  TTT agents: {[a.name for a in ttt_agents]}")


    c4_agents = [
        RandomAgent(),
        DefaultAgent(),
        AlphaBetaAgent(max_depth=args.ab_depth),
        AdvancedAlphaBetaC4Agent(max_depth=args.ab_depth),
    ]

    dqn_c4_path = os.path.join(MODEL_DIR, "c4_dqn_6x7_default.pt")
    if os.path.exists(dqn_c4_path) and HAS_TORCH:
        a = build_rl_agent_from_model("dqn", dqn_c4_path, Connect4,
                                       game_kwargs={"rows":6,"cols":7})
        if a: c4_agents.append(a)

    print(f"  C4  agents: {[a.name for a in c4_agents]}")


    print("\n" + "═"*60)
    print("  PHASE 3 — NODES EXPANDED (search agents)")
    print("═"*60)

    ttt_search_agents = [
        (MinimaxAgent(max_depth=3), f"Minimax(d=3)"),
        (MinimaxAgent(max_depth=args.ab_depth), f"Minimax(d={args.ab_depth})"),
        (AlphaBetaAgent(max_depth=3), f"AlphaBeta(d=3)"),
        (AlphaBetaAgent(max_depth=args.ab_depth), f"AlphaBeta(d={args.ab_depth})"),
    ]
    c4_search_agents = [
        (AlphaBetaAgent(max_depth=3), "AlphaBeta(d=3)"),
        (AlphaBetaAgent(max_depth=args.ab_depth), f"AlphaBeta(d={args.ab_depth})"),
        (AdvancedAlphaBetaC4Agent(max_depth=3), "AdvAB_C4(d=3)"),
        (AdvancedAlphaBetaC4Agent(max_depth=args.ab_depth), f"AdvAB_C4(d={args.ab_depth})"),
    ]

    print("\n  TicTacToe search nodes:")
    node_stats_ttt = collect_nodes_stats(ttt_search_agents, TicTacToe, {}, n_games=30)
    print("\n  Connect4 search nodes:")
    node_stats_c4  = collect_nodes_stats(c4_search_agents, Connect4,
                                         {"rows":6,"cols":7}, n_games=20)
    _save_csv(node_stats_ttt, os.path.join(LOG_DIR, f"nodes_ttt_{TS}.csv"))
    _save_csv(node_stats_c4,  os.path.join(LOG_DIR, f"nodes_c4_{TS}.csv"))


    ttt_vd, c4_vd, ttt_cp, c4_cp = run_evaluation(
        ttt_agents, c4_agents, args.eval_games, args.seed)


    print("\n" + "═"*60)
    print("  PHASE 5 — GENERATING GRAPHS")
    print("═"*60)

    if not HAS_PLOT:
        print("  [SKIPPED — matplotlib not available]")
    else:
        g = GRAPH_DIR


        plot_learning_curves(ql_ttt_rows, dqn_ttt_rows, "TicTacToe",
            os.path.join(g, f"01_ttt_learning_curves_{TS}.png"))


        plot_learning_curves(ql_c4_rows, dqn_c4_rows, "Connect4",
            os.path.join(g, f"02_c4_learning_curves_{TS}.png"))


        plot_winrate_vs_default(ttt_vd,
            "TicTacToe — All Agents vs Default Opponent",
            os.path.join(g, f"03_ttt_winrate_vs_default_{TS}.png"))


        if c4_vd:
            plot_winrate_vs_default(c4_vd,
                "Connect4 — All Agents vs Default Opponent",
                os.path.join(g, f"04_c4_winrate_vs_default_{TS}.png"))


        plot_p1p2_comparison(ttt_vd,
            "TicTacToe — P1 vs P2 Win Rates & First Mover Advantage",
            os.path.join(g, f"05_ttt_p1p2_fma_{TS}.png"))


        if c4_vd:
            plot_p1p2_comparison(c4_vd,
                "Connect4 — P1 vs P2 Win Rates & First Mover Advantage",
                os.path.join(g, f"06_c4_p1p2_fma_{TS}.png"))


        if ttt_cp:
            names_ttt = list(dict.fromkeys(
                [r["agent"] for r in ttt_cp] + [r["opponent"] for r in ttt_cp]))
            plot_crossplay_heatmap(ttt_cp, names_ttt,
                "TicTacToe — Cross-play Win Rate Matrix",
                os.path.join(g, f"07_ttt_crossplay_heatmap_{TS}.png"))


        if c4_cp:
            names_c4 = list(dict.fromkeys(
                [r["agent"] for r in c4_cp] + [r["opponent"] for r in c4_cp]))
            plot_crossplay_heatmap(c4_cp, names_c4,
                "Connect4 — Cross-play Win Rate Matrix",
                os.path.join(g, f"08_c4_crossplay_heatmap_{TS}.png"))


        plot_nodes_expanded(node_stats_ttt, node_stats_c4,
            os.path.join(g, f"09_nodes_expanded_{TS}.png"))


        plot_speed_vs_quality(node_stats_ttt, node_stats_c4,
            os.path.join(g, f"10_speed_vs_quality_{TS}.png"))


        if ql_ttt_rows or dqn_ttt_rows:
            plot_rl_phase_analysis(ql_ttt_rows, dqn_ttt_rows, "TicTacToe",
                os.path.join(g, f"11_ttt_phase_analysis_{TS}.png"))
        if ql_c4_rows or dqn_c4_rows:
            plot_rl_phase_analysis(ql_c4_rows, dqn_c4_rows, "Connect4",
                os.path.join(g, f"12_c4_phase_analysis_{TS}.png"))


        plot_game_length(ttt_vd, c4_vd,
            os.path.join(g, f"13_game_length_{TS}.png"))


        plot_overall_summary(ttt_vd, c4_vd,
            os.path.join(g, f"14_overall_summary_{TS}.png"))


    elapsed = time.time() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    EXPERIMENT COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Total time  : {elapsed/60:>5.1f} minutes                               ║
║  Models saved: {MODEL_DIR}
║  Logs saved  : {LOG_DIR}
║  Graphs saved: {GRAPH_DIR}
╚══════════════════════════════════════════════════════════════════╝

  GRAPHS GENERATED (14 total):
    01  TTT learning curves (win rate + epsilon + DQN loss)
    02  C4  learning curves
    03  TTT win/draw/loss vs Default (all agents)
    04  C4  win/draw/loss vs Default (all agents)
    05  TTT P1 vs P2 win rates + First Mover Advantage
    06  C4  P1 vs P2 win rates + First Mover Advantage
    07  TTT cross-play win rate heatmap
    08  C4  cross-play win rate heatmap
    09  Nodes expanded per move (search agents, log scale)
    10  Speed vs Quality scatter (time/move vs win rate)
    11  TTT curriculum phase analysis (phase1 vs phase2)
    12  C4  curriculum phase analysis
    13  Game length + time per move comparison
    14  Overall summary dashboard (all agents, both games)
""")


if __name__ == "__main__":
    main()
