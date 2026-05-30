"""
rl/train_qlearning_c4_reduced.py
────────────────────────────────────────────────────────────────────────────
Train tabular Q-learning on a REDUCED Connect 4 board.

Why reduced board?
──────────────────
Full 6×7 board: 3^42 ≈ 3×10^20 theoretical states — the sparse Q-table
would never achieve meaningful coverage even with millions of episodes.
Reduced 4×5 board: 3^20 ≈ 3.5×10^9 theoretical states, but in practice
only ~10^5–10^6 distinct states are visited per training run, keeping the
table tractable.  The rules are identical (4-in-a-row wins).

IMPORTANT: Results on the 4×5 board are NOT directly comparable to results
on the 6×7 board.  All CSV rows for this agent are labelled "4x5".

Role alternation (enabled by default):
  Connect 4 has a substantial first-mover advantage.  Training in both
  roles is especially important here.

Usage:
    python rl/train_qlearning_c4_reduced.py --episodes 100000 --seed 42
    python rl/train_qlearning_c4_reduced.py --episodes 100000 --opponent semi --seed 42
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time
from datetime import datetime

from rl.q_learning import TabularQLearning
from rl.env        import Connect4Env
from utils.seed    import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "experiments", "results")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


def _greedy_eval(agent, env, n_each: int):
    """Greedy role-split eval: n_each as P1, n_each as P2."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    p1_wins = p2_wins = 0

    # As P1
    for _ in range(n_each):
        env.reset()
        env._board[:] = 0
        env.done = False
        state = env.encode_state()
        while True:
            legal = env.get_legal_actions()
            action = agent.choose_action(state, legal)
            _, reward, done, _ = env.step(action)
            state = env.encode_state()
            if done:
                if reward >= 1.0:
                    p1_wins += 1
                break

    # As P2: opponent moves first
    for _ in range(n_each):
        env._board[:] = 0
        env.done = False
        legal_opp = env.get_legal_actions()
        if legal_opp:
            opp_action = env._opponent_action(legal_opp)
            env._drop(opp_action, 2)
            if env._winner() == 2:
                continue
        state = env.encode_state()
        while True:
            legal = env.get_legal_actions()
            if not legal:
                break
            action = agent.choose_action(state, legal)
            _, reward, done, _ = env.step(action)
            state = env.encode_state()
            if done:
                if reward >= 1.0:
                    p2_wins += 1
                break

    agent.epsilon = saved_eps
    p1_wr = p1_wins / n_each if n_each > 0 else 0.0
    p2_wr = p2_wins / n_each if n_each > 0 else 0.0
    return p1_wr, p2_wr


def train(episodes: int, seed: int, rows: int, cols: int, model_path: str,
          opponent: str = "random",
          lr: float = 0.1, gamma: float = 0.95,
          epsilon_decay: float = 0.9998, eval_every: int = 5_000):

    set_seed(seed)
    board_cfg = f"{rows}x{cols}"
    print(f"  Board: {board_cfg}  |  state space ≈ 3^{rows*cols} = {3**(rows*cols):.2e}")

    env   = Connect4Env(rows=rows, cols=cols, opponent=opponent)
    env.enable_role_alternation()   # train as both P1 and P2
    agent = TabularQLearning(env, lr=lr, gamma=gamma,
                             epsilon_decay=epsilon_decay)

    # Separate eval envs for generalization
    eval_random_env = Connect4Env(rows=rows, cols=cols, opponent="random")
    eval_semi_env   = Connect4Env(rows=rows, cols=cols, opponent="semi")

    # ── CSV logging ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"rl_training_metrics_c4_qlearning_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game", "algorithm", "board_config", "seed",
            "episode", "eval_interval", "train_opponent",
            "win_rate", "draw_rate", "loss_rate",
            "p1_win_rate", "p2_win_rate",
            "eval_win_rate_random", "eval_win_rate_semi",
            "epsilon", "avg_reward", "loss", "q_table_states",
            "schema_version",
        ])

    start = time.time()
    first_95pct_episode = -1
    peak_win_rate = 0.0
    peak_episode  = 0

    def callback(ep: int, win_rate: float, epsilon: float):
        nonlocal first_95pct_episode, peak_win_rate, peak_episode
        elapsed = time.time() - start

        # Greedy eval vs random (300 games, agent as P1)
        wins = draws = losses = 0
        saved_eps = agent.epsilon
        agent.epsilon = 0.0
        for _ in range(300):
            eval_random_env._board[:] = 0
            eval_random_env.done = False
            state = eval_random_env.encode_state()
            while True:
                legal = eval_random_env.get_legal_actions()
                action = agent.choose_action(state, legal)
                _, reward, done, _ = eval_random_env.step(action)
                state = eval_random_env.encode_state()
                if done:
                    if reward >= 1.0:   wins   += 1
                    elif reward < 0:    losses += 1
                    else:               draws  += 1
                    break
        agent.epsilon = saved_eps
        n = wins + draws + losses
        overall_wr = wins / n if n > 0 else 0.0

        # Role-split eval
        p1_wr, p2_wr = _greedy_eval(agent, eval_random_env, 150)

        # Generalization: eval vs semi opponent
        p1_semi, p2_semi = _greedy_eval(agent, eval_semi_env, 100)
        eval_wr_semi = (p1_semi + p2_semi) / 2

        if overall_wr >= 0.95 and first_95pct_episode == -1:
            first_95pct_episode = ep
        if overall_wr > peak_win_rate:
            peak_win_rate = overall_wr
            peak_episode  = ep

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                "Connect4", "Q-Learning", board_cfg, seed,
                ep, eval_every, opponent,
                round(overall_wr,          4),
                round(draws / n,           4) if n > 0 else 0.0,
                round(losses / n,          4) if n > 0 else 0.0,
                round(p1_wr,               4),
                round(p2_wr,               4),
                round((p1_wr + p2_wr) / 2, 4),
                round(eval_wr_semi,         4),
                round(epsilon, 5), "", "", len(agent.q_table),
                "v2",
            ])

        bar = "█" * int(overall_wr * 20) + "░" * (20 - int(overall_wr * 20))
        print(f"  ep={ep:>7}  ε={epsilon:.4f}  eval_rand={overall_wr:.2%}  "
              f"p1_rand={p1_wr:.1%}  p2_rand={p2_wr:.1%}  "
              f"eval_semi={eval_wr_semi:.1%}  [{bar}]  {elapsed:.0f}s")

    print(f"Training Connect4 Q-learning {board_cfg}  ({episodes} episodes, "
          f"opponent={opponent}, role_alternation=True, seed={seed}) …")
    agent.train(n_episodes=episodes, eval_every=eval_every,
                eval_episodes=300, callback=callback)

    agent.save(model_path)
    final_wr  = agent.evaluate(500)
    q_states  = len(agent.q_table)
    q_entries = sum(len(v) for v in agent.q_table.values())
    print(f"\n  Model saved → {model_path}")
    print(f"  Final win rate (greedy, 500 eps): {final_wr:.2%}")
    print(f"  Q-table: {q_states:,} states, {q_entries:,} entries, "
          f"~{q_entries * 8 / 1024:.1f} KB")
    print(f"  Convergence milestones:")
    print(f"    first_95pct_episode : {first_95pct_episode}")
    print(f"    peak_win_rate       : {peak_win_rate:.4f}  (ep {peak_episode})")
    print(f"  Metrics → {csv_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-learning for reduced Connect 4"
    )
    parser.add_argument("--episodes",      type=int,   default=100_000)
    parser.add_argument("--rows",          type=int,   default=4)
    parser.add_argument("--cols",          type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--opponent",      choices=["random", "semi"], default="random",
                        help="Opponent type during training.")
    parser.add_argument("--lr",            type=float, default=0.1)
    parser.add_argument("--gamma",         type=float, default=0.95)
    parser.add_argument("--epsilon_decay", type=float, default=0.9998)
    parser.add_argument("--eval_every",    type=int,   default=5_000)
    parser.add_argument("--model_path",    default=None)
    args = parser.parse_args()

    default_model = os.path.join(MODEL_DIR, f"c4_qlearning_{args.rows}x{args.cols}.pkl")
    model_path = args.model_path or default_model

    train(args.episodes, args.seed, args.rows, args.cols, model_path,
          opponent=args.opponent,
          lr=args.lr, gamma=args.gamma,
          epsilon_decay=args.epsilon_decay, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
