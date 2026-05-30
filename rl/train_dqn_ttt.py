"""
rl/train_dqn_ttt.py
────────────────────────────────────────────────────────────────────────────
Train a Deep Q-Network (DQN) on Tic-Tac-Toe.

Network: 9 → 128 → 64 → 9
Role alternation is enabled by default so the agent learns to play as
both P1 and P2.  Generalization is reported against random and semi opponents.

Usage:
    python rl/train_dqn_ttt.py --episodes 20000 --seed 42
    python rl/train_dqn_ttt.py --episodes 20000 --opponent semi --seed 42
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import time
from datetime import datetime

import numpy as np

from rl.dqn     import DQNAgent
from rl.env     import TicTacToeEnv
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "experiments", "results")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


def _greedy_eval(agent, env, n_each: int):
    """Greedy role-split eval: n_each as P1, n_each as P2.
    Returns (p1_win_rate, p2_win_rate).
    """
    saved = agent.epsilon
    agent.epsilon = 0.0
    p1_wins = p2_wins = 0

    # As P1
    for _ in range(n_each):
        env._board[:] = 0
        env.done = False
        state = env._state_from_agent()
        while True:
            legal  = env.get_legal_actions()
            action = agent._greedy(state, legal)
            state, reward, done, _ = env.step(action)
            if done:
                if reward >= 1.0:
                    p1_wins += 1
                break

    # As P2: opponent takes first move
    for _ in range(n_each):
        env._board[:] = 0
        env.done = False
        opp_action = env._opponent_action()
        if opp_action >= 0:
            env._board[opp_action] = 2
        if env._winner() == 2:
            continue
        state = env._state_from_agent()
        while True:
            legal = env.get_legal_actions()
            if not legal:
                break
            action = agent._greedy(state, legal)
            state, reward, done, _ = env.step(action)
            if done:
                if reward >= 1.0:
                    p2_wins += 1
                break

    agent.epsilon = saved
    p1_wr = p1_wins / n_each if n_each > 0 else 0.0
    p2_wr = p2_wins / n_each if n_each > 0 else 0.0
    return p1_wr, p2_wr


def train(episodes: int, seed: int, model_path: str, opponent: str = "random",
          lr: float = 5e-4, gamma: float = 0.95,
          decay_steps: int = 15_000, eval_every: int = 500):

    set_seed(seed)
    env   = TicTacToeEnv(opponent=opponent)
    env.enable_role_alternation()   # train as both P1 and P2
    agent = DQNAgent(env, hidden=[128, 64], lr=lr, gamma=gamma,
                     decay_steps=decay_steps, batch_size=64,
                     buffer_size=10_000, target_update=200)

    # Separate envs for generalization eval
    eval_random_env = TicTacToeEnv(opponent="random")
    eval_semi_env   = TicTacToeEnv(opponent="semi")

    # ── CSV logging ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"rl_training_metrics_ttt_dqn_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game", "algorithm", "board_config", "seed",
            "episode", "eval_interval", "train_opponent",
            "win_rate", "draw_rate", "loss_rate",
            "p1_win_rate", "p2_win_rate",
            "eval_win_rate_random", "eval_win_rate_semi",
            "epsilon", "avg_reward", "loss",
            "schema_version",
        ])

    start = time.time()
    first_95pct_episode = -1
    peak_win_rate = 0.0
    peak_episode  = 0

    def callback(ep: int, win_rate: float, epsilon: float):
        nonlocal first_95pct_episode, peak_win_rate, peak_episode
        elapsed = time.time() - start

        # Full greedy eval (200 games, agent as P1 vs random)
        wins = draws = losses = 0
        saved = agent.epsilon
        agent.epsilon = 0.0
        for _ in range(200):
            state = eval_random_env.reset()
            eval_random_env._board[:] = 0
            eval_random_env.done = False
            state = eval_random_env._state_from_agent()
            while True:
                legal  = eval_random_env.get_legal_actions()
                action = agent._greedy(state, legal)
                state, reward, done, _ = eval_random_env.step(action)
                if done:
                    if reward >= 1.0:   wins   += 1
                    elif reward < 0:    losses += 1
                    else:               draws  += 1
                    break
        agent.epsilon = saved
        n    = wins + draws + losses
        overall_wr = wins / n if n > 0 else 0.0
        loss = float(np.mean(agent.losses[-100:])) if agent.losses else 0.0

        # Role-split eval
        p1_wr, p2_wr = _greedy_eval(agent, eval_random_env, 100)

        # Generalization: eval vs semi opponent
        p1_semi, p2_semi = _greedy_eval(agent, eval_semi_env, 75)
        eval_wr_semi = (p1_semi + p2_semi) / 2

        if overall_wr >= 0.95 and first_95pct_episode == -1:
            first_95pct_episode = ep
        if overall_wr > peak_win_rate:
            peak_win_rate = overall_wr
            peak_episode  = ep

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                "TicTacToe", "DQN", "3x3", seed,
                ep, eval_every, opponent,
                round(overall_wr,          4),
                round(draws / n,           4) if n > 0 else 0.0,
                round(losses / n,          4) if n > 0 else 0.0,
                round(p1_wr,               4),
                round(p2_wr,               4),
                round((p1_wr + p2_wr) / 2, 4),   # eval_win_rate_random (avg roles)
                round(eval_wr_semi,         4),
                round(epsilon, 5), "", round(loss, 6),
                "v2",
            ])

        bar = "█" * int(overall_wr * 20) + "░" * (20 - int(overall_wr * 20))
        print(f"  ep={ep:>6}  ε={epsilon:.3f}  eval_rand={overall_wr:.2%}  "
              f"p1_rand={p1_wr:.1%}  p2_rand={p2_wr:.1%}  "
              f"eval_semi={eval_wr_semi:.1%}  loss={loss:.4f}  [{bar}]  {elapsed:.0f}s")

    print(f"Training TTT DQN  ({episodes} episodes, opponent={opponent}, "
          f"role_alternation=True, seed={seed}) …")
    agent.train(n_episodes=episodes, eval_every=eval_every,
                eval_episodes=200, callback=callback)

    agent.save(model_path)
    final_wr = agent.evaluate(300)
    print(f"\n  Model saved → {model_path}")
    print(f"  Final win rate (greedy, 300 eps): {final_wr:.2%}")
    print(f"  Convergence milestones:")
    print(f"    first_95pct_episode : {first_95pct_episode}")
    print(f"    peak_win_rate       : {peak_win_rate:.4f}  (ep {peak_episode})")
    print(f"  Metrics → {csv_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train DQN for Tic-Tac-Toe")
    parser.add_argument("--episodes",    type=int,   default=20_000)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--opponent",    choices=["random", "semi"], default="random",
                        help="Opponent type during training.")
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--gamma",       type=float, default=0.95)
    parser.add_argument("--decay_steps", type=int,   default=15_000)
    parser.add_argument("--eval_every",  type=int,   default=500)
    parser.add_argument("--model_path",  default=os.path.join(MODEL_DIR, "ttt_dqn.pkl"))
    args = parser.parse_args()

    train(args.episodes, args.seed, args.model_path,
          opponent=args.opponent,
          lr=args.lr, gamma=args.gamma,
          decay_steps=args.decay_steps, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
