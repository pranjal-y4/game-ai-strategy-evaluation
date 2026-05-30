"""
rl/train_qlearning_ttt.py
────────────────────────────────────────────────────────────────────────────
Train tabular Q-learning on Tic-Tac-Toe.

State space: 3^9 = 19 683  (fully tractable for a sparse Q-table).
Typically converges to ~90% win rate vs random in ~20 000 episodes.

Role alternation (enabled by default):
  Agent trains as P1 and P2 in alternating episodes, producing a policy
  that is robust to both starting roles rather than overspecialising to
  the first-mover position.

Generalization evaluation:
  At each eval checkpoint the agent is evaluated against both a random
  opponent and a semi-smart (win > block > random) opponent.  The gap
  between these metrics quantifies how well training transfers beyond
  the training distribution.

Usage:
    python rl/train_qlearning_ttt.py --episodes 50000 --seed 42
    python rl/train_qlearning_ttt.py --episodes 50000 --opponent semi --seed 42
    python rl/train_qlearning_ttt.py --episodes 50000 --model_path models/ttt_qlearning.pkl
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time
from datetime import datetime

from rl.q_learning import TabularQLearning
from rl.env        import TicTacToeEnv
from utils.seed    import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "experiments", "results")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)


def _evaluate_policy(agent, env, n_each: int):
    """Greedy evaluation split by role: n_each as P1, n_each as P2.
    Returns (win_rate, draw_rate, loss_rate, p1_win_rate, p2_win_rate, fma).
    Uses env.reset(agent_starts=...) so role semantics are correct.
    """
    saved = agent.epsilon
    agent.epsilon = 0.0
    wins = draws = losses = 0
    p1_wins = p2_wins = 0

    # As P1: agent_starts=True → agent_id=1, moves first
    for _ in range(n_each):
        env.reset(agent_starts=True)
        while True:
            legal = env.get_legal_actions()
            action = agent.choose_action(env.encode_state(), legal)
            _, reward, done, _ = env.step(action)
            if done:
                if reward >= 1.0:
                    p1_wins += 1; wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                break

    # As P2: agent_starts=False → agent_id=2, opponent already moved in reset()
    for _ in range(n_each):
        env.reset(agent_starts=False)
        if env.done:
            losses += 1
            continue
        while True:
            legal = env.get_legal_actions()
            if not legal:
                draws += 1
                break
            action = agent.choose_action(env.encode_state(), legal)
            _, reward, done, _ = env.step(action)
            if done:
                if reward >= 1.0:
                    p2_wins += 1; wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                break

    agent.epsilon = saved
    n_total = n_each * 2
    wr = wins / n_total if n_total > 0 else 0.0
    dr = draws / n_total if n_total > 0 else 0.0
    lr = losses / n_total if n_total > 0 else 0.0
    p1_wr = p1_wins / n_each if n_each > 0 else 0.0
    p2_wr = p2_wins / n_each if n_each > 0 else 0.0
    fma = round(p1_wr - p2_wr, 4)
    return wr, dr, lr, p1_wr, p2_wr, fma


def train(episodes: int, seed: int, model_path: str, opponent: str = "random",
          lr: float = 0.15, gamma: float = 0.95,
          epsilon_decay: float = 0.9995, eval_every: int = 1000,
          curriculum: bool = False, switch_frac: float = 0.5, eval_games: int = 200):

    set_seed(seed)
    env      = TicTacToeEnv(opponent=opponent)
    env.enable_role_alternation()   # train as both P1 and P2
    agent    = TabularQLearning(env, lr=lr, gamma=gamma,
                                epsilon_decay=epsilon_decay)

    # Curriculum: start vs random, switch to semi at switch_frac of training
    # opponent_type is updated in the eval callback (every eval_every episodes)
    if curriculum:
        env.opponent_type = "random"  # start with random

    # Separate envs for generalization eval (never used for training)
    eval_random_env = TicTacToeEnv(opponent="random")
    eval_semi_env   = TicTacToeEnv(opponent="semi")

    # ── CSV logging ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"rl_training_metrics_ttt_qlearning_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game", "algorithm", "board_config", "seed",
            "episode", "eval_interval", "training_opponent",
            "win_rate", "draw_rate", "loss_rate",
            "p1_win_rate", "p2_win_rate",
            "eval_win_rate_random", "eval_draw_rate_random", "eval_loss_rate_random",
            "eval_p1_win_rate_random", "eval_p2_win_rate_random", "eval_fma_random",
            "eval_win_rate_semi", "eval_draw_rate_semi", "eval_loss_rate_semi",
            "eval_p1_win_rate_semi", "eval_p2_win_rate_semi", "eval_fma_semi",
            "epsilon", "avg_reward", "loss", "q_table_states",
            "schema_version",
        ])

    start = time.time()
    # Convergence tracking
    first_95pct_episode = -1
    peak_win_rate = 0.0
    peak_episode  = 0

    def callback(ep: int, win_rate: float, epsilon: float):
        nonlocal first_95pct_episode, peak_win_rate, peak_episode
        elapsed = time.time() - start

        # Curriculum: switch opponent at switch_frac of total episodes
        if curriculum:
            env.opponent_type = "semi" if ep > int(episodes * switch_frac) else "random"

        # Dual-opponent evaluation
        n_each = max(1, eval_games // 2)
        wr_rand, dr_rand, lr_rand, p1_rand, p2_rand, fma_rand = _evaluate_policy(agent, eval_random_env, n_each)
        wr_semi, dr_semi, lr_semi, p1_semi, p2_semi, fma_semi = _evaluate_policy(agent, eval_semi_env, n_each)

        overall_wr = wr_rand  # Use random performance for overall convergence tracking

        # Update convergence milestones
        if overall_wr >= 0.95 and first_95pct_episode == -1:
            first_95pct_episode = ep
        if overall_wr > peak_win_rate:
            peak_win_rate = overall_wr
            peak_episode  = ep

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                "TicTacToe", "Q-Learning", "3x3", seed,
                ep, eval_every, env.opponent_type,
                round(wr_rand, 4), round(dr_rand, 4), round(lr_rand, 4),
                round(p1_rand, 4), round(p2_rand, 4),
                round(wr_rand, 4), round(dr_rand, 4), round(lr_rand, 4), round(p1_rand, 4), round(p2_rand, 4), round(fma_rand, 4),
                round(wr_semi, 4), round(dr_semi, 4), round(lr_semi, 4), round(p1_semi, 4), round(p2_semi, 4), round(fma_semi, 4),
                round(epsilon, 5), "", "", len(agent.q_table),
                "v2",
            ])

        bar = "█" * int(overall_wr * 20) + "░" * (20 - int(overall_wr * 20))
        print(f"  ep={ep:>6}  ε={epsilon:.3f}  eval_rand={overall_wr:.2%}  "
              f"p1_rand={p1_rand:.1%}  p2_rand={p2_rand:.1%}  "
              f"eval_semi={wr_semi:.1%}  [{bar}]  {elapsed:.0f}s")

    print(f"Training TTT Q-learning  ({episodes} episodes, opponent={opponent}, "
          f"role_alternation=True, seed={seed}) …")
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
    parser = argparse.ArgumentParser(description="Train Q-learning for Tic-Tac-Toe")
    parser.add_argument("--episodes",    type=int,   default=50_000)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--opponent",    choices=["random", "semi"], default="random",
                        help="Opponent type during training. "
                             "'random' = random legal moves (default). "
                             "'semi' = win > block > random (harder, better generalisation).")
    parser.add_argument("--lr",          type=float, default=0.15)
    parser.add_argument("--gamma",       type=float, default=0.95)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--eval_every",  type=int,   default=1_000)
    parser.add_argument("--curriculum",  action="store_true", help="Enable curriculum training")
    parser.add_argument("--switch_frac", type=float, default=0.5, help="Fraction of episodes at which to switch to semi opponent")
    parser.add_argument("--eval_games",  type=int,   default=200, help="Number of evaluation games per test")
    parser.add_argument("--model_path",  default=os.path.join(MODEL_DIR, "ttt_qlearning.pkl"))
    args = parser.parse_args()

    train(args.episodes, args.seed, args.model_path,
          opponent=args.opponent,
          lr=args.lr, gamma=args.gamma,
          epsilon_decay=args.epsilon_decay, eval_every=args.eval_every,
          curriculum=args.curriculum, switch_frac=args.switch_frac, eval_games=args.eval_games)


if __name__ == "__main__":
    main()
