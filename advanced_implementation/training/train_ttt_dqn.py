import sys, os, csv, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from rl.game_env import GameEnv
from rl.dqn import AdvancedDQNAgent
from utils.seed import set_seed


def evaluate_split(agent, n_games=200):

    results = []
    saved = agent.epsilon
    agent.epsilon = 0.0
    for starts in [True, False]:
        env = GameEnv(TicTacToe, opponent="random")
        wins = 0
        for _ in range(n_games):
            state = env.reset(agent_starts=starts)
            while True:
                legal = env.get_legal_actions()
                action = agent._greedy(state, legal)
                state, reward, done, _ = env.step(action)
                if done:
                    if reward >= 1.0:
                        wins += 1
                    break
        results.append(wins / n_games)
    agent.epsilon = saved
    return results[0], results[1]


def main(args):
    set_seed(args.seed)
    print(f"[TTT DQN] episodes={args.episodes} lr={args.lr} seed={args.seed}")

    train_env = GameEnv(TicTacToe, opponent=args.opponent)
    train_env.enable_role_alternation()

    agent = AdvancedDQNAgent(
        env=train_env,
        hidden=[128, 64],
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        decay_steps=args.decay_steps,
        batch_size=64,
        buffer_size=20_000,
        target_update=200,
        n_step=args.n_step,
        use_per=True,
    )

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs",
                            f"ttt_dqn_{ts}.csv")
    best_path = os.path.join(os.path.dirname(__file__), "..", "models",
                             "ttt_dqn_best.pt")
    final_path = os.path.join(os.path.dirname(__file__), "..", "models",
                              "ttt_dqn_final.pt")

    curriculum_switch = int(args.episodes * args.curriculum_frac)
    phase = "phase1_random"
    best_wr = 0.0
    csv_rows = []

    def callback(ep, wr, eps, avg_loss):
        nonlocal best_wr, phase
        if ep >= curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.set_epsilon(max(eps, 0.2))
            print(f"  [Curriculum] Switched to default opponent at ep {ep}")

        p1_wr, p2_wr = evaluate_split(agent, n_games=100)
        sm_r = (float(np.mean(agent.episode_rewards[-200:]))
                if len(agent.episode_rewards) >= 200 else 0.0)
        row = {
            "episode": ep, "epsilon": round(eps, 4),
            "win_rate": round(wr, 4), "p1_win_rate": round(p1_wr, 4),
            "p2_win_rate": round(p2_wr, 4), "avg_loss": round(avg_loss, 6),
            "phase": phase, "smoothed_reward": round(sm_r, 4),
        }
        csv_rows.append(row)
        if wr > best_wr:
            best_wr = wr
            agent.save(best_path)
        print(f"  ep={ep:6d} | ε={eps:.3f} | wr={wr:.3f} "
              f"| p1={p1_wr:.3f} | p2={p2_wr:.3f} "
              f"| loss={avg_loss:.4f} | {phase}")

    agent.train(
        n_episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=200,
        reward_shaping=False,
        callback=callback,
    )

    agent.save(final_path)
    fieldnames = ["episode", "epsilon", "win_rate", "p1_win_rate",
                  "p2_win_rate", "avg_loss", "phase", "smoothed_reward"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[TTT DQN] Done. Best win rate: {best_wr:.3f}")
    print(f"  Final model: {final_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--decay_steps", type=int, default=15_000)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", type=str, default="random")
    parser.add_argument("--curriculum_frac", type=float, default=0.6)
    args = parser.parse_args()
    main(args)
