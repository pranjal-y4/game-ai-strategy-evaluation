import sys, os, csv, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.connect4 import Connect4
from rl.game_env import GameEnv
from rl.dqn import AdvancedDQNAgent
from utils.seed import set_seed


def evaluate_split(agent, rows=6, cols=7, n_games=200):

    saved = agent.epsilon
    agent.epsilon = 0.0
    results = []
    for starts in [True, False]:
        env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                      opponent="default")
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


def evaluate_vs_random(agent, rows=6, cols=7, n_games=100):

    saved = agent.epsilon
    agent.epsilon = 0.0
    env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                  opponent="random")
    wins = 0
    for _ in range(n_games):
        state = env.reset()
        while True:
            legal = env.get_legal_actions()
            action = agent._greedy(state, legal)
            state, reward, done, _ = env.step(action)
            if done:
                if reward >= 1.0:
                    wins += 1
                break
    agent.epsilon = saved
    return wins / n_games


def main(args):
    set_seed(args.seed)
    rows, cols = args.rows, args.cols

    print(f"[C4 DQN] episodes={args.episodes} lr={args.lr} "
          f"board={rows}x{cols} seed={args.seed}")
    print(f"  Curriculum: phase1={int(args.episodes*args.curriculum_frac)} eps (random)")
    print(f"              phase2={args.episodes - int(args.episodes*args.curriculum_frac)} eps (default)")
    print(f"  Reward shaping: {args.reward_shaping}")
    print(f"  Double DQN + PER + n-step({args.n_step})")


    train_env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                        opponent="random")
    train_env.enable_role_alternation()

    agent = AdvancedDQNAgent(
        env=train_env,
        hidden=[256, 128],
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        decay_steps=args.decay_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update=args.target_update,
        n_step=args.n_step,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_steps=args.episodes * 10,
        use_per=True,
        grad_clip=10.0,
    )


    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs",
                            f"c4_dqn_{rows}x{cols}_{ts}.csv")
    ckpt_phase1 = os.path.join(os.path.dirname(__file__), "..", "models",
                               f"c4_dqn_{rows}x{cols}_phase1.pt")
    best_path = os.path.join(os.path.dirname(__file__), "..", "models",
                             f"c4_dqn_{rows}x{cols}_best.pt")
    final_path = os.path.join(os.path.dirname(__file__), "..", "models",
                              f"c4_dqn_{rows}x{cols}_final.pt")

    curriculum_switch = int(args.episodes * args.curriculum_frac)
    phase = "phase1_random"
    phase_switched = False
    best_wr = 0.0
    csv_rows = []
    loss_window = []


    print(f"\n=== Phase 1: vs Random (episodes 1–{curriculum_switch}) ===")

    def phase1_callback(ep, wr, eps, avg_loss):
        nonlocal best_wr
        sm_r = (float(np.mean(agent.episode_rewards[-200:]))
                if len(agent.episode_rewards) >= 200 else 0.0)

        p1_wr = wr
        p2_wr = 0.0
        row = {
            "episode": ep, "epsilon": round(eps, 4),
            "win_rate": round(wr, 4), "p1_win_rate": round(p1_wr, 4),
            "p2_win_rate": round(p2_wr, 4), "avg_loss": round(avg_loss, 6),
            "phase": "phase1_random", "smoothed_reward": round(sm_r, 4),
            "vs_random": round(wr, 4), "board": f"{rows}x{cols}",
        }
        csv_rows.append(row)
        if wr > best_wr:
            best_wr = wr
            agent.save(best_path)
        print(f"  ep={ep:6d} | ε={eps:.3f} | wr={wr:.3f} "
              f"| loss={avg_loss:.4f} | phase1_random")

    agent.train(
        n_episodes=curriculum_switch,
        eval_every=args.eval_every,
        eval_episodes=100,
        reward_shaping=args.reward_shaping,
        shaping_weight=args.shaping_weight,
        shaping_clip=args.shaping_clip,
        callback=phase1_callback,
    )


    agent.save(ckpt_phase1)
    print(f"  [Checkpoint] Phase 1 model saved: {ckpt_phase1}")


    phase2_eps = args.episodes - curriculum_switch
    print(f"\n=== Phase 2: vs Default (episodes {curriculum_switch+1}–{args.episodes}) ===")


    train_env._opponent_type = "default"
    phase = "phase2_default"


    new_eps = max(agent.epsilon, args.phase2_epsilon_reset)
    agent.set_epsilon(new_eps)
    print(f"  [Curriculum] Epsilon reset to {new_eps:.3f}")

    def phase2_callback(ep, wr, eps, avg_loss):
        nonlocal best_wr
        actual_ep = curriculum_switch + ep
        p1_wr, p2_wr = evaluate_split(agent, rows, cols, n_games=100)
        wr_rand = evaluate_vs_random(agent, rows, cols, n_games=50)
        sm_r = (float(np.mean(agent.episode_rewards[-200:]))
                if len(agent.episode_rewards) >= 200 else 0.0)
        row = {
            "episode": actual_ep, "epsilon": round(eps, 4),
            "win_rate": round(wr, 4), "p1_win_rate": round(p1_wr, 4),
            "p2_win_rate": round(p2_wr, 4), "avg_loss": round(avg_loss, 6),
            "phase": "phase2_default", "smoothed_reward": round(sm_r, 4),
            "vs_random": round(wr_rand, 4), "board": f"{rows}x{cols}",
        }
        csv_rows.append(row)
        if p1_wr + p2_wr > best_wr * 2:
            best_wr = (p1_wr + p2_wr) / 2
            agent.save(best_path)
        print(f"  ep={actual_ep:6d} | ε={eps:.3f} | wr_default={wr:.3f} "
              f"| p1={p1_wr:.3f} | p2={p2_wr:.3f} "
              f"| wr_rand={wr_rand:.3f} | loss={avg_loss:.4f} | phase2_default")

    agent.train(
        n_episodes=phase2_eps,
        eval_every=args.eval_every,
        eval_episodes=100,
        reward_shaping=args.reward_shaping,
        shaping_weight=args.shaping_weight,
        shaping_clip=args.shaping_clip,
        callback=phase2_callback,
    )


    agent.save(final_path)
    fieldnames = ["episode", "epsilon", "win_rate", "p1_win_rate",
                  "p2_win_rate", "avg_loss", "phase", "smoothed_reward",
                  "vs_random", "board"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[C4 DQN] Training complete.")
    print(f"  Final model: {final_path}")
    print(f"  Best model:  {best_path}")
    print(f"  Log CSV:     {log_path}")
    print(f"\n  IMPORTANT: Phase 2 fine-tuning optimises vs the DEFAULT opponent.")
    print(f"  This may yield strong vs-default performance but is not")
    print(f"  equivalent to solving Connect4 optimally.")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--decay_steps", type=int, default=60_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--target_update", type=int, default=500)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum_frac", type=float, default=0.6,
                        help="Fraction for phase 1 (random)")
    parser.add_argument("--phase2_epsilon_reset", type=float, default=0.3,
                        help="Minimum epsilon at phase 2 start")
    parser.add_argument("--reward_shaping", action="store_true",
                        help="Enable reward shaping during training")
    parser.add_argument("--shaping_weight", type=float, default=0.01)
    parser.add_argument("--shaping_clip", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
