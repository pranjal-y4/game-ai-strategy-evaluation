import sys
import os
import csv
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from rl.game_env import GameEnv
from rl.q_learning import AdvancedQLearning
from utils.seed import set_seed


def evaluate_split(agent, eval_env_p1, eval_env_p2, n_games=200):

    results = []
    for env in [eval_env_p1, eval_env_p2]:
        saved = agent.epsilon
        agent.epsilon = 0.0
        wins = 0
        for _ in range(n_games):
            env.reset()
            state = env.encode_state()
            while True:
                legal = env.get_legal_actions()
                action = agent.choose_action(state, legal)
                _, reward, done, _ = env.step(action)
                state = env.encode_state()
                if done:
                    if reward >= 1.0:
                        wins += 1
                    break
        agent.epsilon = saved
        results.append(wins / n_games)
    return results[0], results[1]


def main(args):
    set_seed(args.seed)


    train_env = GameEnv(TicTacToe, opponent=args.opponent)
    train_env.enable_role_alternation()


    eval_env_p1 = GameEnv(TicTacToe, opponent="random")
    eval_env_p2 = GameEnv(TicTacToe, opponent="random")

    agent = AdvancedQLearning(
        env=train_env,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=0.05,
        n_step=args.n_step,
    )


    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs",
                            f"ttt_qlearning_{ts}.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "ttt_qlearning_best.pkl")
    final_model_path = os.path.join(model_dir, "ttt_qlearning_final.pkl")

    fieldnames = ["episode", "epsilon", "win_rate_random",
                  "p1_win_rate", "p2_win_rate", "q_table_states",
                  "phase", "smoothed_reward"]
    csv_rows = []
    best_wr = 0.0
    phase = "phase1_random"
    reward_window = []


    curriculum_switch = int(args.episodes * args.curriculum_frac)

    print(f"[TTT Q-Learning] episodes={args.episodes} lr={args.lr} "
          f"gamma={args.gamma} n_step={args.n_step} seed={args.seed}")
    print(f"  Curriculum: switch at episode {curriculum_switch} "
          f"({args.curriculum_frac*100:.0f}%)")

    def callback(ep, wr, eps):
        nonlocal best_wr, phase


        if ep >= curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"

            agent.epsilon = max(agent.epsilon, 0.2)
            print(f"  [Curriculum] Switched to default opponent at episode {ep}")

        p1_wr, p2_wr = evaluate_split(agent, eval_env_p1, eval_env_p2, n_games=100)
        sm_r = float(sum(reward_window[-100:]) / max(len(reward_window[-100:]), 1))
        row = {
            "episode": ep,
            "epsilon": round(eps, 4),
            "win_rate_random": round(wr, 4),
            "p1_win_rate": round(p1_wr, 4),
            "p2_win_rate": round(p2_wr, 4),
            "q_table_states": len(agent.q_table),
            "phase": phase,
            "smoothed_reward": round(sm_r, 4),
        }
        csv_rows.append(row)

        if wr > best_wr:
            best_wr = wr
            agent.save(best_model_path)

        print(f"  ep={ep:6d} | ε={eps:.3f} | wr_rand={wr:.3f} "
              f"| p1={p1_wr:.3f} | p2={p2_wr:.3f} | states={len(agent.q_table)} "
              f"| phase={phase}")


    gamma_n = args.gamma ** args.n_step
    for ep in range(1, args.episodes + 1):

        if ep == curriculum_switch:
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.epsilon = max(agent.epsilon, 0.2)
            print(f"  [Curriculum] Switched to default opponent at episode {ep}")

        state_arr = train_env.reset()
        state = train_env.encode_state()
        total_r = 0.0
        ep_transitions = []

        while True:
            legal = train_env.get_legal_actions()
            action = agent.choose_action(state, legal)
            _, reward, done, _ = train_env.step(action)
            next_state = train_env.encode_state()
            next_legal = train_env.get_legal_actions()
            ep_transitions.append((state, action, reward, next_state, done, next_legal))
            total_r += reward
            state = next_state
            if done:
                break


        T = len(ep_transitions)
        for t in range(T):
            s_t, a_t, _, _, _, _ = ep_transitions[t]
            G = 0.0
            actual_n = 0
            for i in range(args.n_step):
                if t + i >= T:
                    break
                _, _, r_i, _, done_i, _ = ep_transitions[t + i]
                G += (args.gamma ** i) * r_i
                actual_n = i + 1
                if done_i:
                    break
            idx_n = t + actual_n - 1
            _, _, _, ns_n, done_n, legal_n = ep_transitions[idx_n]
            gn = args.gamma ** actual_n
            agent.update(s_t, a_t, G, ns_n, done_n, legal_n, gn)

        agent.decay_epsilon()
        agent.episode_rewards.append(total_r)
        reward_window.append(total_r)

        if ep % args.eval_every == 0:
            wr = agent.evaluate(200)
            agent.win_rates.append(wr)
            callback(ep, wr, agent.epsilon)


    agent.save(final_model_path)
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[TTT Q-Learning] Done. Best win rate: {best_wr:.3f}")
    print(f"  Model saved: {final_model_path}")
    print(f"  Best model:  {best_model_path}")
    print(f"  Log CSV:     {log_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60_000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "default"])
    parser.add_argument("--curriculum_frac", type=float, default=0.6,
                        help="Fraction of episodes for phase 1 (random)")
    args = parser.parse_args()
    main(args)
