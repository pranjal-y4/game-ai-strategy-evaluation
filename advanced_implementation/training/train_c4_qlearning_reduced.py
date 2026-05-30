import sys, os, csv, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.connect4 import Connect4
from rl.game_env import GameEnv
from rl.q_learning import AdvancedQLearning
from utils.seed import set_seed


def evaluate_split(agent, rows, cols, n_games=100):

    results = []
    saved = agent.epsilon
    agent.epsilon = 0.0
    for starts in [True, False]:
        env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                      opponent="random")
        wins = 0
        for _ in range(n_games):
            env.reset(agent_starts=starts)
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
        results.append(wins / n_games)
    agent.epsilon = saved
    return results[0], results[1]


def main(args):
    set_seed(args.seed)

    rows, cols = args.rows, args.cols
    print(f"[C4 Q-Learning REDUCED {rows}×{cols}] episodes={args.episodes} "
          f"lr={args.lr} n_step={args.n_step} seed={args.seed}")

    train_env = GameEnv(Connect4, game_kwargs={"rows": rows, "cols": cols},
                        opponent="random")
    train_env.enable_role_alternation()

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
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs",
                            f"c4_qlearning_{rows}x{cols}_{ts}.csv")
    best_path = os.path.join(os.path.dirname(__file__), "..", "models",
                             f"c4_qlearning_{rows}x{cols}_best.pkl")
    final_path = os.path.join(os.path.dirname(__file__), "..", "models",
                              f"c4_qlearning_{rows}x{cols}_final.pkl")

    curriculum_switch = int(args.episodes * args.curriculum_frac)
    phase = "phase1_random"
    best_wr = 0.0
    csv_rows = []
    reward_window = []

    for ep in range(1, args.episodes + 1):

        if ep == curriculum_switch and phase == "phase1_random":
            phase = "phase2_default"
            train_env._opponent_type = "default"
            agent.epsilon = max(agent.epsilon, 0.25)
            print(f"  [Curriculum] Switched to default opponent at ep {ep}")

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
            s_t, a_t = ep_transitions[t][0], ep_transitions[t][1]
            G = 0.0
            actual_n = 0
            for i in range(args.n_step):
                if t + i >= T:
                    break
                r_i = ep_transitions[t + i][2]
                done_i = ep_transitions[t + i][4]
                G += (args.gamma ** i) * r_i
                actual_n = i + 1
                if done_i:
                    break
            idx_n = t + actual_n - 1
            ns_n = ep_transitions[idx_n][3]
            done_n = ep_transitions[idx_n][4]
            legal_n = ep_transitions[idx_n][5]
            gn = args.gamma ** actual_n
            agent.update(s_t, a_t, G, ns_n, done_n, legal_n, gn)

        agent.decay_epsilon()
        agent.episode_rewards.append(total_r)
        reward_window.append(total_r)

        if ep % args.eval_every == 0:
            wr = agent.evaluate(200)
            agent.win_rates.append(wr)
            p1_wr, p2_wr = evaluate_split(agent, rows, cols, n_games=100)
            sm_r = float(sum(reward_window[-200:]) / max(len(reward_window[-200:]), 1))
            row = {
                "episode": ep, "epsilon": round(agent.epsilon, 4),
                "win_rate": round(wr, 4), "p1_win_rate": round(p1_wr, 4),
                "p2_win_rate": round(p2_wr, 4),
                "q_table_states": len(agent.q_table), "phase": phase,
                "smoothed_reward": round(sm_r, 4),
                "board": f"{rows}x{cols}",
            }
            csv_rows.append(row)
            if wr > best_wr:
                best_wr = wr
                agent.save(best_path)
            print(f"  ep={ep:6d} | ε={agent.epsilon:.3f} | wr={wr:.3f} "
                  f"| p1={p1_wr:.3f} | p2={p2_wr:.3f} "
                  f"| states={len(agent.q_table)} | {phase}")

    agent.save(final_path)
    fieldnames = ["episode", "epsilon", "win_rate", "p1_win_rate", "p2_win_rate",
                  "q_table_states", "phase", "smoothed_reward", "board"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[C4 Q-Learning {rows}x{cols}] Done. Best wr: {best_wr:.3f}")
    print(f"  NOTE: Results are for {rows}x{cols} board only, NOT comparable to 6x7.")
    print(f"  Final model: {final_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150_000)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_decay", type=float, default=0.9999)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum_frac", type=float, default=0.6)
    args = parser.parse_args()
    main(args)
