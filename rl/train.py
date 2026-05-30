"""
rl/train.py
───────────────────────────────────────────────────────────────────────────
Ready-made training configurations for both games and both RL agents.

Usage (command-line):
    python3 -m rl.train --game ttt --agent qlearning --episodes 20000
    python3 -m rl.train --game c4  --agent dqn       --episodes 8000

Usage (import):
    from rl.train import train_ttt_qlearning, train_c4_dqn

Models are saved as:
    models/ttt_qlearning.pkl
    models/ttt_dqn.pkl
    models/c4_qlearning.pkl
    models/c4_dqn.pkl

Training metrics are saved to:
    experiments/results/rl_training_metrics_*.csv
"""

from __future__ import annotations

import os, time, argparse
import numpy as np
import csv
from rl.env        import TicTacToeEnv, Connect4Env
from rl.q_learning import TabularQLearning
from rl.dqn        import DQNAgent

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty console reporter
# ─────────────────────────────────────────────────────────────────────────────

def _reporter(label: str):
    start = time.time()

    def _cb(episode: int, win_rate: float, epsilon: float):
        elapsed = time.time() - start
        bar = "█" * int(win_rate * 20) + "░" * (20 - int(win_rate * 20))
        print(
            f"[{label}] ep={episode:>6}  ε={epsilon:.3f}  "
            f"win={win_rate:.2%}  [{bar}]  {elapsed:.1f}s"
        )
    return _cb


# ─────────────────────────────────────────────────────────────────────────────
#  TTT  –  Q-learning
# ─────────────────────────────────────────────────────────────────────────────

def train_ttt_qlearning(
    n_episodes:    int   = 20_000,
    lr:            float = 0.15,
    gamma:         float = 0.95,
    epsilon_decay: float = 0.9995,
    save:          bool  = True,
    callback              = None,
    stop_flag             = None,
) -> TabularQLearning:
    """
    Tabular Q-learning on Tic Tac Toe.

    State space: 3^9 = 19 683 (easily tractable).
    Opponent: random (good exploration coverage).
    Typically converges to ~90% win rate in ~15 000 episodes.
    """
    eval_every = 1000
    
    # Setup CSV logging
    csv_path = os.path.join(RESULTS_DIR, "rl_training_metrics_ttt_qlearning.csv")
    _create_csv_logger("TicTacToe", "Q-Learning", "3x3", csv_path)
    csv_callback = _create_csv_callback("TicTacToe", "Q-Learning", "3x3", eval_every, csv_path)
    
    def combined_callback(episode, win_rate, epsilon):
        # Evaluate full metrics
        wins, draws, losses, avg_reward = _evaluate_full(agent, 300)
        metrics = {
            "win_rate": wins / 300,
            "draw_rate": draws / 300,
            "loss_rate": losses / 300,
            "epsilon": epsilon,
            "avg_reward": avg_reward
        }
        csv_callback(episode, metrics)
        if callback:
            callback(episode, win_rate, epsilon)
    
    env   = TicTacToeEnv(opponent="random")
    agent = TabularQLearning(env, lr=lr, gamma=gamma,
                             epsilon_decay=epsilon_decay)
    print(f"Training TTT Q-learning  ({n_episodes} episodes) …")
    agent.train(
        n_episodes    = n_episodes,
        eval_every    = eval_every,
        eval_episodes = 300,
        callback      = combined_callback,
        stop_flag     = stop_flag,
    )
    if save:
        path = os.path.join(MODEL_DIR, "ttt_qlearning.pkl")
        agent.save(path)
        print(f"  Model saved → {path}")
    wr = agent.evaluate(500)
    print(f"  Final win rate (greedy, 500 eps): {wr:.2%}")
    _print_qtable_info(agent)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  TTT  –  DQN
# ─────────────────────────────────────────────────────────────────────────────

def train_ttt_dqn(
    n_episodes:  int   = 8_000,
    lr:          float = 5e-4,
    gamma:       float = 0.95,
    decay_steps: int   = 12_000,
    save:        bool  = True,
    callback           = None,
    stop_flag          = None,
) -> DQNAgent:
    """
    DQN on Tic Tac Toe.

    Full 6×7 board not applicable here; TTT state is just 9 features.
    Network: 9 → 128 → 64 → 9.
    """
    eval_every = 500
    
    # Setup CSV logging
    csv_path = os.path.join(RESULTS_DIR, "rl_training_metrics_ttt_dqn.csv")
    _create_csv_logger("TicTacToe", "DQN", "3x3", csv_path)
    csv_callback = _create_csv_callback("TicTacToe", "DQN", "3x3", eval_every, csv_path)
    
    def combined_callback(episode, win_rate, epsilon):
        # Evaluate full metrics
        wins, draws, losses, avg_reward = _evaluate_full_dqn(agent, 200)
        metrics = {
            "win_rate": wins / 200,
            "draw_rate": draws / 200,
            "loss_rate": losses / 200,
            "epsilon": epsilon,
            "avg_reward": avg_reward,
            "loss": float(np.mean(agent.losses[-100:])) if agent.losses else 0.0
        }
        csv_callback(episode, metrics)
        if callback:
            callback(episode, win_rate, epsilon)
    
    env   = TicTacToeEnv(opponent="random")
    agent = DQNAgent(env, hidden=[128, 64], lr=lr, gamma=gamma,
                     decay_steps=decay_steps)
    print(f"Training TTT DQN  ({n_episodes} episodes) …")
    agent.train(
        n_episodes    = n_episodes,
        eval_every    = eval_every,
        eval_episodes = 200,
        callback      = combined_callback,
        stop_flag     = stop_flag,
    )
    if save:
        path = os.path.join(MODEL_DIR, "ttt_dqn.pkl")
        agent.save(path)
        print(f"  Model saved → {path}")
    wr = agent.evaluate(300)
    print(f"  Final win rate (greedy, 300 eps): {wr:.2%}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Connect 4  –  Q-learning  (reduced 4×5 board)
# ─────────────────────────────────────────────────────────────────────────────

def train_c4_qlearning(
    n_episodes:    int   = 30_000,
    lr:            float = 0.1,
    gamma:         float = 0.95,
    epsilon_decay: float = 0.9998,
    rows:          int   = 4,
    cols:          int   = 5,
    save:          bool  = True,
    callback              = None,
    stop_flag             = None,
) -> TabularQLearning:
    """
    Tabular Q-learning on a REDUCED Connect 4 board (default 4×5).

    Complexity reduction rationale
    ──────────────────────────────
    Full 6×7: 3^42 ≈ 3 × 10^20 states → a sparse dict table would
    still require too many episodes to cover meaningfully.
    4×5 board: 3^20 ≈ 3.5 × 10^9 theoretical states, but in practice
    only ~10^5–10^6 are visited per run → tractable sparse table.
    The rules are identical; 4-in-a-row still applies.
    """
    board_config = f"{rows}x{cols}"
    eval_every = 2000
    
    # Setup CSV logging
    csv_path = os.path.join(RESULTS_DIR, "rl_training_metrics_c4_qlearning.csv")
    _create_csv_logger("Connect4", "Q-Learning", board_config, csv_path)
    csv_callback = _create_csv_callback("Connect4", "Q-Learning", board_config, eval_every, csv_path)
    
    def combined_callback(episode, win_rate, epsilon):
        # Evaluate full metrics
        wins, draws, losses, avg_reward = _evaluate_full(agent, 300)
        metrics = {
            "win_rate": wins / 300,
            "draw_rate": draws / 300,
            "loss_rate": losses / 300,
            "epsilon": epsilon,
            "avg_reward": avg_reward
        }
        csv_callback(episode, metrics)
        if callback:
            callback(episode, win_rate, epsilon)
    
    print(f"  [C4 Q-learning] Using reduced {rows}×{cols} board "
          f"(full 6×7 → state space 3^{rows*cols:} ≈ {3**(rows*cols):.1e})")
    env   = Connect4Env(rows=rows, cols=cols, opponent="random")
    agent = TabularQLearning(env, lr=lr, gamma=gamma,
                             epsilon_decay=epsilon_decay)
    print(f"Training C4 Q-learning  ({n_episodes} episodes, {rows}×{cols} board) …")
    agent.train(
        n_episodes    = n_episodes,
        eval_every    = eval_every,
        eval_episodes = 300,
        callback      = combined_callback,
        stop_flag     = stop_flag,
    )
    if save:
        path = os.path.join(MODEL_DIR, f"c4_qlearning_{rows}x{cols}.pkl")
        agent.save(path)
        print(f"  Model saved → {path}")
    wr = agent.evaluate(500)
    print(f"  Final win rate (greedy, 500 eps): {wr:.2%}")
    _print_qtable_info(agent)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Connect 4  –  DQN  (full 6×7 board)
# ─────────────────────────────────────────────────────────────────────────────

def train_c4_dqn(
    n_episodes:  int   = 8_000,
    lr:          float = 5e-4,
    gamma:       float = 0.95,
    decay_steps: int   = 25_000,
    rows:        int   = 6,
    cols:        int   = 7,
    save:        bool  = True,
    callback           = None,
    stop_flag          = None,
) -> DQNAgent:
    """
    DQN on the full 6×7 Connect 4 board.

    DQN handles the large state space (42 features) via function
    approximation, making board reduction unnecessary.
    Network: 42 → 128 → 64 → 7
    """
    board_config = f"{rows}x{cols}"
    eval_every = 500
    
    # Setup CSV logging
    csv_path = os.path.join(RESULTS_DIR, f"rl_training_metrics_c4_dqn.csv")
    _create_csv_logger("Connect4", "DQN", board_config, csv_path)
    csv_callback = _create_csv_callback("Connect4", "DQN", board_config, eval_every, csv_path)
    
    def combined_callback(episode, win_rate, epsilon):
        # Evaluate full metrics
        wins, draws, losses, avg_reward = _evaluate_full_dqn(agent, 150)
        metrics = {
            "win_rate": wins / 150,
            "draw_rate": draws / 150,
            "loss_rate": losses / 150,
            "epsilon": epsilon,
            "avg_reward": avg_reward,
            "loss": float(np.mean(agent.losses[-100:])) if agent.losses else 0.0
        }
        csv_callback(episode, metrics)
        if callback:
            callback(episode, win_rate, epsilon)
    
    env   = Connect4Env(rows=rows, cols=cols, opponent="random")
    agent = DQNAgent(env, hidden=[128, 64], lr=lr, gamma=gamma,
                     decay_steps=decay_steps, batch_size=64,
                     buffer_size=20_000, target_update=300)
    print(f"Training C4 DQN  ({n_episodes} episodes, {rows}×{cols} board) …")
    agent.train(
        n_episodes    = n_episodes,
        eval_every    = eval_every,
        eval_episodes = 150,
        callback      = combined_callback,
        stop_flag     = stop_flag,
    )
    if save:
        path = os.path.join(MODEL_DIR, "c4_dqn.pkl")
        agent.save(path)
        print(f"  Model saved → {path}")
    wr = agent.evaluate(200)
    print(f"  Final win rate (greedy, 200 eps): {wr:.2%}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_qtable_info(agent: TabularQLearning) -> None:
    states   = len(agent.q_table)
    entries  = sum(len(v) for v in agent.q_table.values())
    size_kb  = entries * 8 / 1024    # float64 ≈ 8 bytes
    print(f"  Q-table: {states:,} states, {entries:,} entries, ~{size_kb:.1f} KB")


def _create_csv_logger(game, algorithm, board_config, csv_path):
    """Create a CSV file for logging training metrics."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "game", "algorithm", "board_config", "episode", "eval_interval",
            "eval_opponent", "win_rate", "draw_rate", "loss_rate",
            "epsilon", "avg_reward", "loss", "notes"
        ])
    return csv_path


def _create_csv_callback(game, algorithm, board_config, eval_interval, csv_path):
    """Create a callback function that logs metrics to CSV."""
    def callback(episode, metrics):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                game,
                algorithm,
                board_config,
                episode,
                eval_interval,
                "Random Opponent",  # Training evaluation is always vs Random opponent
                metrics.get("win_rate", 0.0),
                metrics.get("draw_rate", 0.0),
                metrics.get("loss_rate", 0.0),
                metrics.get("epsilon", 0.0),
                metrics.get("avg_reward", 0.0),
                metrics.get("loss", 0.0),
                ""  # notes
            ])
    return callback


def _evaluate_full(agent, n_episodes):
    """Evaluate Q-learning agent and return wins, draws, losses, avg_reward."""
    wins = 0
    draws = 0
    losses = 0
    total_reward = 0.0
    
    for _ in range(n_episodes):
        agent.env.reset()
        state = agent.env.encode_state()
        episode_reward = 0.0
        
        while True:
            legal = agent.env.get_legal_actions()
            action = agent.choose_action(state, legal)
            _, reward, done, _ = agent.env.step(action)
            state = agent.env.encode_state()
            episode_reward += reward
            
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                total_reward += episode_reward
                break
    
    return wins, draws, losses, total_reward / n_episodes


def _evaluate_full_dqn(agent, n_episodes):
    """Evaluate DQN agent and return wins, draws, losses, avg_reward."""
    wins = 0
    draws = 0
    losses = 0
    total_reward = 0.0
    
    for _ in range(n_episodes):
        agent.env.reset()
        state = agent.env.reset()
        episode_reward = 0.0
        
        while True:
            legal = agent.env.get_legal_actions()
            action = agent.choose_action(state, legal)
            state, reward, done, _ = agent.env.step(action)
            episode_reward += reward
            
            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1
                total_reward += episode_reward
                break
    
    return wins, draws, losses, total_reward / n_episodes


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="Train RL agents for the board games")
    p.add_argument("--game",     choices=["ttt", "c4"],              default="ttt")
    p.add_argument("--agent",    choices=["qlearning", "dqn"],       default="qlearning")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--rows",     type=int, default=4,
                   help="C4 board rows (Q-learning only, default 4)")
    p.add_argument("--cols",     type=int, default=5,
                   help="C4 board cols (Q-learning only, default 5)")
    args = p.parse_args()

    if args.game == "ttt" and args.agent == "qlearning":
        kw = dict(n_episodes=args.episodes or 20_000)
        if args.lr: kw["lr"] = args.lr
        train_ttt_qlearning(**kw)
    elif args.game == "ttt" and args.agent == "dqn":
        kw = dict(n_episodes=args.episodes or 8_000)
        if args.lr: kw["lr"] = args.lr
        train_ttt_dqn(**kw)
    elif args.game == "c4" and args.agent == "qlearning":
        kw = dict(n_episodes=args.episodes or 30_000, rows=args.rows, cols=args.cols)
        if args.lr: kw["lr"] = args.lr
        train_c4_qlearning(**kw)
    elif args.game == "c4" and args.agent == "dqn":
        kw = dict(n_episodes=args.episodes or 8_000)
        if args.lr: kw["lr"] = args.lr
        train_c4_dqn(**kw)


if __name__ == "__main__":
    _cli()
