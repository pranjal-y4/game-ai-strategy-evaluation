"""
experiments/evaluate_against_default.py
────────────────────────────────────────────────────────────────────────────
Evaluate every algorithm against the Default opponent.

For each algorithm the script runs --games games, alternating which agent
moves first, and saves a CSV with win/draw/loss rates plus move-time and
nodes-expanded metrics.

Usage:
    python experiments/evaluate_against_default.py --game ttt --games 500 --seed 42
    python experiments/evaluate_against_default.py --game c4  --games 200 --seed 42

NOTE: The Connect 4 Q-learning agent uses a 4×5 reduced board; its results
are labelled "QLearning_C4_4x5" and must NOT be compared directly to the
6×7 DQN or depth-limited alpha-beta agents.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

from games.tictactoe_core import TicTacToe
from games.connect4_core  import Connect4
from utils.metrics        import MetricsCollector
from utils.serialization  import save_csv, get_timestamp
from utils.seed           import set_seed
from experiments.run_match import play_game, get_agent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Agent lists ───────────────────────────────────────────────────────────────
TTT_AGENTS = ["minimax", "alphabeta", "qlearning_ttt", "dqn_ttt", "random"]
# NOTE: c4_alphabeta is the depth-limited agent used for practical play (depth 5).
#       qlearning_c4 is evaluated on a 4x5 board; results are clearly labelled.
C4_AGENTS  = ["c4_alphabeta", "dqn_c4", "qlearning_c4", "random"]


def evaluate(game_type: str, n_games: int, seed: int, ts: str = None,
             model_paths: dict = None) -> list:
    set_seed(seed)

    if game_type == "ttt":
        game    = TicTacToe()
        agent_names = TTT_AGENTS
    else:
        game    = Connect4()
        agent_names = C4_AGENTS

    default_agent = get_agent("default", game_type)
    results = []
    if ts is None:
        ts = get_timestamp()

    for aname in agent_names:
        agent = get_agent(aname, game_type, model_paths)
        if agent is None:
            print(f"  [skip] unknown agent: {aname}")
            continue

        # For the reduced-board Q-learning agent use a 4x5 game
        if aname == "qlearning_c4":
            eval_game = Connect4(rows=4, cols=5)
        else:
            eval_game = game

        print(f"  {agent.name:30s} vs Default  ...", end=" ", flush=True)
        collector = MetricsCollector()
        p1_wins = p1_draws = p1_losses = 0
        p2_wins = p2_draws = p2_losses = 0
        for i in range(n_games):
            result = play_game(eval_game, agent, default_agent, game_num=i)
            collector.record_game(agent.name, default_agent.name, *result)
            winner_code = result[0]
            if i % 2 == 0:   # agent was P1 (first mover) per play_game alternation
                if winner_code == 1:   p1_wins   += 1
                elif winner_code == 2: p1_losses += 1
                else:                  p1_draws  += 1
            else:             # agent was P2
                if winner_code == 1:   p2_wins   += 1
                elif winner_code == 2: p2_losses += 1
                else:                  p2_draws  += 1

        s = collector.get_summary()
        p1_games = p1_wins + p1_draws + p1_losses
        p2_games = p2_wins + p2_draws + p2_losses
        board_cfg = "4x5" if aname == "qlearning_c4" else ("3x3" if game_type == "ttt" else "6x7")
        avg_len = max(s["avg_game_length"], 1)
        row = {
            "timestamp":          ts,
            "seed":               seed,
            "game":               game_type.upper(),
            "board_config":       board_cfg,
            "agent":              agent.name,
            "opponent":           default_agent.name,
            "n_games":            s["total_games"],
            "wins":               s["agent1_wins"],
            "losses":             s["agent2_wins"],
            "draws":              s["draws"],
            "win_rate":           round(s["agent1_win_rate"],   4),
            "loss_rate":          round(s["agent2_win_rate"],   4),
            "draw_rate":          round(s["draw_rate"],         4),
            "avg_game_length":    round(s["avg_game_length"],   2),
            "avg_agent_time_ms":  round(s["avg_agent1_time"] * 1000, 3),
            "avg_opp_time_ms":    round(s["avg_agent2_time"] * 1000, 3),
            "avg_agent_nodes":    round(s["avg_agent1_nodes"],  1),
            "avg_opp_nodes":      round(s["avg_agent2_nodes"],  1),
            "avg_agent_time_ms_per_move": round(s["avg_agent1_time"] * 1000 / avg_len, 4),
            "avg_agent_nodes_per_move":   round(s["avg_agent1_nodes"] / avg_len, 4),
            "p1_games":           p1_games,
            "p1_wins":            p1_wins,
            "p1_draws":           p1_draws,
            "p1_losses":          p1_losses,
            "p2_games":           p2_games,
            "p2_wins":            p2_wins,
            "p2_draws":           p2_draws,
            "p2_losses":          p2_losses,
            "p1_win_rate":  round(p1_wins   / p1_games, 4) if p1_games > 0 else 0.0,
            "p1_draw_rate": round(p1_draws  / p1_games, 4) if p1_games > 0 else 0.0,
            "p1_loss_rate": round(p1_losses / p1_games, 4) if p1_games > 0 else 0.0,
            "p2_win_rate":  round(p2_wins   / p2_games, 4) if p2_games > 0 else 0.0,
            "p2_draw_rate": round(p2_draws  / p2_games, 4) if p2_games > 0 else 0.0,
            "p2_loss_rate": round(p2_losses / p2_games, 4) if p2_games > 0 else 0.0,
        }
        p1_wr = row["p1_win_rate"]
        p2_wr = row["p2_win_rate"]
        row["first_mover_advantage"] = round(p1_wr - p2_wr, 4)
        row["schema_version"] = "v2"
        results.append(row)
        print(f"W={row['win_rate']:.1%}  D={row['draw_rate']:.1%}  "
              f"L={row['loss_rate']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all agents against the Default opponent"
    )
    parser.add_argument("--game",   choices=["ttt", "c4"], required=True)
    parser.add_argument("--games",  type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    # Optional model path overrides (use to evaluate alternate trained models
    # without overwriting the default .pkl files)
    parser.add_argument("--qlearning_ttt_model", default=None, metavar="PATH",
                        help="Override path for Q-learning TTT .pkl model")
    parser.add_argument("--dqn_ttt_model",       default=None, metavar="PATH",
                        help="Override path for DQN TTT .pkl model")
    parser.add_argument("--qlearning_c4_model",  default=None, metavar="PATH",
                        help="Override path for Q-learning C4 reduced .pkl model")
    parser.add_argument("--dqn_c4_model",        default=None, metavar="PATH",
                        help="Override path for DQN C4 .pkl model")
    args = parser.parse_args()

    model_paths = {
        'qlearning_ttt': args.qlearning_ttt_model,
        'qlearning_c4':  args.qlearning_c4_model,
        'dqn_ttt':       args.dqn_ttt_model,
        'dqn_c4':        args.dqn_c4_model,
    }

    print(f"\nEvaluating {args.game.upper()} agents vs Default  "
          f"({args.games} games, seed={args.seed})")
    print("-" * 60)

    ts = get_timestamp()
    results = evaluate(args.game, args.games, args.seed, ts=ts, model_paths=model_paths)

    if not results:
        print("No results to save.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f"vs_default_{args.game}_{ts}.csv")

    fieldnames = list(results[0].keys())
    save_csv(results, outfile, fieldnames)

    print(f"\nResults saved to:\n  {outfile}")


if __name__ == "__main__":
    main()
