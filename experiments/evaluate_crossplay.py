"""
experiments/evaluate_crossplay.py
────────────────────────────────────────────────────────────────────────────
Round-robin cross-play between all agents for a given game.

Every ordered pair (A, B) plays --games games with starting player
alternated each game.  Results are saved as a full pairwise matrix CSV
(suitable for heatmap generation by experiments/plotter.py).

Usage:
    python3 experiments/evaluate_crossplay.py --game ttt --games 200 --seed 42
    python3 experiments/evaluate_crossplay.py --game c4  --games 100 --seed 42

RUNTIME NOTE: Cross-play is expensive — N*(N-1) ordered pairs, each running
--games games.  TTT with 6 agents = 30 pairs × 200 games = 6 000 games.
C4 with 4 agents = 12 pairs × 100 games = 1 200 games (much slower per game).
Use --games 50 for a quick sanity-check run.

NOTE: The C4 Q-learning agent (4×5 board) is excluded from the Connect 4
cross-play against 6×7 agents.  It is evaluated separately with:
    python3 experiments/run_match.py --game c4 --agent1 qlearning_c4 --agent2 random ...
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from games.tictactoe_core import TicTacToe
from games.connect4_core  import Connect4
from utils.metrics        import MetricsCollector
from utils.serialization  import save_csv, get_timestamp
from utils.seed           import set_seed
from experiments.run_match import play_game, get_agent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Agent lists ───────────────────────────────────────────────────────────────
TTT_AGENTS = ["random", "default", "minimax", "alphabeta", "qlearning_ttt", "dqn_ttt"]
# Exclude qlearning_c4 (4x5 board) from the 6x7 C4 crossplay
C4_AGENTS  = ["random", "default", "c4_alphabeta", "dqn_c4"]


def crossplay(game_type: str, n_games: int, seed: int, ts: str = None,
              model_paths: dict = None) -> list:
    set_seed(seed)

    if game_type == "ttt":
        game        = TicTacToe()
        agent_names = TTT_AGENTS
        board_cfg   = "3x3"
    else:
        game        = Connect4()
        agent_names = C4_AGENTS
        board_cfg   = "6x7"

    if ts is None:
        ts = get_timestamp()
    results = []

    n_pairs = len(agent_names) * (len(agent_names) - 1)
    pair_num = 0
    for a1_name in agent_names:
        for a2_name in agent_names:
            if a1_name == a2_name:
                continue
            pair_num += 1
            a1 = get_agent(a1_name, game_type, model_paths)
            a2 = get_agent(a2_name, game_type, model_paths)
            if a1 is None or a2 is None:
                continue

            print(f"  [{pair_num:2d}/{n_pairs}]  {a1.name:28s} vs {a2.name:28s} ...", end=" ", flush=True)
            collector = MetricsCollector()
            p1_wins = p1_draws = p1_losses = 0
            p2_wins = p2_draws = p2_losses = 0
            for i in range(n_games):
                result = play_game(game, a1, a2, game_num=i)
                collector.record_game(a1.name, a2.name, *result)
                winner_code = result[0]
                if i % 2 == 0:   # a1 was P1 (first mover)
                    if winner_code == 1:   p1_wins   += 1
                    elif winner_code == 2: p1_losses += 1
                    else:                  p1_draws  += 1
                else:             # a1 was P2
                    if winner_code == 1:   p2_wins   += 1
                    elif winner_code == 2: p2_losses += 1
                    else:                  p2_draws  += 1

            s = collector.get_summary()
            p1_games = p1_wins + p1_draws + p1_losses
            p2_games = p2_wins + p2_draws + p2_losses
            avg_len = max(s["avg_game_length"], 1)
            row = {
                "timestamp":         ts,
                "seed":              seed,
                "game":              game_type.upper(),
                "board_config":      board_cfg,
                "agent":             a1.name,
                "opponent":          a2.name,
                "n_games":           s["total_games"],
                "wins":              s["agent1_wins"],
                "losses":            s["agent2_wins"],
                "draws":             s["draws"],
                "win_rate":          round(s["agent1_win_rate"],  4),
                "loss_rate":         round(s["agent2_win_rate"],  4),
                "draw_rate":         round(s["draw_rate"],        4),
                "avg_game_length":   round(s["avg_game_length"],  2),
                "avg_agent1_time_ms": round(s["avg_agent1_time"] * 1000, 3),
                "avg_agent2_time_ms": round(s["avg_agent2_time"] * 1000, 3),
                "avg_agent1_nodes":  round(s["avg_agent1_nodes"], 1),
                "avg_agent2_nodes":  round(s["avg_agent2_nodes"], 1),
                "avg_agent1_time_ms_per_move": round(s["avg_agent1_time"] * 1000 / avg_len, 4),
                "avg_agent1_nodes_per_move":   round(s["avg_agent1_nodes"] / avg_len, 4),
                "p1_games":          p1_games,
                "p1_wins":           p1_wins,
                "p1_draws":          p1_draws,
                "p1_losses":         p1_losses,
                "p2_games":          p2_games,
                "p2_wins":           p2_wins,
                "p2_draws":          p2_draws,
                "p2_losses":         p2_losses,
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
            print(f"W={row['win_rate']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Round-robin cross-play between all agents. "
                    "WARNING: expensive — N*(N-1) pairs × --games each. "
                    "Use --games 50 for a quick sanity check."
    )
    parser.add_argument("--game",   choices=["ttt", "c4"], required=True)
    parser.add_argument("--games",  type=int, default=200,
                        help="Games per ordered pair (default 200). Use 50 for a quick run.")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    # Optional model path overrides
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

    print(f"\nCross-play {args.game.upper()}  ({args.games} games/pair, seed={args.seed})")
    print("-" * 60)

    ts = get_timestamp()
    results = crossplay(args.game, args.games, args.seed, ts=ts, model_paths=model_paths)

    if not results:
        print("No results to save.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f"crossplay_{args.game}_{ts}.csv")

    fieldnames = list(results[0].keys())
    save_csv(results, outfile, fieldnames)

    print(f"\nResults saved to:\n  {outfile}")


if __name__ == "__main__":
    main()
