"""
experiments/run_tournament.py
────────────────────────────────────────────────────────────────────────────
Full tournament launcher: runs all vs-default and crossplay evaluations
for both games and saves all results in one pass.

Usage:
    python experiments/run_tournament.py --games 200 --seed 42
    python experiments/run_tournament.py --game ttt --games 500 --seed 42
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from utils.seed          import set_seed
from utils.serialization import get_timestamp
from experiments.evaluate_against_default import evaluate as eval_vs_default, RESULTS_DIR
from experiments.evaluate_crossplay        import crossplay
from utils.serialization import save_csv


def run_all(game_type: str, n_games: int, seed: int) -> None:
    ts = get_timestamp()
    print(f"\n{'='*70}")
    print(f"Tournament: {game_type.upper()} | {n_games} games/pair | seed={seed}")
    print(f"{'='*70}")

    # VS Default
    print(f"\n[vs Default]")
    vsd_results = eval_vs_default(game_type, n_games, seed, ts=ts)
    outfile = os.path.join(RESULTS_DIR, f"tournament_vsd_{game_type}_{ts}.csv")
    if vsd_results:
        save_csv(vsd_results, outfile, list(vsd_results[0].keys()))
        print(f"  vs-default results → {outfile}")

    # Cross-play
    print(f"\n[Cross-play]")
    cp_results = crossplay(game_type, n_games, seed, ts=ts)
    outfile = os.path.join(RESULTS_DIR, f"tournament_cp_{game_type}_{ts}.csv")
    if cp_results:
        save_csv(cp_results, outfile, list(cp_results[0].keys()))
        print(f"  crossplay results  → {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Full tournament for one or both games")
    parser.add_argument("--game",  choices=["ttt", "c4", "both"], default="both")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    games_to_run = ["ttt", "c4"] if args.game == "both" else [args.game]
    for g in games_to_run:
        run_all(g, args.games, args.seed)

    print("\nTournament complete.")


if __name__ == "__main__":
    main()
