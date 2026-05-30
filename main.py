"""
main.py
Entry point for the project.

Use --ui to launch the Tkinter GUI demo.
All training and evaluation experiments run headlessly from command line
(see README.txt for full command list).
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI Game Agents — Minimax / RL Comparison")
    parser.add_argument("--game", choices=["ttt", "c4"], default="ttt",
                        help="Game to demo (default: ttt)")
    parser.add_argument("--ui", action="store_true",
                        help="Launch Tkinter GUI demo")
    args = parser.parse_args()

    if args.ui:
        if args.game == "ttt":
            from games.tictactoe_ui import TicTacToeUI
            TicTacToeUI().run()
        else:
            from games.connect4_ui import Connect4UI
            Connect4UI().run()
    else:
        print("=" * 65)
        print("AI Assignment 3 — Minimax / Alpha-Beta / Q-Learning / DQN")
        print("=" * 65)
        print()
        print("Run experiments headlessly (no GUI needed):")
        print()
        print("  # Train RL agents")
        print("  python rl/train_qlearning_ttt.py          --episodes 50000  --seed 42")
        print("  python rl/train_dqn_ttt.py                --episodes 20000  --seed 42")
        print("  python rl/train_qlearning_c4_reduced.py   --episodes 100000 --seed 42")
        print("  python rl/train_dqn_c4.py                 --episodes 100000 --seed 42")
        print()
        print("  # Prove Connect 4 infeasibility")
        print("  python experiments/connect4_infeasibility.py --time_budget 60   # quick test")
        print("  python experiments/connect4_infeasibility.py --time_budget 1800 # full run")
        print()
        print("  # Evaluate vs Default opponent")
        print("  python experiments/evaluate_against_default.py --game ttt --games 500 --seed 42")
        print("  python experiments/evaluate_against_default.py --game c4  --games 200 --seed 42")
        print()
        print("  # Cross-play tournament")
        print("  python experiments/evaluate_crossplay.py --game ttt --games 200 --seed 42")
        print("  python experiments/evaluate_crossplay.py --game c4  --games 100 --seed 42")
        print()
        print("  # Single matchup")
        print("  python experiments/run_match.py --game ttt --agent1 minimax --agent2 default --games 50 --seed 42")
        print()
        print("  # Generate graphs")
        print("  python experiments/plotter.py")
        print()
        print("  # Optional GUI demo")
        print("  python main.py --ui --game ttt")
        print("  python main.py --ui --game c4")
        print()
        print("See README.txt for the full command reference.")


if __name__ == "__main__":
    main()
