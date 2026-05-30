#!/usr/bin/env bash
# run_experiments.sh — Full experiment pipeline for AI Assignment 3
# Usage:
#   bash run_experiments.sh          # full run (may take 4-6 hours)
#   bash run_experiments.sh --quick  # fast smoke test (~5-10 min)
#
# Step order:
#   1. Train RL agents (Q-Learning and DQN for TTT and C4)
#   2. Evaluate all agents vs Default
#   3. Cross-play evaluation
#   4. Connect 4 infeasibility proof
#   5. Generate all graphs

set -e
cd "$(dirname "$0")"

QUICK=0
for arg in "$@"; do [ "$arg" = "--quick" ] && QUICK=1; done

if [ $QUICK -eq 1 ]; then
    TTT_GAMES=50
    C4_GAMES=30
    TTT_CROSS=30
    C4_CROSS=20
    INFEAS=30
    RL_EPISODES_TTT=500
    RL_EPISODES_C4=2000
    RL_EVAL_EVERY_TTT=100
    RL_EVAL_EVERY_C4=500
else
    TTT_GAMES=500
    C4_GAMES=200
    TTT_CROSS=200
    C4_CROSS=100
    INFEAS=60
    RL_EPISODES_TTT=50000
    RL_EPISODES_C4=200000
    RL_EVAL_EVERY_TTT=1000
    RL_EVAL_EVERY_C4=5000
fi

SEED=42
echo "=== AI Assignment 3 — Full Experiment Pipeline ==="
echo "Seed: $SEED  |  Quick mode: $QUICK"
echo ""

echo "[1/9] Training TTT Q-Learning agent..."
python3 rl/train_qlearning_ttt.py \
    --episodes $RL_EPISODES_TTT --eval_every $RL_EVAL_EVERY_TTT \
    --seed $SEED --curriculum --eval_games 200

echo "[2/9] Training TTT DQN agent..."
python3 rl/train_dqn_ttt.py \
    --episodes $RL_EPISODES_TTT --eval_every $((RL_EVAL_EVERY_TTT/2)) \
    --seed $SEED --curriculum --eval_games 200

echo "[3/9] Training Connect4 Q-Learning (4x5 reduced board) agent..."
python3 rl/train_qlearning_c4_reduced.py \
    --episodes $((RL_EPISODES_C4)) --eval_every $RL_EVAL_EVERY_C4 \
    --seed $SEED --curriculum --eval_games 300

echo "[4/9] Training Connect4 DQN (6x7 full board) agent..."
python3 rl/train_dqn_c4.py \
    --episodes $((RL_EPISODES_C4/4)) --eval_every $((RL_EVAL_EVERY_C4/4)) \
    --seed $SEED --curriculum --eval_games 150

echo "[5/9] TTT vs Default ($TTT_GAMES games per agent)..."
python3 experiments/evaluate_against_default.py --game ttt --games $TTT_GAMES --seed $SEED

echo "[6/9] C4 vs Default ($C4_GAMES games per agent)..."
python3 experiments/evaluate_against_default.py --game c4 --games $C4_GAMES --seed $SEED

echo "[7/9] TTT Cross-play ($TTT_CROSS games per pair)..."
python3 experiments/evaluate_crossplay.py --game ttt --games $TTT_CROSS --seed $SEED

echo "[8/9] C4 Cross-play ($C4_CROSS games per pair)..."
python3 experiments/evaluate_crossplay.py --game c4 --games $C4_CROSS --seed $SEED

echo "[9/9] C4 Infeasibility proof ($INFEAS sec budget per algorithm)..."
python3 experiments/connect4_infeasibility.py --time_budget $INFEAS --seed $SEED

echo ""
echo "Generating all graphs..."
python3 experiments/plotter.py

echo ""
echo "=== Pipeline complete ==="
echo "Graphs saved to: experiments/graphs/"
echo "CSVs  saved to:  experiments/results/"
echo ""
echo "NOTE: To validate all results, run:"
echo "  python3 experiments/validate.py"

