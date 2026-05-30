================================================================================
AI Assignment 3 — Game-Playing Agents: Minimax, Alpha-Beta, Q-Learning, DQN
================================================================================

Games     : Tic-Tac-Toe (3×3) and Connect 4 (6×7, with 4×5 reduced board for Q-learning)
Algorithms: Minimax, Alpha-Beta, Tabular Q-Learning, DQN (pure NumPy, no PyTorch)
Schema    : All result CSVs use schema_version="v2"

All training and evaluation runs are HEADLESS (no GUI required).
The GUI is only used for the optional demo (python3 main.py --ui --game ttt).

================================================================================
REQUIREMENTS
================================================================================

    pip install numpy matplotlib pandas seaborn

No PyTorch is required. DQN is implemented in pure NumPy (rl/dqn.py).

================================================================================
PROJECT STRUCTURE
================================================================================

    games/
        tictactoe_core.py          Headless TTT game (clone/legal_moves/apply_move/...)
        connect4_core.py           Headless Connect 4 game (configurable rows/cols)
        tictactoe_ui.py            Tkinter GUI (demo only)
        connect4_ui.py             Tkinter GUI (demo only)

    agents/
        base_agent.py              Abstract interface (reset / select_action / name)
        random_agent.py            Random legal move
        default_agent.py           Heuristic: win > block > smart fallback
        ttt_minimax_agent.py       Full minimax for TTT (node counting, depth scoring)
        ttt_alphabeta_agent.py     Alpha-beta pruning for TTT
        c4_minimax_infeasible_agent.py    Full minimax C4 (INFEASIBILITY DEMO ONLY)
        c4_alphabeta_infeasible_agent.py  Full alpha-beta C4 (INFEASIBILITY DEMO ONLY)
        c4_depthlimited_alphabeta_agent.py  Depth-limited AB (depth 5, heuristic eval)
        qlearning_ttt_agent.py     Loads trained Q-table for TTT inference
        qlearning_c4_reduced_agent.py    Loads Q-table for REDUCED 4×5 C4 inference
        dqn_ttt_agent.py           Loads DQN weights for TTT inference
        dqn_c4_agent.py            Loads DQN weights for full 6×7 C4 inference

    rl/
        env.py                     Gym-like environments (TicTacToeEnv, Connect4Env)
                                   Both support enable_role_alternation() so agents
                                   train as both P1 and P2 in alternating episodes.
        q_learning.py              Tabular Q-learning with epsilon-greedy + action masking
        dqn.py                     Deep Q-Network (pure NumPy MLP, replay buffer, target net)
        train_qlearning_ttt.py     Training script: Q-learning on TTT
        train_qlearning_c4_reduced.py  Training script: Q-learning on reduced 4×5 C4
        train_dqn_ttt.py           Training script: DQN on TTT
        train_dqn_c4.py            Training script: DQN on full 6×7 C4

    experiments/
        run_match.py               Single matchup (any two agents, any game)
        evaluate_against_default.py  All agents vs Default opponent
        evaluate_crossplay.py      Round-robin cross-play matrix
        run_tournament.py          Full tournament (vs-default + crossplay)
        connect4_infeasibility.py  Infeasibility demonstration with time budget
        aggregate_results.py       Multi-seed statistical aggregator (mean ± std, 95% CI)
        plotter.py                 Generate all report graphs from saved CSVs
        validate.py                Correctness validation (11 checks)
        results/                   Saved CSV outputs (timestamped filenames)
        graphs/                    Saved PNG charts

    utils/
        seed.py         set_seed(seed) for random / numpy
        metrics.py      MetricsCollector for win/draw/loss/time/nodes statistics
        serialization.py  save_csv, save_json, get_timestamp
        plotting.py     Basic matplotlib helpers

    models/           Saved model files (.pkl)
    run_experiments.sh  One-command pipeline (bash run_experiments.sh [--quick])
    CHANGELOG.md      Record of all fixes and additions
    main.py           Entry point (GUI demo launcher)

================================================================================
IMPORTANT DESIGN NOTES
================================================================================

DEFAULT OPPONENT
  win > block > smart fallback (deterministic).
  TTT fallback: center → corners → edges.
  C4 fallback:  center column first, then adjacent columns outward.

RL ROLE ALTERNATION
  All RL training scripts call env.enable_role_alternation() before training.
  On even episodes the agent plays as P1 (first mover); on odd episodes the
  opponent takes the opening move and the agent plays as P2.
  This prevents overspecialisation to one starting role and produces policies
  that are robust to both roles.  Role-conditioned metrics (p1_win_rate,
  p2_win_rate) are logged at every eval checkpoint.

RL OPPONENT TYPES
  --opponent random (default): good for initial exploration coverage.
  --opponent semi:  win > block > random; harder, improves generalisation.
  For curriculum training, run the full pipeline twice:
      python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent random --seed 42
      python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent semi   --seed 42

RL GENERALIZATION EVALUATION
  At each eval checkpoint the agent is tested against BOTH a random and a
  semi opponent. The CSV columns eval_win_rate_random and eval_win_rate_semi
  show the generalization gap (how much performance drops on a harder opponent).

CONNECT 4 INFEASIBILITY
  c4_minimax_infeasible_agent.py and c4_alphabeta_infeasible_agent.py are
  used ONLY for the infeasibility experiment — they time out on a 6×7 board.
  For actual C4 play, C4DepthLimitedAlphaBetaAgent (depth 5) is used.

TABULAR Q-LEARNING FOR CONNECT 4 — REDUCED BOARD
  Full 6×7 board: 3^42 ≈ 3×10^20 states — intractable for a sparse Q-table.
  Reduced 4×5 board: 3^20 ≈ 3.5×10^9 theoretical, ~10^5–10^6 visited in
  practice — tractable. Rules are identical (4-in-a-row wins).
  All CSV rows for this agent are labelled "board_config=4x5" and are plotted
  separately. They must NOT be compared directly to 6×7 results.

CSV SCHEMA (v2)
  All evaluation CSVs include: schema_version, game, board_config, seed,
  agent, opponent, win_rate, draw_rate, loss_rate, p1_win_rate, p2_win_rate,
  first_mover_advantage, avg_agent_time_ms_per_move, avg_agent_nodes_per_move.
  The plotter and aggregator reject files that lack schema_version="v2".

WINNER TRACKING
  play_game() in run_match.py correctly maps board player (1 or 2) to agent
  identity (agent1 or agent2) accounting for alternating starting player.

EVALUATION FAIRNESS NOTES
  • In Tic-Tac-Toe, 100% draws vs a competent default indicates optimal or
    near-optimal play, NOT poor performance.
  • High win rate vs Random ≠ strong performance vs Default or stronger agents.
  • A policy with much higher p1_win_rate than p2_win_rate is NOT robust.
  • Reduced-board C4 results are justified approximations; they are NOT
    directly comparable to full-board 6×7 results.

================================================================================
ONE-COMMAND PIPELINE
================================================================================

    bash run_experiments.sh           # full run  (~3–4 hours, 500/200 games)
    bash run_experiments.sh --quick   # quick test (~5–10 min, 50/30 games)

================================================================================
EXPERIMENT COMMANDS (run from project root)
================================================================================

--- STEP 1: Train RL Agents ---

  # Tabular Q-learning — Tic-Tac-Toe (~50k episodes, role alternation enabled)
  python3 rl/train_qlearning_ttt.py --episodes 50000 --seed 42
  python3 rl/train_qlearning_ttt.py --episodes 50000 --opponent semi --seed 42

  # Tabular Q-learning — Connect 4 reduced 4×5 (~100k episodes)
  python3 rl/train_qlearning_c4_reduced.py --episodes 100000 --rows 4 --cols 5 --seed 42

  # DQN — Tic-Tac-Toe (~20k episodes)
  python3 rl/train_dqn_ttt.py --episodes 20000 --seed 42

  # DQN — full 6×7 Connect 4 (~100k episodes)
  python3 rl/train_dqn_c4.py --episodes 100000 --seed 42

  Models saved to models/  |  Training metrics to experiments/results/

--- STEP 2: Connect 4 Infeasibility Demonstration ---

  # Quick test (60 seconds per algorithm)
  python3 experiments/connect4_infeasibility.py --time_budget 60 --seed 42

  # Full assignment run (30 minutes per algorithm)
  python3 experiments/connect4_infeasibility.py --time_budget 1800 --seed 42

--- STEP 3: Evaluate All Agents vs Default Opponent ---

  python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed 42
  python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed 42

--- STEP 4: Cross-Play Round-Robin ---

  python3 experiments/evaluate_crossplay.py --game ttt --games 200 --seed 42
  python3 experiments/evaluate_crossplay.py --game c4  --games 100 --seed 42

--- STEP 5: Full Tournament in One Command ---

  python3 experiments/run_tournament.py --games 200 --seed 42
  python3 experiments/run_tournament.py --game ttt --games 500 --seed 42

--- STEP 6: Single Matchup (debug / spot-check) ---

  python3 experiments/run_match.py --game ttt --agent1 minimax   --agent2 default --games 50 --seed 42
  python3 experiments/run_match.py --game ttt --agent1 alphabeta --agent2 random  --games 100 --seed 42
  python3 experiments/run_match.py --game c4  --agent1 c4_alphabeta --agent2 default --games 50 --seed 42

  Agent names --game ttt: random, default, minimax, alphabeta, qlearning_ttt, dqn_ttt
  Agent names --game c4:  random, default, c4_alphabeta, qlearning_c4, dqn_c4

--- STEP 7: Generate All Graphs ---

  python3 experiments/plotter.py

  Reads the latest schema-compatible v2 CSVs from experiments/results/.
  Saves all PNGs to experiments/graphs/.

--- STEP 8: Validate Correctness ---

  python3 experiments/validate.py

  Runs 11 checks including: default agent behaviour, Minimax vs AlphaBeta
  agreement, legal-move compliance, model loading, role alternation, P1+P2
  metric sums, board-size isolation, and plotter schema safety.

--- STEP 9: Aggregate Multi-Seed Results ---

  python3 experiments/aggregate_results.py
  python3 experiments/aggregate_results.py --game ttt   # TTT only

  Produces mean ± std and 95% CI across seeds for all compatible v2 CSVs.

--- OPTIONAL: GUI Demo ---

  python3 main.py --ui --game ttt
  python3 main.py --ui --game c4

================================================================================
MULTI-SEED ANALYSIS (recommended: 3–5 seeds)
================================================================================

  for SEED in 42 123 999; do
    python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed $SEED
    python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed $SEED
    python3 experiments/evaluate_crossplay.py       --game ttt --games 200 --seed $SEED
    python3 experiments/evaluate_crossplay.py       --game c4  --games 100 --seed $SEED
  done
  python3 experiments/aggregate_results.py

================================================================================
OUTPUT FILES (timestamped — YYYYMMDD_HHMMSS in all filenames)
================================================================================

  experiments/results/
    vs_default_ttt_<ts>.csv          Agents vs Default, TTT (3×3)
    vs_default_c4_<ts>.csv           Agents vs Default, C4 (6×7 + 4×5 rows)
    crossplay_ttt_<ts>.csv           Round-robin, TTT
    crossplay_c4_<ts>.csv            Round-robin, C4 (6×7 agents only)
    c4_infeasibility_<ts>.csv        Infeasibility metrics
    rl_training_metrics_ttt_qlearning_<ts>.csv
    rl_training_metrics_ttt_dqn_<ts>.csv
    rl_training_metrics_c4_qlearning_<ts>.csv  (4×5 board)
    rl_training_metrics_c4_dqn_<ts>.csv        (6×7 board)
    aggregated_vs_default_<ts>.csv   Multi-seed mean ± std
    aggregated_crossplay_<ts>.csv    Multi-seed mean ± std

  experiments/graphs/
    ttt_vs_default_winrate.png           Win rate bar chart with 95% CI
    ttt_vs_default_outcomes.png          Stacked outcome bars
    ttt_vs_default_role_stratified.png   P1 vs P2 outcomes per agent
    c4_vs_default_winrate.png            (6×7 agents only)
    c4_vs_default_outcomes.png
    c4_vs_default_role_stratified.png
    c4_reduced_vs_default_winrate.png    4×5 Q-learning agent (separate chart)
    c4_reduced_vs_default_outcomes.png
    ttt_crossplay_heatmap.png            Win-rate + draw-rate dual heatmap
    c4_crossplay_heatmap.png
    ttt_fma.png                          First-mover advantage bar chart
    c4_fma.png
    overall_comparison_dashboard.png     4-metric grouped bar dashboard
    speed_vs_quality.png                 Decision time vs win rate scatter
    decision_time_comparison.png         Per-move decision time bar chart
    ttt_qlearning_training_winrate.png   Training curves (overall + P1/P2 + semi)
    ttt_dqn_training_winrate.png
    ttt_dqn_loss.png
    c4_qlearning_training_winrate.png
    c4_dqn_training_winrate.png
    c4_dqn_loss.png
    c4_minimax_infeasibility_nodes.png
    c4_minimax_infeasibility_depth.png

================================================================================
EVALUATION FAIRNESS AND CURRICULUM TRAINING
================================================================================

Role Fairness:
In Connect 4, and significantly in Tic-Tac-Toe, playing first (Player 1) gives a 
massive inherent advantage. To rigorously evaluate all agents (Minimax and RL alike), 
they are uniformly tested in both roles (150 games as P1, 150 as P2 against the default opponent)
to avoid the "first-mover advantage" artificially inflating win rates. This is fully validated in 
`validate.py` by ensuring computations remain strictly honest to `p1_win_rate` and `p2_win_rate`.

RL Training Curriculum:
RL agents face a significant hurdle early on: state-space sparseness. 
If they play a strong opponent initially, they lose instantly without collecting meaningful rewards. 
Our RL curriculum starts training against a "random" opponent. At a designated point midway 
through training, the environment seamlessly switches to a "semi-intelligent" opponent, 
increasing the challenge sequentially. Evaluating generalization logged against both random and 
semi opponents exposes vulnerabilities where standard agents might overfit strictly to random behavior.

Reduced-Board Connect 4 (Q-Learning):
Full 6x7 Connect 4 contains ~3e20 states. A sparse tabular array cannot capture meaningful 
knowledge here before catastrophic out-of-memory errors occur, effectively producing a random agent.
Therefore, our Tabular Q-Learning C4 agent intentionally trains and tests on a strictly isolated 
4x5 board solely to practically demonstrate Q-Learning mechanics. Graphically and analytically, 
it is segregated from 6x7 deep/alpha-beta agents to avoid fallacious comparisons.

Overall Comparison:
Agents are ultimately aggregated alongside 'Overall Comparison Dashboards'. These dashboards are descriptive
portraits capturing performance tradeoffs (Speed, VS-Default WR, CrossPlay WR, FMA) against our baselines.
They explicitly do not crown a solitary "General Winner", recognizing that Minimax vs Deep Q-Networks 
encounter wildly heterogeneous system scale constraints.

================================================================================
ALGORITHM ROLES (assignment requirement mapping)
================================================================================

  INFEASIBILITY DEMO (Connect 4):
    c4_minimax_infeasible_agent.py    plain Minimax, no pruning, depth tracked
    c4_alphabeta_infeasible_agent.py  Alpha-Beta, center-first ordering, depth tracked
    Script: python3 experiments/connect4_infeasibility.py --time_budget 1800

  PRACTICAL CONNECT 4 PLAY:
    c4_depthlimited_alphabeta_agent.py  depth-limited AB (depth=5, heuristic eval)
      Heuristic: center column bonus + window evaluation (4/3/2-in-row scoring)

  REDUCED-BOARD Q-LEARNING (Connect 4):
    qlearning_c4_reduced_agent.py  trained on 4×5 board (justification above)
    Evaluated separately; NOT compared to 6×7 agents in the same chart.

  TIC-TAC-TOE:
    ttt_minimax_agent.py    full Minimax, depth-sensitive scoring (win=10-depth)
    ttt_alphabeta_agent.py  Alpha-Beta pruning, same scoring
    qlearning_ttt_agent.py  Tabular Q-learning (full 3×3 state space)
    dqn_ttt_agent.py        DQN (9→128→64→9)

================================================================================
