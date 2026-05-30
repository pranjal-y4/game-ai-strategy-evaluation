================================================================================
GAME AI STRATEGY EVALUATION
================================================================================

Comparative evaluation of classical search and reinforcement learning agents on
Tic-Tac-Toe and Connect 4.

Repository name suggestion: game-ai-strategy-evaluation

--------------------------------------------------------------------------------
PROJECT SNAPSHOT
--------------------------------------------------------------------------------

Games       : Tic-Tac-Toe 3x3 and Connect 4 6x7
Reduced C4  : 4x5 board used only for Tabular Q-Learning experiments
Algorithms  : Minimax, Alpha-Beta Pruning, Tabular Q-Learning, Deep Q-Networks
DQN Stack   : Pure NumPy implementation, no PyTorch required
CSV Schema  : schema_version="v2" for all result files
Execution   : Training and evaluation are headless; GUI is optional demo only

Optional GUI demo:

    python3 main.py --ui --game ttt
    python3 main.py --ui --game c4

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

Install dependencies:

    pip install numpy matplotlib pandas seaborn

No PyTorch is required. The DQN implementation is written in pure NumPy and lives
in:

    rl/dqn.py

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

Run the full experimental pipeline:

    bash run_experiments.sh

Run a faster smoke test:

    bash run_experiments.sh --quick

Expected runtime:

    Full run   : about 3-4 hours, depending on machine performance
    Quick run  : about 5-10 minutes

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

    games/
        tictactoe_core.py                 Headless Tic-Tac-Toe engine
        connect4_core.py                  Headless Connect 4 engine
        tictactoe_ui.py                   Tkinter GUI demo for Tic-Tac-Toe
        connect4_ui.py                    Tkinter GUI demo for Connect 4

    agents/
        base_agent.py                     Shared agent interface
        random_agent.py                   Random legal-move agent
        default_agent.py                  Heuristic baseline agent
        ttt_minimax_agent.py              Full Minimax for Tic-Tac-Toe
        ttt_alphabeta_agent.py            Alpha-Beta for Tic-Tac-Toe
        c4_minimax_infeasible_agent.py    Full Minimax C4 infeasibility demo
        c4_alphabeta_infeasible_agent.py  Full Alpha-Beta C4 infeasibility demo
        c4_depthlimited_alphabeta_agent.py Practical depth-limited C4 Alpha-Beta
        qlearning_ttt_agent.py            Loads trained Tic-Tac-Toe Q-table
        qlearning_c4_reduced_agent.py     Loads reduced-board Connect 4 Q-table
        dqn_ttt_agent.py                  Loads Tic-Tac-Toe DQN weights
        dqn_c4_agent.py                   Loads full-board Connect 4 DQN weights

    rl/
        env.py                            Gym-like environments for both games
        q_learning.py                     Tabular Q-Learning implementation
        dqn.py                            Pure NumPy DQN implementation
        train_qlearning_ttt.py            Train Q-Learning on Tic-Tac-Toe
        train_qlearning_c4_reduced.py     Train Q-Learning on reduced 4x5 C4
        train_dqn_ttt.py                  Train DQN on Tic-Tac-Toe
        train_dqn_c4.py                   Train DQN on full 6x7 C4

    experiments/
        run_match.py                      Run one matchup between two agents
        evaluate_against_default.py       Evaluate agents against baseline
        evaluate_crossplay.py             Round-robin cross-play evaluation
        run_tournament.py                 Full tournament runner
        connect4_infeasibility.py         C4 search infeasibility experiment
        aggregate_results.py              Multi-seed statistical aggregation
        plotter.py                        Generate graphs from result CSVs
        validate.py                       Correctness validation checks
        results/                          Timestamped CSV outputs
        graphs/                           Generated PNG charts

    utils/
        seed.py                           Reproducible random seeds
        metrics.py                        Win/draw/loss/time/node metrics
        serialization.py                  CSV, JSON, and timestamp helpers
        plotting.py                       Matplotlib helper functions

    models/                               Saved trained model files
    run_experiments.sh                    One-command experiment pipeline
    CHANGELOG.md                          Fixes and additions log
    main.py                               GUI demo launcher

--------------------------------------------------------------------------------
CORE DESIGN NOTES
--------------------------------------------------------------------------------

1. Default Opponent

The default opponent is deterministic and follows this priority:

    win -> block -> smart fallback

For Tic-Tac-Toe, the fallback order is:

    center -> corners -> edges

For Connect 4, the fallback order is:

    center column -> adjacent columns outward

2. RL Role Alternation

All RL training scripts call:

    env.enable_role_alternation()

This makes the agent train as both Player 1 and Player 2:

    Even episodes : agent plays first as P1
    Odd episodes  : opponent opens, agent plays as P2

This prevents the RL agents from overfitting to one starting role. Role-specific
metrics such as p1_win_rate and p2_win_rate are logged at evaluation checkpoints.

3. RL Opponent Types

Supported training opponents:

    --opponent random    Useful for early exploration
    --opponent semi      win -> block -> random, harder and better for generalisation

Example curriculum-style training:

    python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent random --seed 42
    python3 rl/train_qlearning_ttt.py --episodes 25000 --opponent semi   --seed 42

4. RL Generalisation Evaluation

At each evaluation checkpoint, RL agents are tested against both random and semi
opponents. The result CSVs include:

    eval_win_rate_random
    eval_win_rate_semi

These columns show how much performance drops when the opponent becomes stronger.

5. Connect 4 Infeasibility

Full Minimax and full Alpha-Beta are included for Connect 4 only as an
infeasibility demonstration. On a full 6x7 board, they time out quickly because
the search space is too large.

Practical Connect 4 play uses:

    C4DepthLimitedAlphaBetaAgent, depth=5

6. Reduced-Board Q-Learning for Connect 4

Full Connect 4 has approximately:

    3^42 ~= 3 x 10^20 states

This is not tractable for a sparse tabular Q-table.

The Tabular Q-Learning Connect 4 agent therefore uses a reduced 4x5 board:

    3^20 ~= 3.5 x 10^9 theoretical states
    about 10^5 to 10^6 states visited in practice

The rules remain the same: 4-in-a-row wins.

Important: reduced-board Q-Learning results are labelled board_config=4x5 and
must not be directly compared with full 6x7 Connect 4 agents.

--------------------------------------------------------------------------------
CSV SCHEMA
--------------------------------------------------------------------------------

All evaluation CSVs use schema_version="v2" and include:

    schema_version
    game
    board_config
    seed
    agent
    opponent
    win_rate
    draw_rate
    loss_rate
    p1_win_rate
    p2_win_rate
    first_mover_advantage
    avg_agent_time_ms_per_move
    avg_agent_nodes_per_move

The plotter and aggregator reject files that do not use schema_version="v2".

--------------------------------------------------------------------------------
EXPERIMENT COMMANDS
--------------------------------------------------------------------------------

Run all commands from the project root.

1. Train RL agents

    # Tabular Q-Learning on Tic-Tac-Toe
    python3 rl/train_qlearning_ttt.py --episodes 50000 --seed 42
    python3 rl/train_qlearning_ttt.py --episodes 50000 --opponent semi --seed 42

    # Tabular Q-Learning on reduced 4x5 Connect 4
    python3 rl/train_qlearning_c4_reduced.py --episodes 100000 --rows 4 --cols 5 --seed 42

    # DQN on Tic-Tac-Toe
    python3 rl/train_dqn_ttt.py --episodes 20000 --seed 42

    # DQN on full 6x7 Connect 4
    python3 rl/train_dqn_c4.py --episodes 100000 --seed 42

Models are saved to:

    models/

Training metrics are saved to:

    experiments/results/

2. Run Connect 4 infeasibility demo

    # Quick test, 60 seconds per algorithm
    python3 experiments/connect4_infeasibility.py --time_budget 60 --seed 42

    # Full assignment run, 30 minutes per algorithm
    python3 experiments/connect4_infeasibility.py --time_budget 1800 --seed 42

3. Evaluate all agents against the default opponent

    python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed 42
    python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed 42

4. Run cross-play round robin

    python3 experiments/evaluate_crossplay.py --game ttt --games 200 --seed 42
    python3 experiments/evaluate_crossplay.py --game c4  --games 100 --seed 42

5. Run tournament

    python3 experiments/run_tournament.py --games 200 --seed 42
    python3 experiments/run_tournament.py --game ttt --games 500 --seed 42

6. Run single matchups for debugging

    python3 experiments/run_match.py --game ttt --agent1 minimax   --agent2 default --games 50  --seed 42
    python3 experiments/run_match.py --game ttt --agent1 alphabeta --agent2 random  --games 100 --seed 42
    python3 experiments/run_match.py --game c4  --agent1 c4_alphabeta --agent2 default --games 50 --seed 42

Supported agent names:

    Tic-Tac-Toe : random, default, minimax, alphabeta, qlearning_ttt, dqn_ttt
    Connect 4  : random, default, c4_alphabeta, qlearning_c4, dqn_c4

7. Generate graphs

    python3 experiments/plotter.py

This reads the latest compatible v2 CSV files from:

    experiments/results/

And saves PNG graphs to:

    experiments/graphs/

8. Validate correctness

    python3 experiments/validate.py

The validation script runs 11 checks, including:

    default agent behaviour
    Minimax vs Alpha-Beta agreement
    legal-move compliance
    model loading
    role alternation
    P1/P2 metric consistency
    board-size isolation
    plotter schema safety

9. Aggregate multi-seed results

    python3 experiments/aggregate_results.py
    python3 experiments/aggregate_results.py --game ttt

This produces mean, standard deviation, and 95% confidence intervals across
compatible v2 CSV files.

--------------------------------------------------------------------------------
MULTI-SEED ANALYSIS
--------------------------------------------------------------------------------

Recommended: run 3 to 5 seeds.

    for SEED in 42 123 999; do
      python3 experiments/evaluate_against_default.py --game ttt --games 500 --seed $SEED
      python3 experiments/evaluate_against_default.py --game c4  --games 200 --seed $SEED
      python3 experiments/evaluate_crossplay.py       --game ttt --games 200 --seed $SEED
      python3 experiments/evaluate_crossplay.py       --game c4  --games 100 --seed $SEED
    done

    python3 experiments/aggregate_results.py

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------

All result files are timestamped using this format:

    YYYYMMDD_HHMMSS

CSV outputs:

    experiments/results/
        vs_default_ttt_<ts>.csv
        vs_default_c4_<ts>.csv
        crossplay_ttt_<ts>.csv
        crossplay_c4_<ts>.csv
        c4_infeasibility_<ts>.csv
        rl_training_metrics_ttt_qlearning_<ts>.csv
        rl_training_metrics_ttt_dqn_<ts>.csv
        rl_training_metrics_c4_qlearning_<ts>.csv
        rl_training_metrics_c4_dqn_<ts>.csv
        aggregated_vs_default_<ts>.csv
        aggregated_crossplay_<ts>.csv

Graph outputs:

    experiments/graphs/
        ttt_vs_default_winrate.png
        ttt_vs_default_outcomes.png
        ttt_vs_default_role_stratified.png
        c4_vs_default_winrate.png
        c4_vs_default_outcomes.png
        c4_vs_default_role_stratified.png
        c4_reduced_vs_default_winrate.png
        c4_reduced_vs_default_outcomes.png
        ttt_crossplay_heatmap.png
        c4_crossplay_heatmap.png
        ttt_fma.png
        c4_fma.png
        overall_comparison_dashboard.png
        speed_vs_quality.png
        decision_time_comparison.png
        ttt_qlearning_training_winrate.png
        ttt_dqn_training_winrate.png
        ttt_dqn_loss.png
        c4_qlearning_training_winrate.png
        c4_dqn_training_winrate.png
        c4_dqn_loss.png
        c4_minimax_infeasibility_nodes.png
        c4_minimax_infeasibility_depth.png

--------------------------------------------------------------------------------
EVALUATION FAIRNESS
--------------------------------------------------------------------------------

Role fairness:

Playing first gives a major advantage in both Tic-Tac-Toe and Connect 4. To avoid
inflated results, agents are evaluated as both Player 1 and Player 2. This makes
p1_win_rate, p2_win_rate, and first_mover_advantage important evaluation metrics.

Tic-Tac-Toe interpretation:

A 100 percent draw rate against a competent default opponent usually indicates
optimal or near-optimal play. It should not automatically be treated as poor
performance.

Random-opponent caution:

A high win rate against a random opponent does not guarantee strong performance
against the default opponent or against stronger agents.

Reduced-board caution:

The 4x5 Connect 4 Q-Learning results are useful for demonstrating the learning
mechanics of tabular Q-Learning, but they are not directly comparable to full
6x7 Connect 4 agents.

--------------------------------------------------------------------------------
ALGORITHM ROLE MAPPING
--------------------------------------------------------------------------------

Connect 4 infeasibility demo:

    c4_minimax_infeasible_agent.py       Plain Minimax, no pruning
    c4_alphabeta_infeasible_agent.py     Alpha-Beta with center-first ordering
    experiments/connect4_infeasibility.py Time-budgeted infeasibility experiment

Practical Connect 4 play:

    c4_depthlimited_alphabeta_agent.py

This uses depth-limited Alpha-Beta search with:

    depth = 5
    center-column bonus
    window evaluation for 4-in-row, 3-in-row, and 2-in-row patterns

Reduced-board Connect 4 Q-Learning:

    qlearning_c4_reduced_agent.py

This is trained and evaluated on the isolated 4x5 board.

Tic-Tac-Toe agents:

    ttt_minimax_agent.py                 Full Minimax with depth-sensitive scoring
    ttt_alphabeta_agent.py               Alpha-Beta with the same scoring
    qlearning_ttt_agent.py               Tabular Q-Learning over the full 3x3 state space
    dqn_ttt_agent.py                     DQN with 9 -> 128 -> 64 -> 9 architecture

--------------------------------------------------------------------------------
NOTES ON WINNER TRACKING
--------------------------------------------------------------------------------

The play_game() function in experiments/run_match.py maps board player identity
back to the correct agent identity. This is important because starting player
can alternate, so board player 1 is not always the same algorithm across games.

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------

This project compares how classical game-tree search and reinforcement learning
approaches behave under different game sizes and computational constraints.

Key takeaway:

    Minimax and Alpha-Beta are strong and explainable on small games such as
    Tic-Tac-Toe, but full-width search becomes infeasible for Connect 4.
    Reinforcement learning methods scale differently, but require careful
    training design, role alternation, opponent curriculum, and fair evaluation.
