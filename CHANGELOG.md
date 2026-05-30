# CHANGELOG — AI Assignment 3

## v2 (2026-04-04)

### Bug Fixes

1. **plotter.py — KeyError on `experiment_type`** (Bug 1)
   - `plot_vs_default_comparison()` and `plot_crossplay_heatmap()` filtered on a column that never existed in the CSVs. Removed the filter lines — each CSV is already typed by filename.

2. **plotter.py — KeyError on `nodes_explored`** (Bug 2)
   - Infeasibility plotter referenced `df["nodes_explored"]` but the column is `nodes_expanded`. Fixed everywhere in `plot_infeasibility()`.

3. **plotter.py — Overall summary never generated** (Bug 3)
   - `plot_overall_summary()` tried to read a non-existent `overall_summary.csv`. Rewrote to accept vs-default and crossplay DataFrames directly from `main()`, computing the 4-metric grouped bar chart (vs-default WR, cross-play WR, FMA, log-normalised time/move) in memory.

---

### Metric Additions (CSV Schema v2)

4. **Gap 1 — P1/P2 role-stratified columns** (`evaluate_against_default.py`, `evaluate_crossplay.py`)
   - Added: `p1_games`, `p1_wins`, `p1_draws`, `p1_losses`, `p1_win_rate`, `p1_draw_rate`, `p1_loss_rate`, and P2 equivalents.
   - Tracked directly in the evaluation loop using `game_num % 2 == 0` to determine agent role, matching `play_game()` alternation logic.

5. **Gap 2 — Per-move timing** (`avg_agent_time_ms_per_move`)
   - Computed as `avg_agent_time_ms / avg_game_length` (guarded against zero with `max(..., 1)`).
   - Added to both `evaluate_against_default.py` and `evaluate_crossplay.py`.

6. **Gap 3 — Per-move node count** (`avg_agent_nodes_per_move`)
   - Same calculation as timing but for nodes expanded.

7. **Gap 4 — Pruning efficiency in infeasibility CSV** (`connect4_infeasibility.py`)
   - Added columns: `pruning_ratio` (Minimax/AlphaBeta nodes), `pruning_savings_pct`, `fraction_of_full_tree` (nodes / 7^42), `extrapolated_time_years` ((7^42 / NPS) / seconds-per-year).

8. **First-mover advantage column** (`first_mover_advantage = p1_win_rate - p2_win_rate`)
   - Added to both `evaluate_against_default.py` and `evaluate_crossplay.py` output rows.

9. **Schema version column** (`schema_version = "v2"`)
   - Added to all evaluation CSVs and training CSVs for future compatibility checking.

---

### RL Training Fixes

10. **Draw detection threshold** (all 4 training scripts)
    - Changed `reward > 0` → `reward >= 1.0` to correctly distinguish wins from draws.
    - Draw reward is +0.2; the old threshold incorrectly counted draws as wins, inflating reported win rates during training.

11. **Role-split evaluation during training** (all 4 training scripts)
    - Added `_greedy_eval_split(n_each)` function that runs `n_each` games as P1 and `n_each` games as P2.
    - P2 evaluation simulates being second mover by letting the environment opponent act first before the agent's first turn.
    - Logged as `p1_win_rate` and `p2_win_rate` columns in training CSVs.

12. **Convergence milestones** (all 4 training scripts)
    - Track and print: `first_95pct_episode` (first episode where overall win rate ≥ 0.95, or -1), `peak_win_rate`, `peak_episode`.

---

### Plotter Improvements

13. **Column validation** — `validate_columns()` helper checks required columns before each plot, prints a warning and skips rather than crashing.

14. **Board-size separation** — `main()` now pre-filters C4 vs-default data into `c4_vs_default_full` (6×7) and `c4_vs_default_reduced` (4×5) before calling any plot function. Eliminates mixed-board warnings and prevents misleading cross-board comparisons.

15. **New chart: `c4_reduced_vs_default_winrate.png` / `c4_reduced_vs_default_outcomes.png`** — dedicated charts for the 4×5 Q-learning agent, clearly titled "Connect 4 (4×5 Reduced Board)".

16. **New chart: Role-stratified bar charts** (`ttt_vs_default_role_stratified.png`, `c4_vs_default_role_stratified.png`) — grouped stacked bars showing P1 and P2 outcomes separately per agent.

17. **New chart: First-mover advantage** (`ttt_fma.png`, `c4_fma.png`) — horizontal bar chart of FMA = P1 win rate − P2 win rate, aggregated across all crossplay opponents.

18. **New chart: Speed vs Quality scatter** (`speed_vs_quality.png`) — log-scale x (decision time/move) vs win rate, labelled by agent, different markers for TTT and C4.

19. **New chart: Decision time comparison** (`decision_time_comparison.png`) — grouped bar chart of per-move decision times, log scale, TTT vs C4 side by side.

20. **Confidence intervals** on win rate bar charts — 95% CI error bars: ±1.96√(p(1−p)/n).

21. **Overall summary enriched** — now uses 4 metrics (vs-default WR, cross-play WR, FMA, log-normalised time/move) and incorporates crossplay data when available.

---

### New Files

- `run_experiments.sh` — safe shell script for the full pipeline; supports `--quick` flag for fast testing.
- `CHANGELOG.md` — this file.

---

### Evaluation Fairness Notes

**First-player incentive**: Every evaluation now separately reports P1 and P2 outcomes. The `first_mover_advantage` column (P1 win rate − P2 win rate) quantifies role bias. In TTT, Minimax and AlphaBeta show ~44% FMA in crossplay, reflecting the structural advantage of moving first in an open board. A policy that collapses when playing second is not considered robust.

**Reduced-board Connect 4**: The Q-learning agent is trained and evaluated exclusively on a 4×5 board. Its results are labelled `board_config = "4x5"` in all CSVs, excluded from the 6×7 crossplay matrix, and plotted in a separate `c4_reduced_vs_default_*.png` chart. Report comparisons against 6×7 agents (AlphaBeta, DQN) must acknowledge this difference explicitly.

**Metric justification**: Win rate alone is insufficient when role, opponent strength, and compute cost differ. The 4-metric summary (vs-default WR, cross-play WR, FMA, decision time) provides a more honest ranking that penalises role-fragile and computationally expensive agents equally.
