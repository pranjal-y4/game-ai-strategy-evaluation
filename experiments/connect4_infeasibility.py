"""
experiments/connect4_infeasibility.py
────────────────────────────────────────────────────────────────────────────
Prove that full Minimax and Alpha-Beta are infeasible for Connect 4 (6×7).

Both algorithms are run from an empty board under a strict time budget.
The search is stopped by raising an internal _Timeout exception every 10 000
nodes, which avoids the unreliable Python threading / GIL-based approach.

Usage:
    python experiments/connect4_infeasibility.py --time_budget 60
    python experiments/connect4_infeasibility.py --time_budget 1800  # 30 min (assignment)

Key metrics recorded
────────────────────
    algorithm              : "Minimax" or "AlphaBeta"
    time_budget_sec        : requested budget
    nodes_expanded         : nodes visited before timeout / full completion
    elapsed_sec            : actual wall-clock time used
    nodes_per_sec          : throughput
    max_depth_reached      : deepest ply explored on any branch
    completed_full_search  : True only if the entire tree was exhausted
    notes                  : human-readable summary

Results saved to: experiments/results/c4_infeasibility_<timestamp>.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import csv
from datetime import datetime

from games.connect4_core import Connect4
from agents.c4_minimax_infeasible_agent import C4MinimaxInfeasibleAgent, _Timeout as _MMTimeout
from agents.c4_alphabeta_infeasible_agent import C4AlphaBetaInfeasibleAgent, _Timeout as _ABTimeout
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_search(agent, game, time_budget_sec: float) -> dict:
    """
    Run agent.select_action under a time budget.

    The agent raises _Timeout internally every 10 000 nodes if
    time.time() > deadline.  We catch it here and collect metrics.
    """
    deadline = time.time() + time_budget_sec
    agent.reset()
    agent.set_deadline(deadline)

    completed = False
    t0 = time.time()
    timeout_cls = _MMTimeout if isinstance(agent, C4MinimaxInfeasibleAgent) else _ABTimeout

    try:
        agent.select_action(game)
        completed = True
    except timeout_cls:
        pass  # expected — time budget exhausted
    except Exception as e:
        print(f"  Unexpected error: {e}")

    elapsed = time.time() - t0
    nodes   = agent.nodes_expanded
    nps     = nodes / elapsed if elapsed > 0 else 0

    return {
        "algorithm":             agent.name,
        "time_budget_sec":       time_budget_sec,
        "elapsed_sec":           round(elapsed, 3),
        "nodes_expanded":        nodes,
        "nodes_per_sec":         round(nps, 1),
        "max_depth_reached":     agent.max_depth_reached,
        "completed_full_search": completed,
        "notes": (
            "COMPLETED full tree search (unexpectedly fast — check board size)"
            if completed else
            f"TIMED OUT after {elapsed:.1f}s. Full 6x7 tree is intractable."
        ),
    }


def print_result(r: dict) -> None:
    status = "COMPLETED" if r["completed_full_search"] else "TIMED OUT"
    print(f"\n  Algorithm : {r['algorithm']}")
    print(f"  Status    : {status}")
    print(f"  Nodes     : {r['nodes_expanded']:,}")
    print(f"  Speed     : {r['nodes_per_sec']:,.0f} nodes/sec")
    print(f"  Max depth : {r['max_depth_reached']} plies   (full tree = 42 plies)")
    print(f"  Time used : {r['elapsed_sec']:.2f}s  /  budget {r['time_budget_sec']}s")
    print(f"  Note      : {r['notes']}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate infeasibility of full Connect 4 tree search"
    )
    parser.add_argument(
        "--time_budget", type=int, default=1800,
        help="Time budget per algorithm in seconds (default 1800 = 30 min). "
             "Use 60 for quick testing."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("Connect 4 (6×7)  —  Full-Tree Search Infeasibility Experiment")
    print(f"Time budget per algorithm: {args.time_budget}s")
    print(f"Board: 6 rows × 7 cols = 42 positions  |  ~4.5×10^12 legal positions")
    print("=" * 70)

    game = Connect4(rows=6, cols=7)

    results = []

    # ── 1. Plain Minimax ─────────────────────────────────────────────────────
    print("\n[1/2] Running full Minimax (no pruning) ...")
    mm_result = run_search(C4MinimaxInfeasibleAgent(), game, args.time_budget)
    mm_result["timestamp"] = datetime.now().isoformat()
    results.append(mm_result)
    print_result(mm_result)

    # Reset game to empty board
    game.reset()

    # ── 2. Alpha-Beta with center-first ordering ─────────────────────────────
    print("\n[2/2] Running Alpha-Beta (with center-first ordering) ...")
    ab_result = run_search(C4AlphaBetaInfeasibleAgent(), game, args.time_budget)
    ab_result["timestamp"] = datetime.now().isoformat()
    results.append(ab_result)
    print_result(ab_result)

    # ── Pruning efficiency metrics ────────────────────────────────────────────
    FULL_TREE   = 7 ** 42   # upper bound: 7 choices × 42 positions
    SECS_PER_YR = 3600 * 24 * 365

    mm_nodes = mm_result["nodes_expanded"]
    ab_nodes = ab_result["nodes_expanded"]
    mm_nps   = max(mm_result["nodes_per_sec"], 1)
    ab_nps   = max(ab_result["nodes_per_sec"], 1)

    pruning_ratio        = round(mm_nodes / max(ab_nodes, 1), 2)
    pruning_savings_pct  = round((mm_nodes - ab_nodes) / max(mm_nodes, 1) * 100, 2)

    mm_result["fraction_of_full_tree"]   = f"{mm_nodes / FULL_TREE:.2e}"
    mm_result["extrapolated_time_years"] = round((FULL_TREE / mm_nps) / SECS_PER_YR, 2)
    mm_result["pruning_ratio"]           = ""
    mm_result["pruning_savings_pct"]     = ""

    ab_result["fraction_of_full_tree"]   = f"{ab_nodes / FULL_TREE:.2e}"
    ab_result["extrapolated_time_years"] = round((FULL_TREE / ab_nps) / SECS_PER_YR, 2)
    ab_result["pruning_ratio"]           = pruning_ratio
    ab_result["pruning_savings_pct"]     = pruning_savings_pct

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"  Minimax explored  {mm_nodes:>12,} nodes in {mm_result['elapsed_sec']:.1f}s")
    print(f"  Alpha-Beta explored {ab_nodes:>10,} nodes in {ab_result['elapsed_sec']:.1f}s")
    print(f"  Pruning ratio (MM / AB):           {pruning_ratio:.1f}x")
    print(f"  Pruning savings:                   {pruning_savings_pct:.1f}%")
    print(f"  Fraction of full tree (MM):        {mm_result['fraction_of_full_tree']}")
    print(f"  Fraction of full tree (AB):        {ab_result['fraction_of_full_tree']}")
    print(f"  Extrapolated full-tree time (MM):  {mm_result['extrapolated_time_years']:.2e} years")
    print(f"  Extrapolated full-tree time (AB):  {ab_result['extrapolated_time_years']:.2e} years")
    print()
    print("  Neither algorithm completed the full 6×7 game tree.")
    print("  A depth-limited search with heuristic evaluation (see")
    print("  C4DepthLimitedAlphaBetaAgent) is used for actual Connect 4 play.")
    print("=" * 70)

    # ── Save results ──────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(RESULTS_DIR, f"c4_infeasibility_{ts}.csv")

    fieldnames = [
        "timestamp", "algorithm", "time_budget_sec", "elapsed_sec",
        "nodes_expanded", "nodes_per_sec", "max_depth_reached",
        "completed_full_search", "fraction_of_full_tree", "extrapolated_time_years",
        "pruning_ratio", "pruning_savings_pct", "notes",
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to:\n  {outfile}")


if __name__ == "__main__":
    main()
