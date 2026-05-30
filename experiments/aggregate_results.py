"""
experiments/aggregate_results.py
────────────────────────────────────────────────────────────────────────────
Multi-seed statistical aggregator for experiment results.

Loads all compatible v2 CSVs from the results directory, rejects files with
wrong or missing schema_version, groups by the natural keys (game, board_config,
agent, opponent), and aggregates across seeds producing mean ± std and 95% CI.

Outputs:
  experiments/results/aggregated_vs_default_<ts>.csv
  experiments/results/aggregated_crossplay_<ts>.csv

Usage:
    python experiments/aggregate_results.py
    python experiments/aggregate_results.py --input_dir experiments/results
    python experiments/aggregate_results.py --game ttt   # only TTT files
"""

import sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

RESULTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REQUIRED_SCHEMA = "v2"

# Columns to aggregate (mean/std/CI)
RATE_COLS = [
    "win_rate", "draw_rate", "loss_rate",
    "p1_win_rate", "p2_win_rate", "first_mover_advantage",
]
TIME_COLS = [
    "avg_agent_time_ms_per_move", "avg_agent_nodes_per_move",
    "avg_agent_time_ms", "avg_agent_nodes",
]
COUNT_COLS = ["n_games", "wins", "draws", "losses",
              "p1_games", "p1_wins", "p1_draws", "p1_losses",
              "p2_games", "p2_wins", "p2_draws", "p2_losses"]


def load_compatible(path: str, required_cols: list = None):
    """Load CSV and return DataFrame only if schema_version == v2 and columns match.
    Returns None with a warning if the file is incompatible.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"  Cannot read {os.path.basename(path)}: {e}")
        return None

    if "schema_version" not in df.columns:
        warnings.warn(
            f"  Skipping {os.path.basename(path)}: no schema_version column "
            f"(old format — re-run evaluation scripts to regenerate)."
        )
        return None

    schema = str(df["schema_version"].iloc[0]).strip()
    if schema != REQUIRED_SCHEMA:
        warnings.warn(
            f"  Skipping {os.path.basename(path)}: schema_version={schema!r}, "
            f"expected {REQUIRED_SCHEMA!r}."
        )
        return None

    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            warnings.warn(
                f"  Skipping {os.path.basename(path)}: missing columns {missing}."
            )
            return None

    return df


def aggregate_group(group_df: "pd.DataFrame", numeric_cols: list) -> dict:
    """Aggregate a group of rows from multiple seeds.

    Strategy:
      - raw count columns (wins, draws, losses, p1_games, etc.) are SUMMED
        across seeds, then rates are RECOMPUTED from those totals.
        This avoids mean-of-means bias when runs have different game counts.
      - timing / node metrics use mean ± std / 95% CI as usual.
    """
    result = {}
    seeds = group_df["seed"].unique() if "seed" in group_df.columns else []
    result["seed_count"] = len(seeds)
    result["seeds"]      = ",".join(str(s) for s in sorted(seeds))

    # ── mixed game count warning ───────────────────────────────────────────
    n_col = "n_games" if "n_games" in group_df.columns else ("games" if "games" in group_df.columns else None)
    if n_col:
        result["warning_mixed_game_counts"] = bool(group_df[n_col].nunique() > 1)
    else:
        result["warning_mixed_game_counts"] = False

    # ── sum raw counts and recompute derived rates ─────────────────────────
    _count_map = {
        "n_games": "n_games", "games": "games",
        "wins": "wins", "draws": "draws", "losses": "losses",
        "p1_games": "p1_games", "p1_wins": "p1_wins",
        "p1_draws": "p1_draws", "p1_losses": "p1_losses",
        "p2_games": "p2_games", "p2_wins": "p2_wins",
        "p2_draws": "p2_draws", "p2_losses": "p2_losses",
    }
    totals = {}
    for col, key in _count_map.items():
        if col in group_df.columns:
            vals = pd.to_numeric(group_df[col], errors="coerce").fillna(0)
            totals[key] = int(vals.sum())

    # Store totals
    for k, v in totals.items():
        result[k] = v

    # Recompute rates from summed counts (not mean-of-means)
    total_games = totals.get("n_games", totals.get("games", 0))
    if total_games > 0:
        result["win_rate"]  = round(totals.get("wins",   0) / total_games, 4)
        result["draw_rate"] = round(totals.get("draws",  0) / total_games, 4)
        result["loss_rate"] = round(totals.get("losses", 0) / total_games, 4)
    elif "win_rate" in group_df.columns:
        # Fallback: no raw counts available — fall back to mean-of-means with warning
        result["win_rate"]  = round(pd.to_numeric(group_df["win_rate"],  errors="coerce").mean(), 4)
        result["draw_rate"] = round(pd.to_numeric(group_df.get("draw_rate",  pd.Series(dtype=float)), errors="coerce").mean(), 4)
        result["loss_rate"] = round(pd.to_numeric(group_df.get("loss_rate",  pd.Series(dtype=float)), errors="coerce").mean(), 4)

    p1_games = totals.get("p1_games", 0)
    p2_games = totals.get("p2_games", 0)
    if p1_games > 0:
        result["p1_win_rate"]  = round(totals.get("p1_wins",   0) / p1_games, 4)
        result["p1_draw_rate"] = round(totals.get("p1_draws",  0) / p1_games, 4)
        result["p1_loss_rate"] = round(totals.get("p1_losses", 0) / p1_games, 4)
    elif "p1_win_rate" in group_df.columns:
        result["p1_win_rate"] = round(pd.to_numeric(group_df["p1_win_rate"], errors="coerce").mean(), 4)

    if p2_games > 0:
        result["p2_win_rate"]  = round(totals.get("p2_wins",   0) / p2_games, 4)
        result["p2_draw_rate"] = round(totals.get("p2_draws",  0) / p2_games, 4)
        result["p2_loss_rate"] = round(totals.get("p2_losses", 0) / p2_games, 4)
    elif "p2_win_rate" in group_df.columns:
        result["p2_win_rate"] = round(pd.to_numeric(group_df["p2_win_rate"], errors="coerce").mean(), 4)

    # Recompute FMA from recomputed P1/P2 win rates
    p1_wr = result.get("p1_win_rate")
    p2_wr = result.get("p2_win_rate")
    if p1_wr is not None and p2_wr is not None:
        result["first_mover_advantage"] = round(p1_wr - p2_wr, 4)

    # ── mean/std/CI for timing and node metrics ────────────────────────────
    for col in numeric_cols:
        # Skip count/rate columns — already handled above
        if col in _count_map or col in ("win_rate", "draw_rate", "loss_rate",
                                         "p1_win_rate", "p2_win_rate",
                                         "first_mover_advantage"):
            continue
        if col not in group_df.columns:
            continue
        vals = pd.to_numeric(group_df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        result[f"{col}_mean"] = round(float(mean), 4)
        result[f"{col}_std"]  = round(float(std),  4)
        result[f"{col}_ci95"] = round(float(ci95), 4)
        result[f"{col}_n"]    = len(vals)

    return result


def aggregate_vs_default(all_dfs: list, game_filter: str = None) -> list:
    """Aggregate vs-default CSVs across seeds."""
    required = ["agent", "opponent", "win_rate", "game", "board_config"]
    dfs = []
    for df in all_dfs:
        if not all(c in df.columns for c in required):
            continue
        dfs.append(df)

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    if game_filter:
        combined = combined[combined["game"].str.upper() == game_filter.upper()]

    # Exclude mixed-board comparisons: keep 6x7 C4 and 3x3 TTT separate
    group_keys = ["game", "board_config", "agent", "opponent"]
    numeric_cols = RATE_COLS + TIME_COLS

    rows = []
    for keys, group in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        row.update(aggregate_group(group, numeric_cols))
        rows.append(row)
    return rows


def aggregate_crossplay(all_dfs: list, game_filter: str = None) -> list:
    """Aggregate crossplay CSVs across seeds."""
    required = ["agent", "opponent", "win_rate", "game", "board_config"]
    dfs = []
    for df in all_dfs:
        if not all(c in df.columns for c in required):
            continue
        dfs.append(df)

    if not dfs:
        return []

    combined = pd.concat(dfs, ignore_index=True)
    if game_filter:
        combined = combined[combined["game"].str.upper() == game_filter.upper()]

    group_keys   = ["game", "board_config", "agent", "opponent"]
    numeric_cols = RATE_COLS + ["avg_agent1_time_ms_per_move", "avg_agent1_nodes_per_move"]

    rows = []
    for keys, group in combined.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        row.update(aggregate_group(group, numeric_cols))
        rows.append(row)
    return rows


def save_csv(rows: list, path: str) -> None:
    if not rows:
        print(f"  (no data to save to {os.path.basename(path)})")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main():
    if not HAS_PANDAS:
        print("Error: pandas and numpy required. pip install pandas numpy")
        return

    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment results"
    )
    parser.add_argument("--input_dir",  default=RESULTS_DIR)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument("--game",       choices=["ttt", "c4"], default=None,
                        help="Filter to a single game (default: both)")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nAggregating results from: {args.input_dir}")
    print(f"Schema required: {REQUIRED_SCHEMA}")
    print("-" * 60)

    # ── Load vs-default CSVs ─────────────────────────────────────────────
    vsd_files = sorted(glob.glob(os.path.join(args.input_dir, "vs_default_*.csv")))
    vsd_dfs   = []
    print(f"\nVS-Default files found: {len(vsd_files)}")
    for f in vsd_files:
        df = load_compatible(f, required_cols=["agent", "win_rate", "game"])
        if df is not None:
            print(f"  [OK] {os.path.basename(f)}  ({len(df)} rows, "
                  f"seeds={df['seed'].unique().tolist() if 'seed' in df.columns else '?'})")
            vsd_dfs.append(df)

    # ── Load crossplay CSVs ───────────────────────────────────────────────
    cp_files = sorted(glob.glob(os.path.join(args.input_dir, "crossplay_*.csv")))
    cp_dfs   = []
    print(f"\nCross-play files found: {len(cp_files)}")
    for f in cp_files:
        df = load_compatible(f, required_cols=["agent", "opponent", "win_rate", "game"])
        if df is not None:
            print(f"  [OK] {os.path.basename(f)}  ({len(df)} rows, "
                  f"seeds={df['seed'].unique().tolist() if 'seed' in df.columns else '?'})")
            cp_dfs.append(df)

    # ── Aggregate ─────────────────────────────────────────────────────────
    print(f"\nAggregating…")

    vsd_rows = aggregate_vs_default(vsd_dfs, game_filter=args.game)
    cp_rows  = aggregate_crossplay(cp_dfs,   game_filter=args.game)

    suffix = f"_{args.game}" if args.game else ""
    save_csv(vsd_rows, os.path.join(args.output_dir, f"aggregated_vs_default{suffix}_{ts}.csv"))
    save_csv(cp_rows,  os.path.join(args.output_dir, f"aggregated_crossplay{suffix}_{ts}.csv"))

    # ── Summary to terminal ───────────────────────────────────────────────
    if vsd_rows:
        df_sum = pd.DataFrame(vsd_rows)
        print(f"\n{'='*60}")
        print("VS-DEFAULT SUMMARY  (mean win_rate across seeds)")
        print(f"{'='*60}")
        show_cols = ["game", "board_config", "agent", "seed_count",
                     "win_rate_mean", "win_rate_std",
                     "p1_win_rate_mean", "p2_win_rate_mean",
                     "first_mover_advantage_mean"]
        show_cols = [c for c in show_cols if c in df_sum.columns]
        print(df_sum[show_cols].to_string(index=False))

    if cp_rows:
        df_cp = pd.DataFrame(cp_rows)
        print(f"\n{'='*60}")
        print("CROSS-PLAY SUMMARY  (mean win_rate across seeds)")
        print(f"{'='*60}")
        show_cols = ["game", "board_config", "agent", "opponent", "seed_count",
                     "win_rate_mean", "win_rate_std"]
        show_cols = [c for c in show_cols if c in df_cp.columns]
        print(df_cp[show_cols].to_string(index=False))

    if not vsd_rows and not cp_rows:
        print("\nNo compatible v2 results found.")
        print("Tip: re-run evaluation scripts to generate v2-schema CSVs:")
        print("  python experiments/evaluate_against_default.py --game ttt --games 500 --seed 42")


if __name__ == "__main__":
    main()
