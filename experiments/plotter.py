"""
experiments/plotter.py
───────────────────────────────────────────────────────────────────────────
Phase 5 requirement: Graph generation and empirical conclusions.

Reads saved experiment results to generate report-ready graphs.

Usage:
    python3 experiments/plotter.py
    python3 -m experiments.plotter
"""

import argparse
import os
import sys
import csv
import warnings
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GRAPHS_DIR  = os.path.join(os.path.dirname(__file__), "graphs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(GRAPHS_DIR, exist_ok=True)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: 'pandas' and 'matplotlib' not installed. Plotting disabled.")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot-path helper  (prevents output files from overwriting previous runs)
# ─────────────────────────────────────────────────────────────────────────────

# Set by main() before any plot function is called.
_PLOT_TAG: str = ""


def _plot_path(basename: str) -> str:
    """Return the full path for a plot file, inserting _PLOT_TAG before .png.

    Example: _plot_path("ttt_vs_default_winrate.png") with _PLOT_TAG="20240404_120000"
             → experiments/graphs/ttt_vs_default_winrate_20240404_120000.png
    """
    if _PLOT_TAG:
        stem, ext = os.path.splitext(basename)
        return os.path.join(GRAPHS_DIR, f"{stem}_{_PLOT_TAG}{ext}")
    return os.path.join(GRAPHS_DIR, basename)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_csv(results_dir, prefix):
    """Return the most recent CSV whose name starts with <prefix>_.
    Scans newest-to-oldest; prefers files with schema_version=='v2' and the
    required columns, but falls back to the most recent file if none qualify.
    """
    import glob
    matches = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.csv")),
                     reverse=True)   # newest first (timestamp in name)
    return matches[0] if matches else None


def find_latest_compatible_csv(results_dir, prefix, required_cols=None, schema_version="v2"):
    """Return newest compatible CSV only.

    Compatibility requires:
    - readable CSV
    - matching schema_version
    - all required columns present

    If no compatible file exists, return None.
    """
    import glob
    matches = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.csv")), reverse=True)

    for path in matches:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if schema_version:
            if "schema_version" not in df.columns:
                continue
            if str(df["schema_version"].iloc[0]).strip() != schema_version:
                continue

        if required_cols and not all(c in df.columns for c in required_cols):
            continue

        return path

    warnings.warn(
        f"No compatible file found for prefix '{prefix}' "
        f"(schema_version={schema_version}, required_cols={required_cols})."
    )
    return None


def find_all_compatible_csvs(results_dir, prefix, required_cols=None, schema_version="v2"):
    """Return ALL compatible CSV paths (newest first) matching the same rules as
    find_latest_compatible_csv.  Used by --aggregate mode to concatenate runs."""
    import glob
    matches = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.csv")), reverse=True)
    compatible = []
    for path in matches:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if schema_version:
            if "schema_version" not in df.columns:
                continue
            if str(df["schema_version"].iloc[0]).strip() != schema_version:
                continue
        if required_cols and not all(c in df.columns for c in required_cols):
            continue
        compatible.append(path)
    if not compatible:
        warnings.warn(
            f"No compatible files found for prefix '{prefix}' "
            f"(schema_version={schema_version}, required_cols={required_cols})."
        )
    return compatible


def load_csv(filepath):
    if filepath is None or not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    return pd.read_csv(filepath)


def check_board_config_consistency(df, graph_name, expected_board_config=None):
    configs = df["board_config"].unique()
    if len(configs) > 1:
        if expected_board_config:
            warnings.warn(
                f"[{graph_name}] Mixed board_configs found: {configs}. "
                f"Expected: {expected_board_config}. Excluding incompatible rows."
            )
            df = df[df["board_config"] == expected_board_config]
        else:
            warnings.warn(
                f"[{graph_name}] Mixed board_configs found: {configs}. "
                "This may lead to misleading comparison charts."
            )
    return df


def validate_columns(df, required_columns, graph_name):
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"  [{graph_name}] Skipping — missing columns: {missing}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  1. VS Default Comparison Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_vs_default_comparison(df, game_name, board_config, output_prefix):
    """Win-rate bar chart + stacked outcome bar chart."""
    if df is None or len(df) == 0:
        print(f"  No data for {game_name} vs Default")
        return
    required = ["agent", "win_rate", "draw_rate", "loss_rate", "board_config", "n_games"]
    if not validate_columns(df, required, f"{game_name} vs Default"):
        return

    df = check_board_config_consistency(df, f"{game_name} vs Default", board_config)
    if df is None or len(df) == 0:
        return

    df = df.sort_values("win_rate", ascending=False)
    agents     = df["agent"].tolist()
    win_rates  = df["win_rate"].tolist()
    draw_rates = df["draw_rate"].tolist()
    loss_rates = df["loss_rate"].tolist()
    n_games    = df["n_games"].tolist()

    # Compute 95% CI half-widths: ±1.96*sqrt(p*(1-p)/n)
    ci_half = [1.96 * np.sqrt(max(p * (1 - p), 0) / max(n, 1))
               for p, n in zip(win_rates, n_games)]

    # Figure 1: Win Rate Bar Chart with CI error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agents)))
    bars = ax.bar(agents, [w * 100 for w in win_rates], color=colors,
                  yerr=[ci * 100 for ci in ci_half],
                  error_kw=dict(ecolor='black', capsize=4, elinewidth=1.5))
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title(f"{game_name} ({board_config}): Win Rate vs Default Opponent\n"
                 f"Error bars = 95% CI (±1.96√(p(1−p)/n))", fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    for bar, wr in zip(bars, win_rates):
        ax.annotate(f'{wr*100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _out = _plot_path(f"{output_prefix}_vs_default_winrate.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")

    # Figure 2: Stacked outcome bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agents))
    w = 0.6
    ax.bar(x, [v * 100 for v in win_rates],  w, label='Win',  color='#2ecc71')
    ax.bar(x, [v * 100 for v in draw_rates], w,
           bottom=[v * 100 for v in win_rates], label='Draw', color='#f1c40f')
    ax.bar(x, [v * 100 for v in loss_rates], w,
           bottom=[(w_ + d) * 100 for w_, d in zip(win_rates, draw_rates)],
           label='Loss', color='#e74c3c')
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title(f"{game_name} ({board_config}): Game Outcomes vs Default Opponent", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _out = _plot_path(f"{output_prefix}_vs_default_outcomes.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Cross-Play Heatmaps  (win rate + draw rate dual subplot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_crossplay_heatmap(df, game_name, board_config, output_prefix):
    """Two-subplot heatmap: left = win rate, right = draw rate."""
    if df is None or len(df) == 0:
        print(f"  No data for {game_name} Cross-Play")
        return
    if not validate_columns(df, ["agent", "opponent", "win_rate", "draw_rate", "board_config"],
                             f"{game_name} Cross-Play"):
        return

    df = check_board_config_consistency(df, f"{game_name} Cross-Play", board_config)
    if df is None or len(df) == 0:
        return

    # Sort agents consistently
    agent_order = sorted(df["agent"].unique())

    win_pivot  = df.pivot_table(index="agent", columns="opponent",
                                values="win_rate",  aggfunc="mean").reindex(
                    index=agent_order, columns=agent_order)
    draw_pivot = df.pivot_table(index="agent", columns="opponent",
                                values="draw_rate", aggfunc="mean").reindex(
                    index=agent_order, columns=agent_order)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, pivot, title_suffix, cmap in [
        (axes[0], win_pivot,  "Win Rate (%)",  "RdYlGn"),
        (axes[1], draw_pivot, "Draw Rate (%)", "YlOrBr"),
    ]:
        vals = pivot.values * 100
        im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(title_suffix, rotation=-90, va="bottom", fontsize=11)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = vals[i, j]
                text_color = "black" if 20 < v < 80 else "white"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color=text_color, fontsize=7)
        ax.set_title(f"{title_suffix}", fontsize=12)
        ax.set_xlabel("Opponent", fontsize=11)
        ax.set_ylabel("Agent", fontsize=11)

    fig.suptitle(
        f"{game_name} ({board_config}): Cross-Play Matrix\n"
        f"Row = Agent, Column = Opponent",
        fontsize=13
    )
    plt.tight_layout()
    _out = _plot_path(f"{output_prefix}_crossplay_heatmap.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Overall Algorithm Summary  (built from vs_default data)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_summary(ttt_vsd_df, c4_vsd_df,
                         ttt_crossplay_df=None, c4_crossplay_df=None):
    """4-metric grouped bar summary: vs-default win rate, cross-play win rate,
    FMA, and normalised decision time per move — one group per agent.

    Falls back gracefully when crossplay data is absent.
    """
    if (ttt_vsd_df is None or len(ttt_vsd_df) == 0) and \
       (c4_vsd_df  is None or len(c4_vsd_df)  == 0):
        print("  No data for Overall Summary")
        return

    def _build_summary(vsd_df, cp_df, std_cfg, game_label):
        """Return a DataFrame with per-agent summary metrics."""
        if vsd_df is None or len(vsd_df) == 0:
            return None
        sub = vsd_df[vsd_df["board_config"] == std_cfg].copy()
        if len(sub) == 0:
            sub = vsd_df.copy()
        if len(sub) == 0:
            return None

        rows = []
        for _, r in sub.iterrows():
            agent = r["agent"]
            vs_wr = r.get("win_rate", np.nan)

            # Cross-play avg win rate
            cp_wr = np.nan
            if cp_df is not None and len(cp_df) > 0 and "agent" in cp_df.columns:
                cp_sub = cp_df[(cp_df["board_config"] == std_cfg) &
                               (cp_df["agent"] == agent)]
                if len(cp_sub) > 0:
                    cp_wr = cp_sub["win_rate"].mean()

            # FMA from crossplay
            fma = np.nan
            if cp_df is not None and len(cp_df) > 0:
                cp_sub = cp_df[(cp_df["board_config"] == std_cfg) &
                               (cp_df["agent"] == agent)]
                if len(cp_sub) > 0 and "first_mover_advantage" in cp_df.columns:
                    fma = cp_sub["first_mover_advantage"].mean()
                elif len(cp_sub) > 0 and {"p1_win_rate", "p2_win_rate"} <= set(cp_df.columns):
                    fma = (cp_sub["p1_win_rate"] - cp_sub["p2_win_rate"]).mean()

            # Decision time per move (log-normalised later)
            t_col = "avg_agent_time_ms_per_move"
            t_val = r.get(t_col, np.nan) if t_col in sub.columns else np.nan

            rows.append({
                "agent": agent,
                "vs_default_wr": vs_wr,
                "cp_win_rate":   cp_wr,
                "fma":           fma,
                "time_per_move": t_val,
            })

        return pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    configs = [
        (axes[0], ttt_vsd_df, ttt_crossplay_df, "3x3", "TicTacToe"),
        (axes[1], c4_vsd_df,  c4_crossplay_df,  "6x7", "Connect 4"),
    ]

    for ax, vsd_df, cp_df, std_cfg, game_name in configs:
        summary = _build_summary(vsd_df, cp_df, std_cfg, game_name)
        if summary is None or len(summary) == 0:
            ax.text(0.5, 0.5, f"No {game_name} data",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(game_name)
            continue

        # Log-normalise time so it fits on 0–1 scale alongside rates
        t_vals = summary["time_per_move"].replace(0, np.nan).dropna()
        if len(t_vals) > 0:
            log_min = np.log10(t_vals.min() + 1e-9)
            log_max = np.log10(t_vals.max() + 1e-9)
            denom   = max(log_max - log_min, 1e-9)
            summary["time_norm"] = summary["time_per_move"].apply(
                lambda v: (np.log10(v + 1e-9) - log_min) / denom
                if pd.notnull(v) and v > 0 else np.nan
            )
        else:
            summary["time_norm"] = np.nan

        agents  = summary["agent"].tolist()
        n       = len(agents)
        x       = np.arange(n)
        w       = 0.2
        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
        metrics = [
            ("vs_default_wr", "VS-Default WR",    "#2980b9"),
            ("cp_win_rate",   "Cross-Play WR",     "#27ae60"),
            ("fma",           "FMA (P1−P2)",        "#e67e22"),
            ("time_norm",     "Time/Move (norm.)",  "#8e44ad"),
        ]

        for (col, label, color), off in zip(metrics, offsets):
            vals = summary[col].tolist() if col in summary.columns else [np.nan] * n
            # Replace NaN with 0 for bars, mark with hatching
            bar_vals = [v * 100 if pd.notnull(v) and col != "time_norm"
                        else (v * 100 if pd.notnull(v) else 0)
                        for v in vals]
            ax.bar(x + off, bar_vals, w, label=label, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Score (%, or normalised)", fontsize=11)
        ax.set_title(f"{game_name} — 4-Metric Summary", fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)

    plt.suptitle("Overall Algorithm Performance Summary\n"
                 "FMA = First-Mover Advantage  |  Time = log-normalised",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _out = _plot_path("overall_algorithm_summary.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  4. RL Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_rl_training_curves():
    print("\nGenerating RL Training Curves...")

    rl_configs = [
        {"csv": find_latest_compatible_csv(RESULTS_DIR, "rl_training_metrics_ttt_qlearning",
                                           required_cols=["episode", "win_rate"]),
         "game": "TicTacToe", "algorithm": "Q-Learning",
         "board_config": "3x3", "output_prefix": "ttt_qlearning", "has_loss": False},
        {"csv": find_latest_compatible_csv(RESULTS_DIR, "rl_training_metrics_ttt_dqn",
                                           required_cols=["episode", "win_rate", "loss"]),
         "game": "TicTacToe", "algorithm": "DQN",
         "board_config": "3x3", "output_prefix": "ttt_dqn", "has_loss": True},
        {"csv": find_latest_compatible_csv(RESULTS_DIR, "rl_training_metrics_c4_qlearning",
                                           required_cols=["episode", "win_rate"]),
         "game": "Connect4",  "algorithm": "Q-Learning",
         "board_config": "4x5", "output_prefix": "c4_qlearning", "has_loss": False},
        {"csv": find_latest_compatible_csv(RESULTS_DIR, "rl_training_metrics_c4_dqn",
                                           required_cols=["episode", "win_rate", "loss"]),
         "game": "Connect4",  "algorithm": "DQN",
         "board_config": "6x7", "output_prefix": "c4_dqn", "has_loss": True},
    ]

    for cfg in rl_configs:
        csv_path = cfg["csv"]
        if csv_path is None or not os.path.exists(csv_path):
            print(f"  Warning: RL training CSV not found: {cfg['output_prefix']}")
            continue

        df = load_csv(csv_path)
        if df is None or len(df) == 0:
            continue

        df = check_board_config_consistency(
            df, f"{cfg['game']} {cfg['algorithm']}", cfg["board_config"])
        if df is None or len(df) == 0:
            continue

        eval_opponent = df["eval_opponent"].iloc[0] if "eval_opponent" in df.columns else "Unknown"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["episode"], df["win_rate"]  * 100, label="Win Rate (overall)",
                color='green',  linewidth=2)
        ax.plot(df["episode"], df["draw_rate"] * 100, label="Draw Rate",
                color='orange', linewidth=2, alpha=0.7)
        ax.plot(df["episode"], df["loss_rate"] * 100, label="Loss Rate",
                color='red',    linewidth=2, alpha=0.7)
        # Role-conditioned lines if present (v2 schema)
        if "p1_win_rate" in df.columns:
            ax.plot(df["episode"], df["p1_win_rate"] * 100, label="Win Rate as P1",
                    color='#1a9850', linewidth=1.2, linestyle='--', alpha=0.8)
        if "p2_win_rate" in df.columns:
            ax.plot(df["episode"], df["p2_win_rate"] * 100, label="Win Rate as P2",
                    color='#91cf60', linewidth=1.2, linestyle=':', alpha=0.8)
        # Generalization line if present
        if "eval_win_rate_semi" in df.columns:
            ax.plot(df["episode"], df["eval_win_rate_semi"] * 100,
                    label="Win Rate vs Semi (gen.)",
                    color='#756bb1', linewidth=1.5, linestyle='-.', alpha=0.85)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        train_opp = df["train_opponent"].iloc[0] if "train_opponent" in df.columns else eval_opponent
        ax.set_title(
            f"{cfg['game']} ({cfg['board_config']}) {cfg['algorithm']} Training\n"
            f"Train opponent: {train_opp}  |  Role alternation: enabled",
            fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        _out = _plot_path(f"{cfg['output_prefix']}_training_winrate.png")
        plt.savefig(_out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.basename(_out)}")

        if cfg["has_loss"] and "loss" in df.columns:
            loss_data = df["loss"].dropna().astype(float)
            if len(loss_data) > 0:
                window   = max(1, len(loss_data) // 100)
                smoothed = loss_data.rolling(window=window, min_periods=1).mean()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df["episode"], smoothed, label="Loss (Smoothed)", color='red', linewidth=2)
                ax.set_xlabel("Episode", fontsize=12)
                ax.set_ylabel("Loss", fontsize=12)
                ax.set_title(
                    f"{cfg['game']} ({cfg['board_config']}) DQN Training Loss\n"
                    f"Evaluation vs {eval_opponent}", fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                plt.tight_layout()
                _out = _plot_path(f"{cfg['output_prefix']}_loss.png")
                plt.savefig(_out, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  5. Connect 4 Infeasibility Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_infeasibility():
    print("\nGenerating Connect 4 Infeasibility Charts...")

    infeasibility_csv = find_latest_csv(RESULTS_DIR, "c4_infeasibility")
    if infeasibility_csv is None or not os.path.exists(infeasibility_csv):
        print(f"  Warning: Infeasibility CSV not found")
        return

    df = load_csv(infeasibility_csv)
    if df is None or len(df) == 0:
        return

    colors = ['#3498db', '#e74c3c']

    # Plot: Nodes Expanded
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["algorithm"], df["nodes_expanded"], color=colors)
    ax.set_ylabel("Nodes Expanded (log scale)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title("Connect 4 (6×7): Nodes Expanded Under Time Budget", fontsize=14)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    for bar, nodes in zip(bars, df["nodes_expanded"]):
        ax.annotate(f'{int(nodes):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    _out = _plot_path("c4_minimax_infeasibility_nodes.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")

    # Plot: Max Depth Reached
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["algorithm"], df["max_depth_reached"], color=colors)
    ax.set_ylabel("Max Depth Reached (plies)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title("Connect 4 (6×7): Max Depth Reached Under Time Budget", fontsize=14)
    ax.set_ylim(0, 45)
    ax.axhline(y=42, color='red', linestyle='--', alpha=0.5, label='Full depth (42 plies)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar, depth in zip(bars, df["max_depth_reached"]):
        ax.annotate(f'{int(depth)}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    _out = _plot_path("c4_minimax_infeasibility_depth.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Role-Stratified Bar Charts  (P1 vs P2 outcomes per agent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_role_stratified(df, game_name, board_config, output_prefix):
    """Grouped stacked bars: P1 role (left) and P2 role (right) per agent."""
    if df is None or len(df) == 0:
        print(f"  No data for {game_name} role-stratified chart")
        return

    required = ["agent", "board_config",
                "p1_win_rate", "p1_draw_rate", "p1_loss_rate",
                "p2_win_rate", "p2_draw_rate", "p2_loss_rate"]
    if not validate_columns(df, required, f"{game_name} Role-Stratified"):
        return

    df = check_board_config_consistency(df, f"{game_name} Role-Stratified", board_config)
    if df is None or len(df) == 0:
        return

    df = df.sort_values("win_rate", ascending=False)
    agents = df["agent"].tolist()
    n      = len(agents)
    x      = np.arange(n)
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 7))

    p1_w = df["p1_win_rate"].tolist()
    p1_d = df["p1_draw_rate"].tolist()
    p1_l = df["p1_loss_rate"].tolist()
    p2_w = df["p2_win_rate"].tolist()
    p2_d = df["p2_draw_rate"].tolist()
    p2_l = df["p2_loss_rate"].tolist()

    # P1 stacked bars (solid)
    ax.bar(x - width / 2, [v * 100 for v in p1_w], width,
           label='P1 Win',  color='#2ecc71')
    ax.bar(x - width / 2, [v * 100 for v in p1_d], width,
           bottom=[v * 100 for v in p1_w],
           label='P1 Draw', color='#f1c40f')
    ax.bar(x - width / 2, [v * 100 for v in p1_l], width,
           bottom=[(w + d) * 100 for w, d in zip(p1_w, p1_d)],
           label='P1 Loss', color='#e74c3c')

    # P2 stacked bars (hatched)
    ax.bar(x + width / 2, [v * 100 for v in p2_w], width,
           label='P2 Win',  color='#27ae60', hatch='//')
    ax.bar(x + width / 2, [v * 100 for v in p2_d], width,
           bottom=[v * 100 for v in p2_w],
           label='P2 Draw', color='#d4ac0d', hatch='//')
    ax.bar(x + width / 2, [v * 100 for v in p2_l], width,
           bottom=[(w + d) * 100 for w, d in zip(p2_w, p2_d)],
           label='P2 Loss', color='#c0392b', hatch='//')

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title(
        f"{game_name} ({board_config}): Outcomes by Role vs Default\n"
        f"Left bar = As P1 (first mover)   Right bar = As P2",
        fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _out = _plot_path(f"{output_prefix}_vs_default_role_stratified.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. First-Mover Advantage Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fma(df, game_name, board_config, output_prefix):
    """Horizontal bar: FMA = P1 win rate − P2 win rate per agent (from crossplay)."""
    if df is None or len(df) == 0:
        print(f"  No data for {game_name} FMA chart")
        return

    required = ["agent", "board_config",
                "p1_wins", "p1_games", "p2_wins", "p2_games"]
    if not validate_columns(df, required, f"{game_name} FMA"):
        return

    df = check_board_config_consistency(df, f"{game_name} FMA", board_config)
    if df is None or len(df) == 0:
        return

    # Aggregate totals per agent across all opponents, then compute rates
    agg = df.groupby("agent").agg(
        p1_wins_total=("p1_wins",  "sum"),
        p1_games_total=("p1_games", "sum"),
        p2_wins_total=("p2_wins",  "sum"),
        p2_games_total=("p2_games", "sum"),
    ).reset_index()

    agg["p1_wr"] = agg["p1_wins_total"] / agg["p1_games_total"].clip(lower=1)
    agg["p2_wr"] = agg["p2_wins_total"] / agg["p2_games_total"].clip(lower=1)
    agg["fma"]   = agg["p1_wr"] - agg["p2_wr"]
    agg = agg.sort_values("fma", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.8)))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in agg["fma"]]
    bars   = ax.barh(agg["agent"], agg["fma"] * 100, color=colors)
    ax.axvline(x=0, color='black', linewidth=1.2)

    for bar, val in zip(bars, agg["fma"]):
        offset = 3 if val >= 0 else -3
        ha     = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val * 100:+.1f}%',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha=ha, va='center', fontsize=9)

    ax.set_xlabel("First-Mover Advantage (%)\n(P1 win rate − P2 win rate)", fontsize=12)
    ax.set_title(
        f"{game_name} ({board_config}): First-Mover Advantage\n"
        f"Positive = P1 (first mover) advantage   Negative = P2 advantage",
        fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _out = _plot_path(f"{output_prefix}_fma.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  8. Speed vs Quality Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_vs_quality(ttt_df, c4_df):
    """Scatter: per-move decision time (log x) vs win rate (y), one point per agent×game."""
    col_time = "avg_agent_time_ms_per_move"
    col_wr   = "win_rate"

    frames = []
    for df, game_label, std_cfg, marker in [
        (ttt_df, "TTT", "3x3", "o"),
        (c4_df,  "C4",  "6x7", "s"),
    ]:
        if df is None or len(df) == 0:
            continue
        if not validate_columns(df, [col_time, col_wr, "agent", "board_config"],
                                f"Speed vs Quality ({game_label})"):
            continue
        sub = df[df["board_config"] == std_cfg][[col_time, col_wr, "agent"]].copy()
        sub["game"]   = game_label
        sub["marker"] = marker
        frames.append(sub)

    if not frames:
        print("  Skipping speed_vs_quality (no data with required columns)")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    combined = pd.concat(frames, ignore_index=True)

    for _, row in combined.iterrows():
        ax.scatter(row[col_time], row[col_wr] * 100,
                   marker=row["marker"], s=120,
                   label=f"{row['agent']} ({row['game']})")
        ax.annotate(f"{row['agent']}\n({row['game']})",
                    xy=(row[col_time], row[col_wr] * 100),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel("Avg Decision Time per Move (ms, log scale)", fontsize=12)
    ax.set_ylabel("Win Rate vs Default (%)", fontsize=12)
    ax.set_title("Speed vs Quality: Decision Time per Move vs Win Rate\n"
                 "○ = Tic-Tac-Toe   □ = Connect 4", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _out = _plot_path("speed_vs_quality.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  9. Decision Time Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_time(ttt_df, c4_df):
    """Grouped bar chart: per-move decision time for each agent, TTT vs C4, log scale."""
    col_time = "avg_agent_time_ms_per_move"

    def extract(df, std_cfg, game_label):
        if df is None or len(df) == 0:
            return {}
        if not validate_columns(df, [col_time, "agent", "board_config"],
                                f"Decision Time ({game_label})"):
            return {}
        sub = df[df["board_config"] == std_cfg][["agent", col_time]].copy()
        return dict(zip(sub["agent"], sub[col_time]))

    ttt_times = extract(ttt_df, "3x3", "TTT")
    c4_times  = extract(c4_df,  "6x7", "C4")

    all_agents = sorted(set(list(ttt_times.keys()) + list(c4_times.keys())))
    if not all_agents:
        print("  Skipping decision_time_comparison (no data)")
        return

    x     = np.arange(len(all_agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(all_agents) * 1.4), 6))
    ttt_vals = [ttt_times.get(a, np.nan) for a in all_agents]
    c4_vals  = [c4_times.get(a,  np.nan) for a in all_agents]

    ax.bar(x - width / 2, ttt_vals, width, label='Tic-Tac-Toe (3×3)',
           color='#3498db', alpha=0.85)
    ax.bar(x + width / 2, c4_vals,  width, label='Connect 4 (6×7)',
           color='#e67e22', alpha=0.85)

    ax.set_yscale('log')
    ax.set_ylabel("Avg Decision Time per Move (ms, log scale)", fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title("Decision Time per Move: All Agents × Games", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(all_agents, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _out = _plot_path("decision_time_comparison.png")
    plt.savefig(_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(_out)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not HAS_PLOTTING:
        print("Error: pandas and matplotlib required. pip install pandas matplotlib")
        return

    parser = argparse.ArgumentParser(
        description="Generate all report graphs from saved CSVs."
    )
    parser.add_argument(
        "--tag", default=None, metavar="TAG",
        help="String appended to every output plot filename to prevent overwriting "
             "previous runs (default: current timestamp YYYYMMDD_HHMMSS)."
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Concatenate ALL matching v2 CSVs instead of only the latest. "
             "Useful for comparing or summarising results across multiple runs."
    )
    args = parser.parse_args()

    global _PLOT_TAG
    _PLOT_TAG = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(" Phase 5: Result Analysis & Graphing Script")
    print("=" * 60)
    print(f"  Plot tag  : {_PLOT_TAG}")
    print(f"  CSV mode  : {'aggregate (all matching runs)' if args.aggregate else 'latest only'}")

    _vsd_cols = ["agent", "win_rate", "game", "board_config",
                 "p1_win_rate", "p2_win_rate", "first_mover_advantage",
                 "avg_agent_time_ms_per_move"]
    _cp_cols  = ["agent", "opponent", "win_rate", "game", "board_config",
                 "p1_wins", "p1_games", "p2_wins", "p2_games"]

    if args.aggregate:
        def _load_agg(prefix, required_cols):
            paths = find_all_compatible_csvs(RESULTS_DIR, prefix, required_cols=required_cols)
            if not paths:
                return None
            print(f"  [aggregate] {prefix}: loading {len(paths)} file(s)")
            return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        ttt_vs_default = _load_agg("vs_default_ttt", _vsd_cols)
        c4_vs_default  = _load_agg("vs_default_c4",  _vsd_cols)
        ttt_crossplay  = _load_agg("crossplay_ttt",  _cp_cols)
        c4_crossplay   = _load_agg("crossplay_c4",   _cp_cols)
    else:
        ttt_vs_default = load_csv(find_latest_compatible_csv(
            RESULTS_DIR, "vs_default_ttt", required_cols=_vsd_cols))
        c4_vs_default  = load_csv(find_latest_compatible_csv(
            RESULTS_DIR, "vs_default_c4",  required_cols=_vsd_cols))
        ttt_crossplay  = load_csv(find_latest_compatible_csv(
            RESULTS_DIR, "crossplay_ttt",  required_cols=_cp_cols))
        c4_crossplay   = load_csv(find_latest_compatible_csv(
            RESULTS_DIR, "crossplay_c4",   required_cols=_cp_cols))

    # Pre-filter C4 vs_default: separate full-board (6x7) from reduced-board (4x5)
    c4_vs_default_full    = c4_vs_default[c4_vs_default["board_config"] == "6x7"] \
                            if c4_vs_default is not None else None
    c4_vs_default_reduced = c4_vs_default[c4_vs_default["board_config"] == "4x5"] \
                            if c4_vs_default is not None else None

    # 1. VS Default Charts
    print("\n1. Generating VS Default Charts...")
    plot_vs_default_comparison(ttt_vs_default,        "TicTacToe",                    "3x3", "ttt")
    plot_vs_default_comparison(c4_vs_default_full,    "Connect4",                     "6x7", "c4")
    plot_vs_default_comparison(c4_vs_default_reduced, "Connect4 (4×5 Reduced Board)", "4x5", "c4_reduced")

    # 2. Cross-Play Heatmaps (win + draw dual subplot)
    print("\n2. Generating Cross-Play Heatmaps...")
    plot_crossplay_heatmap(ttt_crossplay, "TicTacToe", "3x3", "ttt")
    plot_crossplay_heatmap(c4_crossplay,  "Connect4",  "6x7", "c4")

    # 3. Overall Summary (built from vs_default + crossplay data)
    print("\n3. Generating Overall Summary...")
    plot_overall_summary(ttt_vs_default, c4_vs_default_full,
                         ttt_crossplay_df=ttt_crossplay,
                         c4_crossplay_df=c4_crossplay)

    # 4. RL Training Curves
    plot_rl_training_curves()

    # 5. Infeasibility Charts
    plot_infeasibility()

    # 6. Role-Stratified Charts
    print("\n6. Generating Role-Stratified Charts...")
    plot_role_stratified(ttt_vs_default,     "TicTacToe", "3x3", "ttt")
    plot_role_stratified(c4_vs_default_full, "Connect4",  "6x7", "c4")

    # 7. First-Mover Advantage
    print("\n7. Generating First-Mover Advantage Charts...")
    plot_fma(ttt_crossplay, "TicTacToe", "3x3", "ttt")
    plot_fma(c4_crossplay,  "Connect4",  "6x7", "c4")

    # 8. Speed vs Quality
    print("\n8. Generating Speed vs Quality Chart...")
    plot_speed_vs_quality(ttt_vs_default, c4_vs_default_full)

    # 9. Decision Time Comparison
    print("\n9. Generating Decision Time Comparison...")
    plot_decision_time(ttt_vs_default, c4_vs_default_full)

    print(f"\n{'='*60}")
    print(f"All graphs saved to: {GRAPHS_DIR}/")
    print(f"Plot tag: {_PLOT_TAG}  (filenames end with _{_PLOT_TAG}.png)")
    print("=" * 60)


if __name__ == "__main__":
    main()
