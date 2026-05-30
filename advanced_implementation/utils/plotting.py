from __future__ import annotations
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT = True
except ImportError:
    _PLOT = False


def plot_training_curves(log_csv: str, out_dir: str) -> None:


    if not _PLOT:
        return
    try:
        import pandas as pd
        df = pd.read_csv(log_csv)
    except Exception:
        return

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(log_csv))[0]


    if "win_rate" in df.columns and "episode" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["episode"], df["win_rate"], label="win rate", color="steelblue")
        if "p1_win_rate" in df.columns:
            ax.plot(df["episode"], df["p1_win_rate"], "--", label="P1 wr", alpha=0.7)
        if "p2_win_rate" in df.columns:
            ax.plot(df["episode"], df["p2_win_rate"], ":", label="P2 wr", alpha=0.7)

        if "phase" in df.columns:
            switch = df[df["phase"] == "phase2_default"]
            if not switch.empty:
                ax.axvline(x=switch.iloc[0]["episode"], color="red",
                           linestyle="--", alpha=0.5, label="curriculum switch")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Training Win Rate — {base}")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_win_rate.png"), dpi=150)
        plt.close()


    if "avg_loss" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["episode"], df["avg_loss"], color="tomato")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Loss")
        ax.set_title(f"Training Loss — {base}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_loss.png"), dpi=150)
        plt.close()


    if "epsilon" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["episode"], df["epsilon"], color="orange")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title(f"Epsilon Decay — {base}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_epsilon.png"), dpi=150)
        plt.close()


def plot_crossplay_heatmap(rows: list[dict], title: str, out_path: str) -> None:

    if not _PLOT or not rows:
        return
    import numpy as np
    agents = list(dict.fromkeys([r["agent"] for r in rows]))
    n = len(agents)
    mat = np.full((n, n), np.nan)
    idx = {a: i for i, a in enumerate(agents)}
    for r in rows:
        i, j = idx.get(r["agent"], -1), idx.get(r["opponent"], -1)
        if i >= 0 and j >= 0:
            mat[i, j] = r["win_rate"]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Win Rate")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=30, ha="right")
    ax.set_yticklabels(agents)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
