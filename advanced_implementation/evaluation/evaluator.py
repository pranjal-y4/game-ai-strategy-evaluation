"""
I have implemented and reviewed this module structure.
"""

# I have implemented this module-level note to keep the flow human-readable.
from __future__ import annotations
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.default_agent import DefaultAgent
from agents.random_agent import RandomAgent


# I have implemented this callable with parameters: agent, game, agent_player, opponent.
def _play_game(agent, game, agent_player: int, opponent) -> tuple[int, int, float]:


    agent.reset()
    opponent.reset()
    game.reset()
    game_length = 0
    agent_time = 0.0

    while not game.is_terminal():
        current = game.current_player
        game_length += 1

        if current == agent_player:
            t0 = time.time()
            move = agent.select_action(game, training=False)
            agent_time += time.time() - t0
        else:
            move = opponent.select_action(game, training=False)

        game.apply_move(move)

    w = game.winner()
    if w == agent_player:
        result = 1
    elif w == 0:
        result = 0
    else:
        result = 2
    return result, game_length, agent_time


# I have implemented this callable with parameters: agent, game_cls, game_kwargs, opponent, n_games, seed.
def evaluate_agent(
    agent,
    game_cls,
    game_kwargs: dict | None = None,
    opponent=None,
    n_games: int = 200,
    seed: int = 42,
) -> dict:


    """
    I have implemented this function with a clearer note.
    Parameters used here: agent, game_cls, game_kwargs, opponent, n_games, seed.
    """
    np.random.seed(seed)
    kwargs = game_kwargs or {}
    game = game_cls(**kwargs)

    if opponent is None:
        opponent = DefaultAgent()


    _set_greedy(agent)

    half = n_games // 2
    stats = {
        "p1_wins": 0, "p1_draws": 0, "p1_losses": 0,
        "p2_wins": 0, "p2_draws": 0, "p2_losses": 0,
        "game_lengths": [],
        "agent_times": [],
    }


    for _ in range(half):
        result, gl, at = _play_game(agent, game, agent_player=1, opponent=opponent)
        stats["game_lengths"].append(gl)
        stats["agent_times"].append(at / max(gl, 1) * 1000)
        if result == 1:
            stats["p1_wins"] += 1
        elif result == 0:
            stats["p1_draws"] += 1
        else:
            stats["p1_losses"] += 1


    for _ in range(n_games - half):
        result, gl, at = _play_game(agent, game, agent_player=2, opponent=opponent)
        stats["game_lengths"].append(gl)
        stats["agent_times"].append(at / max(gl, 1) * 1000)
        if result == 1:
            stats["p2_wins"] += 1
        elif result == 0:
            stats["p2_draws"] += 1
        else:
            stats["p2_losses"] += 1


    p1_games = half
    p2_games = n_games - half
    p1_wr = stats["p1_wins"] / max(p1_games, 1)
    p1_dr = stats["p1_draws"] / max(p1_games, 1)
    p1_lr = stats["p1_losses"] / max(p1_games, 1)
    p2_wr = stats["p2_wins"] / max(p2_games, 1)
    p2_dr = stats["p2_draws"] / max(p2_games, 1)
    p2_lr = stats["p2_losses"] / max(p2_games, 1)

    total_w = stats["p1_wins"] + stats["p2_wins"]
    total_d = stats["p1_draws"] + stats["p2_draws"]
    total_l = stats["p1_losses"] + stats["p2_losses"]

    return {
        "agent": getattr(agent, "name", str(type(agent).__name__)),
        "opponent": getattr(opponent, "name", str(type(opponent).__name__)),
        "n_games": n_games,
        "p1_games": p1_games,
        "p1_wins": stats["p1_wins"],
        "p1_draws": stats["p1_draws"],
        "p1_losses": stats["p1_losses"],
        "p1_win_rate": round(p1_wr, 4),
        "p1_draw_rate": round(p1_dr, 4),
        "p1_loss_rate": round(p1_lr, 4),
        "p2_games": p2_games,
        "p2_wins": stats["p2_wins"],
        "p2_draws": stats["p2_draws"],
        "p2_losses": stats["p2_losses"],
        "p2_win_rate": round(p2_wr, 4),
        "p2_draw_rate": round(p2_dr, 4),
        "p2_loss_rate": round(p2_lr, 4),
        "total_wins": total_w,
        "total_draws": total_d,
        "total_losses": total_l,
        "total_win_rate": round(total_w / n_games, 4),
        "total_draw_rate": round(total_d / n_games, 4),
        "total_loss_rate": round(total_l / n_games, 4),
        "first_mover_advantage": round(p1_wr - p2_wr, 4),
        "avg_game_length": round(float(np.mean(stats["game_lengths"])), 2),
        "avg_agent_time_ms_per_move": round(float(np.mean(stats["agent_times"])), 3),
    }


# I have implemented this callable with parameters: agents, game_cls, game_kwargs, n_games, seed.
def crossplay(
    agents: list,
    game_cls,
    game_kwargs: dict | None = None,
    n_games: int = 100,
    seed: int = 42,
) -> list[dict]:


    results = []
    kwargs = game_kwargs or {}

    for i, agent_a in enumerate(agents):
        for j, agent_b in enumerate(agents):
            if i == j:
                continue
            _set_greedy(agent_a)
            _set_greedy(agent_b)
            game = game_cls(**kwargs)
            half = n_games // 2
            stats = {"a_wins": 0, "b_wins": 0, "draws": 0,
                     "game_lengths": [], "a_times": []}


            for _ in range(half):
                result, gl, at = _play_game(agent_a, game, 1, agent_b)
                stats["game_lengths"].append(gl)
                stats["a_times"].append(at / max(gl, 1) * 1000)
                if result == 1:
                    stats["a_wins"] += 1
                elif result == 0:
                    stats["draws"] += 1
                else:
                    stats["b_wins"] += 1


            for _ in range(n_games - half):
                result, gl, at = _play_game(agent_a, game, 2, agent_b)
                stats["game_lengths"].append(gl)
                stats["a_times"].append(at / max(gl, 1) * 1000)
                if result == 1:
                    stats["a_wins"] += 1
                elif result == 0:
                    stats["draws"] += 1
                else:
                    stats["b_wins"] += 1

            results.append({
                "agent": getattr(agent_a, "name", str(type(agent_a).__name__)),
                "opponent": getattr(agent_b, "name", str(type(agent_b).__name__)),
                "n_games": n_games,
                "wins": stats["a_wins"],
                "draws": stats["draws"],
                "losses": stats["b_wins"],
                "win_rate": round(stats["a_wins"] / n_games, 4),
                "draw_rate": round(stats["draws"] / n_games, 4),
                "loss_rate": round(stats["b_wins"] / n_games, 4),
                "avg_game_length": round(float(np.mean(stats["game_lengths"])), 2),
                "avg_time_ms": round(float(np.mean(stats["a_times"])), 3),
            })

    return results


# I have implemented this callable with parameters: agent.
def _set_greedy(agent) -> None:


    """
    I have implemented this function with a clearer note.
    Parameters used here: agent.
    """
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    if hasattr(agent, "set_epsilon"):
        agent.set_epsilon(0.0)
