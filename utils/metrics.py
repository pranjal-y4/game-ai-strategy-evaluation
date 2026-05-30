"""
utils/metrics.py
Utilities for collecting and computing metrics.
"""

import time
from typing import Dict, List


class MetricsCollector:
    """Collect metrics for experiments."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.games = []
        self.start_time = time.time()

    def record_game(self, agent1, agent2, winner, game_length, agent1_time, agent2_time, agent1_nodes=0, agent2_nodes=0):
        """Record a single game result."""
        self.games.append({
            'agent1': agent1,
            'agent2': agent2,
            'winner': winner,  # 1 for agent1, 2 for agent2, 0 for draw
            'game_length': game_length,
            'agent1_time': agent1_time,
            'agent2_time': agent2_time,
            'agent1_nodes': agent1_nodes,
            'agent2_nodes': agent2_nodes,
        })

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.games:
            return {}

        total_games = len(self.games)
        agent1_wins = sum(1 for g in self.games if g['winner'] == 1)
        agent2_wins = sum(1 for g in self.games if g['winner'] == 2)
        draws = sum(1 for g in self.games if g['winner'] == 0)

        avg_game_length = sum(g['game_length'] for g in self.games) / total_games
        avg_agent1_time = sum(g['agent1_time'] for g in self.games) / total_games
        avg_agent2_time = sum(g['agent2_time'] for g in self.games) / total_games
        avg_agent1_nodes = sum(g['agent1_nodes'] for g in self.games) / total_games
        avg_agent2_nodes = sum(g['agent2_nodes'] for g in self.games) / total_games

        total_time = time.time() - self.start_time

        return {
            'total_games': total_games,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'agent1_win_rate': agent1_wins / total_games,
            'agent2_win_rate': agent2_wins / total_games,
            'draw_rate': draws / total_games,
            'avg_game_length': avg_game_length,
            'avg_agent1_time': avg_agent1_time,
            'avg_agent2_time': avg_agent2_time,
            'avg_agent1_nodes': avg_agent1_nodes,
            'avg_agent2_nodes': avg_agent2_nodes,
            'total_time': total_time,
        }