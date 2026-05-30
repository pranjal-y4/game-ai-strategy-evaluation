from __future__ import annotations
import numpy as np


class MetricsTracker:


    def __init__(self):
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.game_lengths: list[int] = []
        self.times: list[float] = []

    def record(self, result: int, game_length: int = 0,
               time_sec: float = 0.0) -> None:


        if result == 1:
            self.wins += 1
        elif result == 0:
            self.draws += 1
        else:
            self.losses += 1
        self.game_lengths.append(game_length)
        self.times.append(time_sec)

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_games, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.total_games, 1)

    @property
    def loss_rate(self) -> float:
        return self.losses / max(self.total_games, 1)

    def summary(self) -> dict:
        return {
            "total_games": self.total_games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "draw_rate": round(self.draw_rate, 4),
            "loss_rate": round(self.loss_rate, 4),
            "avg_game_length": round(float(np.mean(self.game_lengths))
                                     if self.game_lengths else 0.0, 2),
            "avg_time_ms": round(float(np.mean(self.times)) * 1000
                                 if self.times else 0.0, 3),
        }
