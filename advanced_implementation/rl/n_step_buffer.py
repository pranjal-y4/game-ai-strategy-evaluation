from __future__ import annotations
from collections import deque


class NStepBuffer:


    def __init__(self, n: int = 3, gamma: float = 0.99):
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque()

    def push(self, state, action: int, reward: float,
             next_state, done: bool, legal_next: list) -> list:


        self._buf.append((state, action, reward, next_state, done, legal_next))
        ready = []

        if done:

            while self._buf:
                ready.append(self._compute_head())
                self._buf.popleft()
        elif len(self._buf) >= self.n:
            ready.append(self._compute_head())
            self._buf.popleft()

        return ready

    def flush(self) -> list:

        ready = []
        while self._buf:
            ready.append(self._compute_head())
            self._buf.popleft()
        return ready

    def _compute_head(self) -> tuple:

        s0, a0 = self._buf[0][0], self._buf[0][1]
        R = 0.0
        final_next_s = self._buf[-1][3]
        final_legal = self._buf[-1][5]
        final_done = False

        for i, (_, _, r_i, ns_i, done_i, legal_i) in enumerate(self._buf):
            R += (self.gamma ** i) * r_i
            if done_i:
                final_done = True
                final_next_s = ns_i
                final_legal = legal_i
                break

        return (s0, a0, R, final_next_s, final_done, final_legal)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
