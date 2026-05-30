from __future__ import annotations
import random
import numpy as np
from collections import deque


class UniformReplayBuffer:


    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float,
             next_state, done: bool, next_legal: list) -> None:
        self._buf.append((state, action, reward, next_state, done, next_legal))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s, a, r, ns, d, nl = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
            list(nl),
            np.ones(batch_size, dtype=np.float32),
            None,
        )

    def __len__(self) -> int:
        return len(self._buf)


class _SumTree:


    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list = [None] * capacity
        self._ptr = 0
        self._size = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def size(self) -> int:
        return self._size

    @property
    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        leaves = self.tree[self.capacity - 1: self.capacity - 1 + self._size]
        return float(max(leaves.max(), 1e-6))

    def add(self, priority: float, data) -> None:
        leaf_idx = self._ptr + self.capacity - 1
        self._data[self._ptr] = data
        self._update(leaf_idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:

        self._update(leaf_idx, priority)

    def _update(self, idx: int, priority: float) -> None:
        delta = priority - self.tree[idx]
        self.tree[idx] = priority

        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, value: float) -> tuple[int, float, object]:


        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self._data[data_idx]


class PrioritizedReplayBuffer:


    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_inc = (beta_end - beta_start) / max(beta_steps, 1)
        self.eps = eps
        self._tree = _SumTree(capacity)
        self._step = 0

    def push(self, state, action: int, reward: float,
             next_state, done: bool, next_legal: list) -> None:

        priority = self._tree.max_priority ** self.alpha
        self._tree.add(priority, (state, action, reward, next_state, done, next_legal))

    def sample(self, batch_size: int):


        indices, priorities, data_batch = [], [], []
        segment = self._tree.total / batch_size

        self.beta = min(self.beta_end, self.beta + self.beta_inc)
        self._step += 1

        for i in range(batch_size):

            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, pri, data = self._tree.get(s)
            if data is None:

                s = random.uniform(0, self._tree.total)
                idx, pri, data = self._tree.get(s)
            indices.append(idx)
            priorities.append(max(pri, self.eps))
            data_batch.append(data)


        N = self._tree.size
        min_prob = (self.eps / self._tree.total)
        max_weight = (N * min_prob) ** (-self.beta)

        weights = []
        for pri in priorities:
            prob = pri / self._tree.total
            w = (N * prob) ** (-self.beta)
            weights.append(w / max_weight)

        s, a, r, ns, d, nl = zip(*data_batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
            list(nl),
            np.array(weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:

        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.eps) ** self.alpha
            self._tree.update(idx, priority)

    def __len__(self) -> int:
        return self._tree.size
