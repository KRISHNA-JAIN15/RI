"""Uniform experience replay (paper: pool size 100,000, batch 64)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, done: bool) -> None:
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device):
        idx = rng.integers(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return to(self.state), to(self.action), to(self.reward), to(self.next_state), to(self.done)
