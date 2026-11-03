"""Hindsight experience replay (Andrychowicz et al. 2017) -- research direction C3.

The state carries the goal (``st`` slice, target xyz) and the achieved goal (``se`` slice,
end-effector xyz).  At the end of every training episode each transition is relabelled
``k`` times with a goal that was actually reached later in the episode ("future"
strategy) or at its end ("final").  Reward and termination are recomputed with the
environment's own reward function, so the relabelled transitions obey eq. 13 exactly.

Options (``method.her``):
    k: 4                 relabelled goals per transition
    strategy: future     future | final
    seed: 0              rng seed for the future-index sampling

Collision transitions keep the collision penalty (``env.reward_for`` returns it whatever
the distance); only the goal slice is rewritten.
"""

from __future__ import annotations

import numpy as np

from arm_rl.plugins.base import Plugin


class Plugin(Plugin):
    name = "her"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        super().__init__(cfg, env, agent, phase)
        self.k = int(self.cfg.get("k", 4))
        self.strategy = str(self.cfg.get("strategy", "future")).lower()
        assert self.strategy in ("future", "final"), f"her.strategy must be future|final, got {self.strategy}"
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        layout = getattr(env, "obs_layout", None) or {}
        self.se = slice(*layout["se"])
        self.st = slice(*layout["st"])
        self.vt = slice(*layout["vt"]) if "vt" in layout else None
        self.cost["relabelled_episodes"] = 0

    # ------------------------------------------------------------------ helpers
    def _future_indices(self, t: int, n: int) -> np.ndarray:
        """Indices of the transitions whose achieved goal relabels transition ``t``."""
        if self.strategy == "final":
            return np.full(self.k, n - 1, dtype=int)
        return self.rng.integers(t, n, size=self.k)

    def relabel_one(self, transition: tuple, goal: np.ndarray) -> tuple:
        obs, action, _reward, next_obs, _terminated, info = transition
        obs = np.array(obs, dtype=np.float32, copy=True)
        next_obs = np.array(next_obs, dtype=np.float32, copy=True)
        obs[self.st] = goal
        next_obs[self.st] = goal
        if self.vt is not None:
            obs[self.vt] = 0.0
            next_obs[self.vt] = 0.0
        distance = float(np.linalg.norm(next_obs[self.se] - goal))
        collision = bool(info.get("collision", False))
        reward = float(self.env.reward_for(distance, collision))
        terminated = bool(distance < self.env.reach_threshold or (collision and self.env.collision_mode == "terminate"))
        return obs, action, reward, next_obs, terminated

    def relabel(self, transitions: list[tuple]) -> list[tuple]:
        n = len(transitions)
        out: list[tuple] = []
        for t in range(n):
            for j in self._future_indices(t, n):
                goal = np.asarray(transitions[int(j)][3], dtype=np.float32)[self.se].copy()
                out.append(self.relabel_one(transitions[t], goal))
        return out

    # ------------------------------------------------------------------ hook
    def on_episode_end(self, transitions: list[tuple], episode: int) -> list[tuple]:
        if self.phase != "train" or self.k <= 0 or not transitions:
            return []
        with self.timed():
            extra = self.relabel(transitions)
        self.cost["extra_transitions"] += len(extra)
        self.cost["relabelled_episodes"] += 1
        return extra
