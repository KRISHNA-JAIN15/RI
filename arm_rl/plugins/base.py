"""Hook contract shared by all plugins.

Every hook has a no-op default, so a plugin overrides only what it needs.  ``phase`` is
"train" or "eval"; a plugin that only acts at training time (e.g. HER) checks it.
``cost`` is a dict the benchmark reads to attribute compute to the plugin (seconds spent
inside the plugin, extra simulator calls it made).
"""

from __future__ import annotations

import time

import numpy as np


class Plugin:
    name = "plugin"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        self.cfg = dict(cfg or {})
        self.env = env
        self.agent = agent
        self.phase = phase
        self.cost = {"time_s": 0.0, "extra_sim_steps": 0, "extra_transitions": 0}

    # ------------------------------------------------------------------ episode hooks
    def choose_scene(self, episode: int):
        """Return a Scene to use for this episode (curriculum) or None for the default sampler."""
        return None

    def on_episode_start(self, episode: int, obs: np.ndarray, info: dict) -> None:
        pass

    def on_episode_end(self, transitions: list[tuple], episode: int) -> list[tuple]:
        """``transitions`` = [(obs, action, reward, next_obs, terminated, info), ...] of the
        finished episode. Return extra transitions (obs, action, reward, next_obs, terminated)
        to add to the replay buffer (e.g. hindsight relabelling)."""
        return []

    # ------------------------------------------------------------------ step hooks
    def transform_action(self, obs: np.ndarray, action: np.ndarray, deterministic: bool) -> np.ndarray:
        """Modify the action before it is executed (safety shield, control prior)."""
        return action

    def shape_reward(self, obs, action, reward: float, next_obs, terminated: bool, info: dict) -> float:
        """Return the training reward. The logged episode return always uses the raw reward."""
        return reward

    def extra_transitions(self, obs, action, reward, next_obs, terminated: bool, info: dict) -> list[tuple]:
        """Additional transitions to store right after this step (e.g. opposition-based)."""
        return []

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        return dict(self.cost)

    # helper for subclasses: time a block and charge it to the plugin
    class _timer:
        def __init__(self, plugin):
            self.p = plugin

        def __enter__(self):
            self.t0 = time.perf_counter()

        def __exit__(self, *a):
            self.p.cost["time_s"] += time.perf_counter() - self.t0

    def timed(self):
        return Plugin._timer(self)
