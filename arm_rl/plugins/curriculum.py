"""C5 curriculum: pick each training scene by its persistent-homology difficulty ``tau``.

Every episode the plugin samples ``candidates`` scenes from the environment's generator,
computes ``tau`` for each (:func:`arm_rl.topology.scene_complexity`, cached) and returns
the candidate whose rank quantile among the candidates is closest to the schedule's
target quantile.  The target grows from ``q_start`` to ``q_end`` over the first
``frac`` of training (``warm_episodes``), then the plugin falls back to uniform sampling
(returns ``None``) so the final policy sees the natural scene distribution.

Options (``method.curriculum``):
    candidates: 8          scenes drawn per episode
    schedule: linear       linear | cosine | step (step: q_start then q_end)
    q_start: 0.1           target quantile at episode 1
    q_end: 0.9             target quantile at the end of the warm-up
    frac: 0.6              warm-up fraction of the training episodes
    episodes: null         total training episodes; inferred from the command line
                           (--episodes / --config / --set train.episodes=...) when null
    warm_episodes: null    overrides frac * episodes
    n_samples: 1500        grid points for the complexity estimate (speed/accuracy)
    max_dim: 1             homology dimensions (1 = H0/H1)
    pad: 0.15              region-of-interest padding (m) around start and target

The time spent sampling candidates and computing ``tau`` is charged to ``cost["time_s"]``.
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from arm_rl.plugins.base import Plugin
from arm_rl.topology import scene_complexity


def _infer_total_episodes(default: int = 10000) -> int:
    """Best-effort read of the training length from ``sys.argv`` (train.py's flags)."""
    argv = list(sys.argv)
    episodes = None
    config_path = None
    for i, tok in enumerate(argv):
        if tok == "--episodes" and i + 1 < len(argv):
            episodes = int(argv[i + 1])
        elif tok.startswith("--episodes="):
            episodes = int(tok.split("=", 1)[1])
        elif tok == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif tok.startswith("--config="):
            config_path = tok.split("=", 1)[1]
        elif tok == "--set" and i + 1 < len(argv) and argv[i + 1].startswith("train.episodes="):
            episodes = int(yaml.safe_load(argv[i + 1].split("=", 1)[1]))
    if episodes is not None:
        return episodes
    if config_path is not None:
        try:
            with open(config_path) as f:
                return int(yaml.safe_load(f)["train"]["episodes"])
        except Exception:
            pass
    return default


class Plugin(Plugin):
    name = "curriculum"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        super().__init__(cfg, env, agent, phase)
        c = self.cfg
        self.candidates = int(c.get("candidates", 8))
        self.schedule = str(c.get("schedule", "linear"))
        self.q_start = float(c.get("q_start", 0.1))
        self.q_end = float(c.get("q_end", 0.9))
        self.frac = float(c.get("frac", 0.6))
        self.n_samples = int(c.get("n_samples", 1500))
        self.max_dim = int(c.get("max_dim", 1))
        self.pad = float(c.get("pad", 0.15))
        self.episodes = int(c["episodes"]) if c.get("episodes") is not None else _infer_total_episodes()
        self.warm_episodes = int(c["warm_episodes"]) if c.get("warm_episodes") is not None else max(1, int(round(self.frac * self.episodes)))
        self.start = np.asarray(env.arm.ee_position(env.arm.home), dtype=np.float64)
        self.history: list[tuple[int, float, float]] = []  # (episode, target quantile, chosen tau)
        self.n_chosen = 0
        self.n_uniform = 0

    # ------------------------------------------------------------------ schedule
    def target_quantile(self, episode: int) -> float | None:
        """Target difficulty quantile for ``episode`` (1-based) or None after the warm-up."""
        if episode > self.warm_episodes:
            return None
        u = (episode - 1) / max(1, self.warm_episodes - 1)
        if self.schedule == "cosine":
            u = 0.5 - 0.5 * np.cos(np.pi * u)
        elif self.schedule == "step":
            u = 0.0 if u < 0.5 else 1.0
        return float(self.q_start + (self.q_end - self.q_start) * np.clip(u, 0.0, 1.0))

    def complexity(self, scene) -> float:
        return float(scene_complexity(scene, self.env.arm.shoulder, self.env.arm.reach, n_samples=self.n_samples,
                                      max_dim=self.max_dim, start=self.start, pad=self.pad)["tau"])

    # ------------------------------------------------------------------ hook
    def choose_scene(self, episode: int):
        if self.phase != "train":
            return None
        q = self.target_quantile(episode)
        if q is None:
            self.n_uniform += 1
            return None
        with self.timed():
            scenes = [self.env.scene_gen.sample(self.env.rng) for _ in range(self.candidates)]
            taus = np.array([self.complexity(s) for s in scenes])
            order = np.argsort(taus, kind="stable")
            # rank quantile of the k-th smallest candidate: (k + 0.5) / n
            ranks = (np.arange(len(order)) + 0.5) / len(order)
            pick = int(order[int(np.argmin(np.abs(ranks - q)))])
        self.history.append((episode, q, float(taus[pick])))
        self.n_chosen += 1
        return scenes[pick]

    def stats(self) -> dict:
        out = dict(self.cost)
        out.update({"episodes_curriculum": self.n_chosen, "episodes_uniform": self.n_uniform, "warm_episodes": self.warm_episodes,
                    "candidates": self.candidates, "n_samples": self.n_samples})
        if self.history:
            taus = np.array([h[2] for h in self.history])
            k = max(1, len(taus) // 5)
            out.update({"tau_first_fifth": float(taus[:k].mean()), "tau_last_fifth": float(taus[-k:].mean()),
                        "tau_mean": float(taus.mean())})
        return out
