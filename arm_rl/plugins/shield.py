"""C1: control-barrier-function safety shield plugin.

Enable with ``method.plugins: [shield]``; options under ``method.shield``:

    d_safe: 0.02        minimum surface-to-surface clearance to keep [m]
    alpha: 0.5          class-K gain per step (h_next >= (1 - alpha) h)
    k_nearest: 3        obstacles per monitored link that get a constraint
    train_time: true    filter during training (always filtered during evaluation)
    solver: auto        auto | osqp | numpy  (numpy = Hildreth dual coordinate ascent)
    use_velocity: true  include the obstacle velocity term (scene.obs_vel)
    max_iter: 2000      solver iteration cap
    d_active: 0.25      only pairs closer than this get a constraint (None = all)
    floor: null         add a floor barrier (defaults to the env's floor_collision flag)
    margin: 0.0         extra clearance added to d_safe to absorb the one-step linearisation error [m]

See :mod:`arm_rl.safety` for the constraint derivation and the Jacobians per backend.
"""

from __future__ import annotations

import numpy as np

from arm_rl.plugins.base import Plugin as _Base
from arm_rl.safety import CBFShield


class Plugin(_Base):
    name = "shield"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        super().__init__(cfg, env, agent, phase)
        c = self.cfg
        self.train_time = bool(c.get("train_time", True))
        self.shield = CBFShield(
            env,
            d_safe=float(c.get("d_safe", 0.02)),
            alpha=float(c.get("alpha", 0.5)),
            k_nearest=int(c.get("k_nearest", 3)),
            use_velocity=bool(c.get("use_velocity", True)),
            solver=str(c.get("solver", "auto")),
            max_iter=int(c.get("max_iter", 2000)),
            d_active=c.get("d_active", 0.25),
            floor=c.get("floor", None),
            margin=float(c.get("margin", 0.0)),
        )
        self.cost.update({"calls": 0, "activations": 0, "infeasible": 0, "qp_solves": 0})
        self.last_info: dict = {}

    @property
    def active(self) -> bool:
        return self.phase != "train" or self.train_time

    def transform_action(self, obs: np.ndarray, action: np.ndarray, deterministic: bool) -> np.ndarray:
        if not self.active:
            return action
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        with self.timed():
            dq, info = self.shield.filter(a * self.env.max_delta)
        self.cost["calls"] += 1
        self.cost["qp_solves"] += int(info.get("qp", False))
        if info.get("infeasible"):
            self.cost["infeasible"] += 1
        out = np.clip(dq / self.env.max_delta, -1.0, 1.0)
        if np.max(np.abs(out - a)) > 1e-6:
            self.cost["activations"] += 1
        self.last_info = info
        return out.astype(np.asarray(action).dtype if np.asarray(action).dtype.kind == "f" else np.float32)

    def stats(self) -> dict:
        s = dict(self.cost)
        s["mean_solve_ms"] = 1000.0 * self.cost["time_s"] / max(1, self.cost["calls"])
        s["activation_rate"] = self.cost["activations"] / max(1, self.cost["calls"])
        return s
