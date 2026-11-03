"""Baseline: Rapidly-exploring Random Tree in joint space (paper Table I, "RRT").

The tree grows in the 6-D configuration space using the same cylinder collision
checker as the DDPG environment.  Because the goal is a Cartesian picking point (no
inverse kinematics), goal biasing is implemented as a Jacobian pseudo-inverse step
from the tree node whose end effector is closest to the target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from arm_rl.env import ArmPathPlanningEnv


@dataclass
class RRTResult:
    success: bool
    path: list[np.ndarray] = field(default_factory=list)   # joint configurations
    ee_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ee_path_length: float = 0.0
    final_error: float = float("inf")
    time_ms: float = 0.0
    iterations: int = 0
    n_nodes: int = 0


class RRTPlanner:
    def __init__(self, env: ArmPathPlanningEnv, cfg: dict | None = None, seed: int = 0):
        cfg = dict(cfg or {})
        self.env = env
        self.arm = env.arm
        self.step = np.deg2rad(float(cfg.get("step_deg", 5.0)))
        self.goal_bias = float(cfg.get("goal_bias", 0.1))
        self.max_iter = int(cfg.get("max_iter", 3000))
        self.max_time = cfg.get("max_time_s")
        self.goal_tol = float(cfg.get("goal_tol", 0.03))
        self.check_resolution = np.deg2rad(float(cfg.get("check_resolution_deg", 2.5)))
        self.task_step = float(cfg.get("task_step_m", 0.03))
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def _edge_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        n = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) / self.check_resolution)))
        for k in range(1, n + 1):
            if self.env.is_collision(q_from + (q_to - q_from) * (k / n)):
                return False
        return True

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        delta = q_rand - q_near
        scale = np.max(np.abs(delta))
        if scale > self.step:
            delta = delta * (self.step / scale)
        return self.arm.clip_joints(q_near + delta)

    def _goal_step(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        ee = self.arm.ee_position(q)
        err = target - ee
        n = np.linalg.norm(err)
        if n > self.task_step:
            err = err * (self.task_step / n)
        J = self.arm.position_jacobian(q)
        dq = np.linalg.pinv(J, rcond=1e-3) @ err
        scale = np.max(np.abs(dq))
        if scale > self.step:
            dq = dq * (self.step / scale)
        return self.arm.clip_joints(q + dq)

    # ------------------------------------------------------------------ planning
    def plan(self, q_start: np.ndarray, target: np.ndarray) -> RRTResult:
        t0 = time.perf_counter()
        nodes = [np.asarray(q_start, dtype=np.float64)]
        parents = [-1]
        ee_pos = [self.arm.ee_position(nodes[0])]
        ee_err = [float(np.linalg.norm(ee_pos[0] - target))]
        goal_idx = 0 if ee_err[0] < self.goal_tol else -1
        it = 0
        while goal_idx < 0 and it < self.max_iter:
            it += 1
            if self.max_time is not None and (time.perf_counter() - t0) > float(self.max_time):
                break
            if self.rng.random() < self.goal_bias:
                near = int(np.argmin(ee_err))
                q_new = self._goal_step(nodes[near], target)
            else:
                q_rand = self.arm.sample_joints(self.rng)
                d = np.linalg.norm(np.asarray(nodes) - q_rand, axis=1)
                near = int(np.argmin(d))
                q_new = self._steer(nodes[near], q_rand)
            if not self._edge_free(nodes[near], q_new):
                continue
            nodes.append(q_new)
            parents.append(near)
            ee = self.arm.ee_position(q_new)
            ee_pos.append(ee)
            e = float(np.linalg.norm(ee - target))
            ee_err.append(e)
            if e < self.goal_tol:
                goal_idx = len(nodes) - 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if goal_idx < 0:
            return RRTResult(False, time_ms=elapsed_ms, iterations=it, n_nodes=len(nodes), final_error=float(min(ee_err)))
        # back-track
        idx = goal_idx
        path_idx = []
        while idx != -1:
            path_idx.append(idx)
            idx = parents[idx]
        path_idx.reverse()
        path = [nodes[i] for i in path_idx]
        ee_path = np.asarray([ee_pos[i] for i in path_idx])
        length = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1))) if len(ee_path) > 1 else 0.0
        return RRTResult(True, path, ee_path, length, ee_err[goal_idx], elapsed_ms, it, len(nodes))
