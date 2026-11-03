"""Gymnasium environment implementing Sec. III of the paper, with pluggable simulators.

State  (eq. 8-12): S = [Sl, Sa, Se, St]
    Sl  - direction vector of the shortest distance from each monitored link to its
          nearest obstacle (5 x 3 = 15 dims)
    Sa  - joint angles (6 dims, radians)
    Se  - end-effector xyz (3 dims)
    St  - picking-target xyz (3 dims)
    optional (include_velocities): nearest-obstacle velocity per monitored link (5 x 3)
    and target velocity (3) for dynamic scenes -> 45 dims.
Action (Sec. III-B): incremental joint rotation in [-5 deg, 5 deg] for all six axes.
Reward (eq. 13):     -|d| if |d| > 3 cm, 0 if |d| < 3 cm, -50 on collision, with
                     d = 10 x ||EE - target||.
A round ends after 100 steps, when the target is touched, or (collision_mode =
terminate) when a link collides with an obstacle.

Simulators (env_cfg.sim.backend):
    analytic - numpy DH kinematics + exact capsule geometry (paper's simplified arm)
    mujoco   - MuJoCo physics with the Universal Robots UR5e model (see mujoco_sim.py)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arm_rl.geometry import segment_distance_matrix
from arm_rl.kinematics import SixAxisArm
from arm_rl.scene import Scene, make_scene_generator


# ============================================================================ backends
class AnalyticBackend:
    """Paper's simplification: links and obstacles are capsules, exact closest points."""

    name = "analytic"

    def __init__(self, arm_cfg: dict, sim_cfg: dict, floor_collision: bool):
        self.arm = SixAxisArm.from_config(arm_cfg)
        self.floor_collision = floor_collision
        self.scene: Scene | None = None
        self.n_links = self.arm.n_joints
        self.link_names = ["base", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def sync_scene(self) -> None:  # obstacles moved -> nothing cached
        pass

    def apply_joints(self, q: np.ndarray, dt: float) -> np.ndarray:
        """Kinematic arm: the commanded configuration is reached exactly."""
        return np.asarray(q, dtype=np.float64).copy()

    def geometry(self, q: np.ndarray):
        """Return (origins (7,3), clearance (6,M), direction (6,M,3), capsules)."""
        p1, p2 = self.arm.link_segments(q)
        origins = np.vstack([p1, p2[-1:]])
        if self.scene is None or self.scene.n_obstacles == 0:
            return origins, np.zeros((self.n_links, 0)), np.zeros((self.n_links, 0, 3)), None
        dist, direction = segment_distance_matrix(p1, p2, self.scene.obs_q1, self.scene.obs_q2)
        clearance = dist - self.arm.link_radius - self.scene.radii[None, :]
        return origins, clearance, direction, None

    def is_collision(self, q: np.ndarray, clearance: np.ndarray | None = None) -> bool:
        p1, p2 = self.arm.link_segments(q)
        if clearance is None:
            clearance = self.geometry(q)[1]
        if clearance.size and np.any(clearance[1:] < 0.0):
            return True
        if self.floor_collision and np.any(p2[1:, 2] < 0.0):
            return True
        return False


def make_backend(arm_cfg: dict, sim_cfg: dict, floor_collision: bool):
    backend = str((sim_cfg or {}).get("backend", "analytic"))
    if backend == "analytic":
        return AnalyticBackend(arm_cfg, sim_cfg, floor_collision)
    if backend == "mujoco":
        from arm_rl.mujoco_sim import MujocoBackend

        return MujocoBackend(arm_cfg, sim_cfg, floor_collision)
    raise ValueError(f"unknown sim backend {backend}")


# ============================================================================ environment
class ArmPathPlanningEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        arm_cfg: dict | None = None,
        scene_cfg: dict | None = None,
        env_cfg: dict | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        arm_cfg = dict(arm_cfg or {})
        scene_cfg = dict(scene_cfg or {})
        env_cfg = dict(env_cfg or {})
        self.floor_collision = bool(scene_cfg.get("floor_collision", True))
        self.sim_cfg = dict(env_cfg.get("sim") or {})
        self.backend = make_backend(arm_cfg, self.sim_cfg, self.floor_collision)
        self.arm = self.backend.arm
        self.scene_gen = make_scene_generator(self.arm, scene_cfg)

        self.max_steps = int(env_cfg.get("max_steps", 100))
        self.max_delta = np.deg2rad(float(env_cfg.get("max_delta_deg", 5.0)))
        self.reach_threshold = float(env_cfg.get("reach_threshold", 0.03))
        self.distance_scale = float(env_cfg.get("distance_scale", 10.0))
        self.collision_penalty = float(env_cfg.get("collision_penalty", -50.0))
        self.collision_mode = str(env_cfg.get("collision_mode", "terminate"))
        assert self.collision_mode in ("terminate", "continue")
        self.monitored_links = list(env_cfg.get("monitored_links", [1, 2, 3, 4, 5]))
        self.normalize_angles = bool(env_cfg.get("normalize_angles", False))
        self.include_velocities = bool(env_cfg.get("include_velocities", False))
        self.dt = float(env_cfg.get("dt", 0.1))  # seconds of scene motion per step
        self.render_mode = render_mode

        n_links = len(self.monitored_links)
        self.state_dim = 3 * n_links + self.arm.n_joints + 3 + 3
        if self.include_velocities:
            self.state_dim += 3 * n_links + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.arm.n_joints,), dtype=np.float32)

        self.rng = np.random.default_rng()
        self.scene: Scene | None = None
        self.q = self.arm.home.copy()
        self.ee = self.arm.ee_position(self.q)
        self.step_count = 0
        self.ee_path_length = 0.0
        self.n_collisions = 0
        self.ee_history: list[np.ndarray] = []
        self._geom_cache: tuple | None = None

    # ------------------------------------------------------------------ geometry
    def _geometry(self, q: np.ndarray | None = None):
        if q is None:
            if self._geom_cache is None:
                self._geom_cache = self.backend.geometry(self.q)
            return self._geom_cache
        return self.backend.geometry(q)

    def is_collision(self, q: np.ndarray) -> bool:
        return self.backend.is_collision(q)

    def is_collision_free(self, q: np.ndarray) -> bool:
        return not self.is_collision(q)

    def min_obstacle_clearance(self, q: np.ndarray | None = None) -> float:
        _, clearance, _, _ = self._geometry(q)
        return float(clearance[1:].min()) if clearance.size else float("inf")

    # ------------------------------------------------------------------ state
    def _observation(self) -> np.ndarray:
        _, clearance, direction, _ = self._geometry()
        if clearance.shape[1] == 0:
            sl = np.zeros((len(self.monitored_links), 3))
            nearest = None
        else:
            nearest = np.argmin(clearance, axis=1)  # nearest obstacle per link
            sl = np.stack([direction[i, nearest[i]] for i in self.monitored_links])  # eq. 8
        sa = self.q / np.pi if self.normalize_angles else self.q  # eq. 9
        se = self.ee  # eq. 10
        st = self.scene.target  # eq. 11
        parts = [sl.ravel(), sa, se, st]  # eq. 12
        if self.include_velocities:
            if nearest is None:
                vl = np.zeros((len(self.monitored_links), 3))
            else:
                vl = np.stack([self.scene.obs_vel[nearest[i]] for i in self.monitored_links])
            parts += [vl.ravel(), self.scene.target_vel]
        return np.concatenate(parts).astype(np.float32)

    def distance_to_target(self, q: np.ndarray | None = None) -> float:
        ee = self.ee if q is None else self.arm.ee_position(q)
        return float(np.linalg.norm(ee - self.scene.target))

    def reward_for(self, distance_m: float, collision: bool) -> float:
        """Eq. 13 with d = distance x distance_scale (3 cm -> 0.3)."""
        if collision:
            return self.collision_penalty
        d = self.distance_scale * distance_m
        if distance_m < self.reach_threshold:
            return 0.0
        return -abs(d)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scene = (options or {}).get("scene")
        q0 = (options or {}).get("q0")
        self.q = self.arm.home.copy() if q0 is None else np.asarray(q0, dtype=np.float64).copy()
        for attempt in range(50):
            self.scene = scene.copy() if scene is not None else self.scene_gen.sample(self.rng)
            self.scene.set_time(0.0)
            self.backend.set_scene(self.scene)
            # the generator checks the simplified geometry; the backend has the final say
            if scene is not None or not self.backend.is_collision(self.q):
                break
        else:
            raise RuntimeError("could not find a collision-free start; check the scene preset")
        self.q = self.backend.apply_joints(self.q, 0.0)
        self.ee = self.arm.ee_position(self.q)
        self.step_count = 0
        self.ee_path_length = 0.0
        self.n_collisions = 0
        self.ee_history = [self.ee.copy()]
        self._geom_cache = None
        obs = self._observation()
        info = self._info(collision=False, reached=self.distance_to_target() < self.reach_threshold)
        return obs, info

    def step(self, action: np.ndarray):
        assert self.scene is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_cmd = self.arm.clip_joints(self.q + a * self.max_delta)
        # the world moves while the arm executes the increment
        self.scene.advance(self.dt)
        self.backend.sync_scene()
        q_new = self.backend.apply_joints(q_cmd, self.dt)
        self._geom_cache = self.backend.geometry(q_new)
        collision = self.backend.is_collision(q_new, self._geom_cache[1]) or getattr(self.backend, "collided_during_motion", False)
        if collision:
            self.n_collisions += 1
        prev_ee = self.ee
        self.q = q_new
        self.ee = self.arm.ee_position(self.q)
        self.ee_path_length += float(np.linalg.norm(self.ee - prev_ee))
        self.step_count += 1
        self.ee_history.append(self.ee.copy())

        distance = self.distance_to_target()
        reward = self.reward_for(distance, collision)
        reached = distance < self.reach_threshold
        terminated = bool(reached or (collision and self.collision_mode == "terminate"))
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        obs = self._observation()
        return obs, float(reward), terminated, truncated, self._info(collision, reached)

    def _info(self, collision: bool, reached: bool) -> dict[str, Any]:
        return {
            "distance": self.distance_to_target(),
            "collision": bool(collision),
            "reached": bool(reached),
            "n_collisions": self.n_collisions,
            "ee_path_length": self.ee_path_length,
            "ee": self.ee.copy(),
            "q": self.q.copy(),
            "step": self.step_count,
            "t": self.scene.t,
        }

    # ------------------------------------------------------------------ render
    def render(self):
        from arm_rl.visualize import render_scene

        return render_scene(self.arm, self.scene, self.q, ee_history=np.asarray(self.ee_history))
