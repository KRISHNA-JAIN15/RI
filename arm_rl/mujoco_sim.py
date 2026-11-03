"""MuJoCo backend: Universal Robots UR5e (MuJoCo Menagerie) + capsule obstacles.

* Robot: ``third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml`` (meshes,
  per-link capsule collision geometry, position-servo actuators of the real UR5e).
* Obstacles: up to ``max_obstacles`` capsule geoms on mocap bodies, re-posed every step
  from the :class:`arm_rl.scene.Scene` (static or moving).
* Paper state vectors S_l: ``mujoco.mj_geomDistance`` between every robot collision
  geom and every obstacle geom gives the exact closest points on the *real* collision
  model; the minimum over the geoms of a link is that link's nearest-obstacle vector.
* Collisions: MuJoCo contacts (robot-obstacle, robot-floor, robot self-collision).
* Control: ``kinematic`` (joint targets reached exactly, like the paper) or ``dynamic``
  (position servos + ``mj_step`` physics for ``dt`` seconds; the achieved joint
  configuration is what the agent observes).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from arm_rl.scene import Scene

_HERE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = _HERE / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"

LINK_BODIES = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]


def _quat_from_z(direction: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) rotating +z onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    s = np.sqrt((1 + c) * 2)
    return np.array([s / 2, axis[0] / s, axis[1] / s, axis[2] / s])


class MujocoArm:
    """Arm interface (FK, Jacobian, limits) backed by a private MjData, so queries never
    disturb the live simulation state."""

    def __init__(self, model: mujoco.MjModel, home: np.ndarray, link_radius: float, reach: float, joint_limit_deg: float | None):
        self.model = model
        self._data = mujoco.MjData(model)
        self.n_joints = 6
        self.home = np.asarray(home, dtype=np.float64)
        self.link_radius = float(link_radius)
        self._reach = float(reach)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in LINK_BODIES]
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        lo, hi = model.jnt_range[:6, 0].copy(), model.jnt_range[:6, 1].copy()
        if joint_limit_deg is not None:
            lim = np.deg2rad(float(joint_limit_deg))
            lo, hi = np.maximum(lo, -lim), np.minimum(hi, lim)
        self.joint_low, self.joint_high = lo, hi
        self.shoulder = np.array([0.0, 0.0, float(model.body_pos[self.body_ids[0]][2])])

    @property
    def reach(self) -> float:
        return self._reach

    def _fk(self, q: np.ndarray) -> None:
        self._data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self._data)

    def forward(self, q: np.ndarray):
        self._fk(np.asarray(q, dtype=np.float64))
        origins = self.origins_from(self._data)
        T = np.eye(4)
        T[:3, 3] = self._data.site_xpos[self.site_id]
        T[:3, :3] = self._data.site_xmat[self.site_id].reshape(3, 3)
        return origins, T

    def origins_from(self, data: mujoco.MjData) -> np.ndarray:
        pts = [data.xpos[self.base_id]] + [data.xpos[b] for b in self.body_ids] + [data.site_xpos[self.site_id]]
        return np.asarray(pts, dtype=np.float64)

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        self._fk(np.asarray(q, dtype=np.float64))
        return self._data.site_xpos[self.site_id].copy()

    def link_segments(self, q: np.ndarray):
        o, _ = self.forward(q)
        # 7 segments: base->shoulder, shoulder->upper_arm, ..., wrist_3->tool
        return o[:-1], o[1:]

    def position_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        self._fk(np.asarray(q, dtype=np.float64))
        mujoco.mj_comPos(self.model, self._data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self._data, jacp, None, self.site_id)
        return jacp[:, :6]

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_low, self.joint_high)

    def sample_joints(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.joint_low, self.joint_high)


class MujocoBackend:
    name = "mujoco"

    def __init__(self, arm_cfg: dict, sim_cfg: dict, floor_collision: bool):
        sim_cfg = dict(sim_cfg or {})
        model_path = Path(sim_cfg.get("model", DEFAULT_MODEL))
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path} (download MuJoCo Menagerie's universal_robots_ur5e)")
        self.max_obstacles = int(sim_cfg.get("max_obstacles", 32))
        self.control = str(sim_cfg.get("control", "kinematic"))
        assert self.control in ("kinematic", "dynamic")
        self.self_collision = bool(sim_cfg.get("self_collision", True))
        self.floor_collision = floor_collision
        self.distmax = float(sim_cfg.get("distmax", 2.0))

        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.option.timestep = float(sim_cfg.get("timestep", 0.002))
        spec.visual.global_.offwidth = 1280  # offscreen render size for scripts/mujoco_demo.py --video
        spec.visual.global_.offheight = 960
        world = spec.worldbody
        # floor + lights (the menagerie scene.xml equivalents)
        floor = world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.2, 0.25, 0.3, 1])
        floor.contype, floor.conaffinity = 1, 1
        world.add_light(pos=[0, -1, 2], dir=[0, 0.7, -0.7])
        world.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        # obstacle slots: capsule on a mocap body
        for i in range(self.max_obstacles):
            b = world.add_body(name=f"obs{i}", mocap=True, pos=[0, 0, -5.0 - i])
            g = b.add_geom(name=f"obs{i}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.04, 0.1, 0], rgba=[0.13, 0.77, 0.37, 1])
            g.contype, g.conaffinity = 1, 1
        # target marker (no collision)
        tb = world.add_body(name="target", mocap=True, pos=[0, 0, -3])
        tg = tb.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], rgba=[0.94, 0.27, 0.27, 1])
        tg.contype, tg.conaffinity = 0, 0
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # ids
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.obs_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs{i}_geom") for i in range(self.max_obstacles)]
        self.obs_mocap = [self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs{i}")] for i in range(self.max_obstacles)]
        self.target_mocap = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.target_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.link_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in LINK_BODIES]
        robot_root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_bodies = set()
        for b in range(self.model.nbody):
            p = b
            while p > 0:
                if p == robot_root:
                    self.robot_bodies.add(b)
                    break
                p = self.model.body_parentid[p]
        # collision geoms per link (group 3 capsules/cylinders of the menagerie model)
        self.link_geoms: list[list[int]] = []
        for bid in self.link_body_ids:
            gids = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid and self.model.geom_contype[g] != 0]
            self.link_geoms.append(gids)
        self.robot_geoms = {g for gids in self.link_geoms for g in gids}
        self.obs_gid_set = set(self.obs_gids)

        # arm interface
        home_deg = arm_cfg.get("home_deg")
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home = np.deg2rad(np.asarray(home_deg, dtype=np.float64)) if home_deg is not None else self.model.key_qpos[key][:6].copy()
        self.arm = MujocoArm(self.model, home, float(arm_cfg.get("link_radius", 0.05)), float(arm_cfg.get("reach", 0.85)), arm_cfg.get("joint_limits_deg"))
        # link index convention: 0 = base pedestal (shoulder body), 1..5 = moving links -> matches monitored_links [1..5]
        self.n_links = 6
        self.scene: Scene | None = None
        self.n_active = 0
        self.collided_during_motion = False
        self._fromto = np.zeros(6)
        self.set_joints(self.arm.home)

    # ------------------------------------------------------------------ scene
    def set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self.n_active = min(scene.n_obstacles, self.max_obstacles)
        radii = scene.radii
        for i in range(self.max_obstacles):
            gid = self.obs_gids[i]
            if i < self.n_active:
                self.model.geom_size[gid][0] = radii[i]
            else:
                self.model.geom_size[gid] = [0.001, 0.001, 0]
                self.data.mocap_pos[self.obs_mocap[i]] = [0, 0, -5.0 - i]
        self.model.geom_size[self.target_gid][0] = scene.target_radius
        self.sync_scene()

    def sync_scene(self) -> None:
        sc = self.scene
        for i in range(self.n_active):
            a, b = sc.obs_q1[i], sc.obs_q2[i]
            d = b - a
            half = 0.5 * np.linalg.norm(d)
            gid = self.obs_gids[i]
            self.model.geom_size[gid][1] = half
            self.data.mocap_pos[self.obs_mocap[i]] = 0.5 * (a + b)
            self.data.mocap_quat[self.obs_mocap[i]] = _quat_from_z(d)
        self.data.mocap_pos[self.target_mocap] = sc.target
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ joints
    def set_joints(self, q: np.ndarray) -> None:
        self.data.qpos[:6] = q
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = q
        mujoco.mj_forward(self.model, self.data)

    def apply_joints(self, q_cmd: np.ndarray, dt: float) -> np.ndarray:
        self.collided_during_motion = False
        if self.control == "kinematic" or dt <= 0.0:
            self.set_joints(q_cmd)
            return np.asarray(q_cmd, dtype=np.float64).copy()
        self.data.ctrl[:6] = q_cmd
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self._contacts_collide():
                self.collided_during_motion = True
        return self.data.qpos[:6].copy()

    # ------------------------------------------------------------------ geometry
    def _contacts_collide(self) -> bool:
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            in1, in2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if not (in1 or in2):
                continue
            other = g2 if in1 else g1
            if other in self.obs_gid_set:
                return True
            if self.floor_collision and other == self.floor_gid:
                return True
            if self.self_collision and in1 and in2:
                return True
        return False

    def geometry(self, q: np.ndarray):
        """(origins (8,3), clearance (6,M), direction (6,M,3), capsules) at configuration q.

        Uses the live MjData when q equals the current configuration, otherwise a kinematic
        update of the live data (restored afterwards by the caller via apply_joints/set_joints).
        """
        if not np.allclose(self.data.qpos[:6], q):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
        M = self.n_active
        clearance = np.full((self.n_links, M), np.inf)
        direction = np.zeros((self.n_links, M, 3))
        for li, gids in enumerate(self.link_geoms):
            for j in range(M):
                og = self.obs_gids[j]
                best = np.inf
                for g in gids:
                    dist = mujoco.mj_geomDistance(self.model, self.data, g, og, self.distmax, self._fromto)
                    if dist < best:
                        best = dist
                        direction[li, j] = self._fromto[3:] - self._fromto[:3]
                clearance[li, j] = best
        origins = self.arm.origins_from(self.data)
        return origins, clearance, direction, self.capsules()

    def capsules(self) -> list[dict]:
        """Current robot collision capsules in world frame (for the viewer)."""
        out = []
        for li, gids in enumerate(self.link_geoms):
            for g in gids:
                size = self.model.geom_size[g]
                pos = self.data.geom_xpos[g]
                mat = self.data.geom_xmat[g].reshape(3, 3)
                axis = mat[:, 2] * size[1]
                out.append({"a": (pos - axis).round(4).tolist(), "b": (pos + axis).round(4).tolist(), "r": round(float(size[0]), 4), "link": li})
        return out

    def is_collision(self, q: np.ndarray, clearance: np.ndarray | None = None) -> bool:
        if not np.allclose(self.data.qpos[:6], q):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
        if self._contacts_collide():
            return True
        if clearance is not None and clearance.size and np.any(clearance[1:] < 0.0):
            return True
        return False
