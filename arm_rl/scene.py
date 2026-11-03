"""Scenes: obstacles (capsules), picking target (sphere), and their motion.

Two generators are provided:

* :class:`FruitTreeSceneGenerator` - the paper's fruit tree (trunk + branches, r = 4 cm,
  target sphere r = 3 cm, pose noise 0..10 cm / deg, target noise 0..5 cm).  Optional
  ``wind`` makes the branches swing and the fruit sway.
* :class:`ConfigurableSceneGenerator` - any list of obstacles and a target spec read from
  YAML (``configs/scenes/*.yaml``), each with an optional motion law (oscillate, linear
  with bounce, orbit, swing, random walk, conveyor).  Used for the industrial presets.

A :class:`Scene` owns the *base* geometry and the motion laws; :meth:`Scene.advance`
moves obstacles and target forward in time, so both the analytic and the MuJoCo backend
see identical, reproducible dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from arm_rl.geometry import point_segment_distance, segment_distance_matrix

_PRESET_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"


# ============================================================================ motion laws
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _random_unit_3d(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.normal(size=3))


class Motion:
    """Time law p(t) applied to a pair of base points (obstacle axis) or one point (target).

    spec keys (all optional, metres / seconds / degrees):
      type: static | oscillate | linear | orbit | swing | random_walk | conveyor
      axis: direction for oscillate / linear / swing-rotation axis / orbit normal
      amplitude: metres (oscillate) or degrees (swing)
      period: seconds (oscillate, orbit, swing)
      speed: m/s (linear, conveyor), sigma m/sqrt(s) (random_walk)
      bounds: [lo, hi] travel limits along axis for linear (bounce) / conveyor (wrap)
      phase: radians or "random"
      pivot: point for swing / orbit (defaults to the first base point)
      radius: orbit radius
    """

    def __init__(self, spec: dict | None, rng: np.random.Generator):
        spec = dict(spec or {})
        self.type = str(spec.get("type", "static"))
        self.axis = _unit(spec.get("axis", [0.0, 1.0, 0.0]))
        self.amplitude = float(spec.get("amplitude", 0.0))
        self.period = float(spec.get("period", 4.0))
        self.speed = float(spec.get("speed", 0.0))
        self.bounds = spec.get("bounds")
        self.radius = float(spec.get("radius", 0.1))
        self.pivot = spec.get("pivot")
        ph = spec.get("phase", "random")
        self.phase = float(rng.uniform(0, 2 * np.pi)) if ph == "random" else float(ph)
        self.direction = 1.0 if spec.get("direction", 1) >= 0 else -1.0
        self._walk = np.zeros(3)
        self._rng = rng

    def is_static(self) -> bool:
        return self.type == "static"

    def apply(self, base: np.ndarray, t: float, dt: float = 0.0) -> np.ndarray:
        """Return the moved copy of ``base`` (shape (k, 3)) at time ``t``."""
        base = np.asarray(base, dtype=np.float64)
        if self.type == "static":
            return base.copy()
        if self.type == "oscillate":
            off = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
            return base + off * self.axis
        if self.type == "linear":  # bounce between bounds along axis
            lo, hi = (self.bounds or [-0.2, 0.2])
            span = hi - lo
            s0 = lo + (self.phase / (2 * np.pi)) * span  # start inside the range
            x = s0 - lo + self.direction * self.speed * t
            # triangle wave in [0, span]
            x = np.mod(x, 2 * span)
            x = span - abs(x - span)
            return base + (lo + x) * self.axis
        if self.type == "conveyor":  # wrap-around travel along axis (belt)
            lo, hi = (self.bounds or [-0.3, 0.3])
            span = hi - lo
            s0 = (self.phase / (2 * np.pi)) * span
            x = np.mod(s0 + self.speed * t, span)
            return base + (lo + x) * self.axis
        if self.type == "orbit":
            pivot = np.asarray(self.pivot if self.pivot is not None else base[0], dtype=np.float64)
            ang = 2 * np.pi * t / self.period + self.phase
            R = _rot(self.axis, ang)
            return (R @ (base - pivot).T).T + pivot
        if self.type == "swing":  # rotate about pivot by amplitude*sin(...) degrees
            pivot = np.asarray(self.pivot if self.pivot is not None else base[0], dtype=np.float64)
            ang = np.deg2rad(self.amplitude) * np.sin(2 * np.pi * t / self.period + self.phase)
            R = _rot(self.axis, ang)
            return (R @ (base - pivot).T).T + pivot
        if self.type == "random_walk":
            if dt > 0:
                self._walk += self._rng.normal(0.0, self.speed * np.sqrt(dt), size=3)
                lim = self.amplitude if self.amplitude > 0 else 0.1
                self._walk = np.clip(self._walk, -lim, lim)
            return base + self._walk
        raise ValueError(f"unknown motion type {self.type}")


# ============================================================================ scene
@dataclass
class Scene:
    """Obstacle capsule axes, target sphere and their motion laws."""

    obs_q1: np.ndarray  # (M, 3) current
    obs_q2: np.ndarray  # (M, 3) current
    obs_radius: np.ndarray | float  # scalar or (M,)
    target: np.ndarray  # (3,) current
    target_radius: float
    fruits: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obs_motion: list = field(default_factory=list)  # Motion per obstacle
    target_motion: Motion | None = None
    base_q1: np.ndarray | None = None
    base_q2: np.ndarray | None = None
    base_target: np.ndarray | None = None
    t: float = 0.0
    obs_vel: np.ndarray | None = None  # (M, 3) velocity of the obstacle axis midpoint
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = "scene"

    def __post_init__(self):
        self.obs_q1 = np.asarray(self.obs_q1, dtype=np.float64).reshape(-1, 3)
        self.obs_q2 = np.asarray(self.obs_q2, dtype=np.float64).reshape(-1, 3)
        self.target = np.asarray(self.target, dtype=np.float64)
        if self.base_q1 is None:
            self.base_q1 = self.obs_q1.copy()
        if self.base_q2 is None:
            self.base_q2 = self.obs_q2.copy()
        if self.base_target is None:
            self.base_target = self.target.copy()
        if self.obs_vel is None:
            self.obs_vel = np.zeros_like(self.obs_q1)
        if np.isscalar(self.obs_radius):
            self.obs_radius = float(self.obs_radius)
        if self.is_dynamic:
            self.set_time(self.t)  # current geometry = motion law evaluated at t

    # ------------------------------------------------------------------ properties
    @property
    def n_obstacles(self) -> int:
        return int(self.obs_q1.shape[0])

    @property
    def radii(self) -> np.ndarray:
        """Per-obstacle radius as an (M,) array."""
        if np.isscalar(self.obs_radius):
            return np.full(self.n_obstacles, float(self.obs_radius))
        return np.asarray(self.obs_radius, dtype=np.float64)

    @property
    def is_dynamic(self) -> bool:
        if self.target_motion is not None and not self.target_motion.is_static():
            return True
        return any(not m.is_static() for m in self.obs_motion)

    # ------------------------------------------------------------------ dynamics
    def advance(self, dt: float) -> None:
        """Move obstacles and target forward by ``dt`` seconds (finite-difference velocities)."""
        if not self.is_dynamic:
            self.t += dt
            return
        prev_mid = 0.5 * (self.obs_q1 + self.obs_q2)
        prev_target = self.target.copy()
        self.t += dt
        self.set_time(self.t, dt)
        if dt > 0:
            self.obs_vel = (0.5 * (self.obs_q1 + self.obs_q2) - prev_mid) / dt
            self.target_vel = (self.target - prev_target) / dt

    def set_time(self, t: float, dt: float = 0.0) -> None:
        self.t = t
        for i, m in enumerate(self.obs_motion):
            if m.is_static():
                continue
            moved = m.apply(np.stack([self.base_q1[i], self.base_q2[i]]), t, dt)
            self.obs_q1[i], self.obs_q2[i] = moved[0], moved[1]
        if self.target_motion is not None and not self.target_motion.is_static():
            self.target = self.target_motion.apply(self.base_target[None, :], t, dt)[0]

    def copy(self) -> "Scene":
        import copy as _copy

        return _copy.deepcopy(self)


# ============================================================================ validity helpers
class _SceneChecks:
    def __init__(self, arm, cfg: dict):
        self.arm = arm
        self.min_reach = float(cfg.get("min_target_reach", 0.20))
        self.max_reach = float(cfg.get("max_target_reach", 0.95 * arm.reach))
        self.floor_collision = bool(cfg.get("floor_collision", True))
        self.target_radius = float(cfg.get("target_radius", 0.03))

    def target_is_valid(self, target: np.ndarray, q1: np.ndarray, q2: np.ndarray, radii: np.ndarray) -> bool:
        if target[2] < self.target_radius + 0.02:
            return False
        r = np.linalg.norm(target - self.arm.shoulder)
        if not (self.min_reach <= r <= self.max_reach):
            return False
        if len(q1) == 0:
            return True
        clearance = point_segment_distance(target, q1, q2) - radii - self.target_radius
        return bool(np.all(clearance > 0.005))

    def arm_is_collision_free(self, q: np.ndarray, scene: Scene) -> bool:
        p1, p2 = self.arm.link_segments(q)
        if scene.n_obstacles:
            dist, _ = segment_distance_matrix(p1[1:], p2[1:], scene.obs_q1, scene.obs_q2)
            if np.any(dist < self.arm.link_radius + scene.radii[None, :]):
                return False
        if self.floor_collision and np.any(p2[1:, 2] < 0.0):
            return False
        return True


# ============================================================================ fruit tree (paper)
class FruitTreeSceneGenerator(_SceneChecks):
    """Builds a randomised fruit tree in front of the arm (paper Sec. IV-A/B)."""

    def __init__(self, arm, cfg: dict):
        super().__init__(arm, cfg)
        self.arm = arm
        self.obstacle_radius = float(cfg.get("obstacle_radius", 0.04))
        self.tree_position = np.asarray(cfg.get("tree_position", [0.65, 0.0, 0.0]), dtype=np.float64)
        self.trunk_height = float(cfg.get("trunk_height", 0.9))
        self.branch_heights = np.asarray(cfg.get("branch_heights", [0.2, 0.3, 0.4, 0.48]), dtype=np.float64)
        lo, hi = cfg.get("branch_length_range", [0.15, 0.28])
        n = len(self.branch_heights)
        self.branch_lengths = np.linspace(lo, hi, n)
        elo, ehi = cfg.get("branch_elevation_deg", [10.0, 40.0])
        self.branch_elev = np.deg2rad(np.linspace(elo, ehi, n))
        self.branch_azimuth = np.deg2rad(np.array([180.0, 135.0, 225.0, 180.0, 160.0, 200.0])[:n])
        self.pose_noise = float(cfg.get("obstacle_pose_noise", 0.10))
        self.target_noise = float(cfg.get("target_noise", 0.05))
        # optional dynamics: wind swings the branches, fruit sways with the tip
        self.wind = dict(cfg.get("wind") or {})
        self.target_motion_spec = cfg.get("target_motion")
        self.max_tries = 200

    def _branch_direction(self, azimuth: float, elevation: float) -> np.ndarray:
        return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])

    def _build_tree(self, rng: np.random.Generator):
        root = self.tree_position + rng.uniform(0.0, self.pose_noise) * _unit(np.r_[rng.normal(size=2), 0.0])
        q1 = [root]
        q2 = [root + np.array([0.0, 0.0, self.trunk_height])]
        fruits, motions = [], [Motion(None, rng)]
        ang_noise = np.deg2rad(self.pose_noise * 100.0)
        for h, L, el, az in zip(self.branch_heights, self.branch_lengths, self.branch_elev, self.branch_azimuth):
            az_n = az + rng.uniform(-ang_noise, ang_noise)
            el_n = el + rng.uniform(-ang_noise, ang_noise)
            start = root + np.array([0.0, 0.0, h])
            direction = self._branch_direction(az_n, el_n)
            end = start + L * direction
            q1.append(start)
            q2.append(end)
            horiz = direction.copy()
            horiz[2] = 0.0
            horiz = _unit(horiz) if np.linalg.norm(horiz) > 1e-6 else np.zeros(3)
            gap = self.obstacle_radius + self.target_radius + 0.02
            fruits.append(end + 0.5 * gap * horiz - np.array([0.0, 0.0, gap]))
            if self.wind:
                # swing about the branch root around a horizontal axis perpendicular to the branch
                axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
                motions.append(Motion({"type": "swing", "axis": axis, "pivot": start, "amplitude": self.wind.get("amplitude_deg", 8.0),
                                       "period": self.wind.get("period", 3.0), "phase": "random"}, rng))
            else:
                motions.append(Motion(None, rng))
        return np.asarray(q1), np.asarray(q2), np.asarray(fruits), motions

    def sample(self, rng: np.random.Generator) -> Scene:
        radii_fn = lambda n: np.full(n, self.obstacle_radius)  # noqa: E731
        for _ in range(self.max_tries):
            q1, q2, fruits, motions = self._build_tree(rng)
            chosen = None
            chosen_idx = None
            for f in rng.permutation(len(fruits)):
                target = fruits[f] + rng.uniform(0.0, self.target_noise) * _random_unit_3d(rng)
                if self.target_is_valid(target, q1, q2, radii_fn(len(q1))):
                    chosen, chosen_idx = target, int(f)
                    break
            if chosen is None:
                continue
            if self.wind:
                # fruit sways with its branch (same pivot / axis / phase as the branch it hangs from)
                tm = motions[chosen_idx + 1]
                target_motion = Motion({"type": "swing", "axis": tm.axis, "pivot": tm.pivot, "amplitude": tm.amplitude,
                                        "period": tm.period, "phase": tm.phase}, rng)
            else:
                target_motion = Motion(self.target_motion_spec, rng)
            scene = Scene(q1, q2, self.obstacle_radius, chosen, self.target_radius, fruits, motions, target_motion, name="fruit_tree")
            if self.arm_is_collision_free(self.arm.home, scene):
                return scene
        raise RuntimeError("Could not sample a valid scene; relax the scene configuration.")


# ============================================================================ configurable
class ConfigurableSceneGenerator(_SceneChecks):
    """Scene from a YAML/dict spec with per-obstacle motion.

    spec:
      name: industrial_cell
      obstacles:
        - {p1: [x,y,z], p2: [x,y,z], radius: 0.05, noise: 0.03, motion: {...}}
        - {center: [x,y,z], radius: 0.1}                      # sphere (p1 == p2)
        - {box: {center: [..], size: [sx,sy,sz]}, radius: 0.02}  # 4 vertical capsules at the corners + top ring
      target:
        radius: 0.03
        region: {center: [x,y,z], extent: [ex,ey,ez]}        # uniform box sample
        points: [[x,y,z], ...]                                # or a list of nominal pick points
        noise: 0.03
        motion: {...}
      randomize: {count: [min,max], region: {...}, radius: [..], length: [..], motion_pool: [...]}  # optional extra random obstacles
    """

    def __init__(self, arm, cfg: dict):
        super().__init__(arm, cfg)
        spec = cfg.get("spec")
        if spec is None:
            preset = cfg.get("preset")
            if preset is None:
                raise ValueError("ConfigurableSceneGenerator needs scene.spec or scene.preset")
            path = Path(preset)
            if not path.exists():
                path = _PRESET_DIR / f"{preset}.yaml"
            spec = yaml.safe_load(open(path))
        self.spec = spec
        self.name = str(spec.get("name", "custom"))
        self.target_spec = dict(spec.get("target", {}))
        self.target_radius = float(self.target_spec.get("radius", cfg.get("target_radius", 0.03)))
        self.max_tries = 300

    def _expand_obstacle(self, o: dict, rng: np.random.Generator):
        """Return list of (p1, p2, radius, motion_spec) for one spec entry."""
        r = float(o.get("radius", 0.04))
        noise = float(o.get("noise", 0.0))
        shift = rng.uniform(0.0, noise) * _random_unit_3d(rng) if noise > 0 else np.zeros(3)
        motion = o.get("motion")
        if "box" in o:
            c = np.asarray(o["box"]["center"], dtype=np.float64) + shift
            sx, sy, sz = np.asarray(o["box"]["size"], dtype=np.float64) / 2.0
            corners = [c + [sx, sy, 0], c + [-sx, sy, 0], c + [-sx, -sy, 0], c + [sx, -sy, 0]]
            out = []
            for k in range(4):
                lo = corners[k] - [0, 0, sz]
                hi = corners[k] + [0, 0, sz]
                out.append((lo, hi, r, motion))
                out.append((corners[k] + [0, 0, sz], corners[(k + 1) % 4] + [0, 0, sz], r, motion))
            return out
        if "center" in o:
            c = np.asarray(o["center"], dtype=np.float64) + shift
            return [(c, c.copy(), r, motion)]
        p1 = np.asarray(o["p1"], dtype=np.float64) + shift
        p2 = np.asarray(o["p2"], dtype=np.float64) + shift
        return [(p1, p2, r, motion)]

    def _random_extras(self, rng: np.random.Generator):
        rs = self.spec.get("randomize")
        if not rs:
            return []
        lo, hi = rs.get("count", [0, 0])
        n = int(rng.integers(lo, hi + 1))
        reg = rs["region"]
        c = np.asarray(reg["center"], dtype=np.float64)
        e = np.asarray(reg["extent"], dtype=np.float64) / 2.0
        rr = rs.get("radius", [0.03, 0.06])
        ll = rs.get("length", [0.1, 0.4])
        pool = rs.get("motion_pool", [None])
        out = []
        for _ in range(n):
            p1 = c + rng.uniform(-e, e)
            d = _random_unit_3d(rng)
            p2 = p1 + rng.uniform(*ll) * d
            p2[2] = max(p2[2], 0.02)
            motion = pool[int(rng.integers(len(pool)))]
            out.append((p1, p2, float(rng.uniform(*rr)), motion))
        return out

    def _sample_target(self, rng: np.random.Generator) -> np.ndarray:
        ts = self.target_spec
        noise = float(ts.get("noise", 0.0))
        if "points" in ts:
            pts = np.asarray(ts["points"], dtype=np.float64)
            base = pts[int(rng.integers(len(pts)))]
        elif "region" in ts:
            c = np.asarray(ts["region"]["center"], dtype=np.float64)
            e = np.asarray(ts["region"]["extent"], dtype=np.float64) / 2.0
            base = c + rng.uniform(-e, e)
        else:
            base = np.asarray(ts.get("position", [0.4, 0.0, 0.3]), dtype=np.float64)
        return base + (rng.uniform(0.0, noise) * _random_unit_3d(rng) if noise > 0 else 0.0)

    def sample(self, rng: np.random.Generator) -> Scene:
        for _ in range(self.max_tries):
            items = []
            for o in self.spec.get("obstacles", []):
                items += self._expand_obstacle(o, rng)
            items += self._random_extras(rng)
            q1 = np.asarray([it[0] for it in items]).reshape(-1, 3)
            q2 = np.asarray([it[1] for it in items]).reshape(-1, 3)
            radii = np.asarray([it[2] for it in items], dtype=np.float64)
            motions = [Motion(it[3], rng) for it in items]
            target = self._sample_target(rng)
            if not self.target_is_valid(target, q1, q2, radii):
                continue
            tmotion = Motion(self.target_spec.get("motion"), rng)
            scene = Scene(q1, q2, radii, target, self.target_radius, obs_motion=motions, target_motion=tmotion, name=self.name)
            # the target must stay valid along its motion and the home pose free along the obstacle motion
            ok = True
            if scene.is_dynamic:
                probe = scene.copy()
                for t in np.linspace(0.0, 10.0, 21):
                    probe.set_time(float(t))
                    if not self.target_is_valid(probe.target, probe.obs_q1, probe.obs_q2, probe.radii) or not self.arm_is_collision_free(self.arm.home, probe):
                        ok = False
                        break
            elif not self.arm_is_collision_free(self.arm.home, scene):
                ok = False
            if ok:
                return scene
        raise RuntimeError(f"Could not sample a valid '{self.name}' scene; relax the spec.")


def make_scene_generator(arm, cfg: dict):
    cfg = dict(cfg or {})
    if cfg.get("spec") is not None or cfg.get("preset") is not None:
        return ConfigurableSceneGenerator(arm, cfg)
    return FruitTreeSceneGenerator(arm, cfg)
