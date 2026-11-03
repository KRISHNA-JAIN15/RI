"""Export deterministic policy rollouts for the 3-D web viewer.

For every checkpoint the same fixed set of scenes is replayed so the viewer can show how
the policy improves on identical tasks across training.  Each checkpoint becomes one JSON
file (`viewer/ckpt_XXXXXX.json`) and `viewer/manifest.json` indexes them together with the
training curve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from arm_rl.ddpg import DDPGAgent
from arm_rl.env import ArmPathPlanningEnv
from arm_rl.scene import Scene

VIEWER_SCENE_SEED = 777_000


def _r(x, nd: int = 4):
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def viewer_scenes(env: ArmPathPlanningEnv, n: int) -> list[Scene]:
    rng = np.random.default_rng(VIEWER_SCENE_SEED)
    return [env.scene_gen.sample(rng) for _ in range(n)]


def _closest_point_on_link(env, origins, link: int, obs: int, direction: np.ndarray, capsules) -> np.ndarray:
    """P' (closest point on the link) so that Q' = P' + V can be drawn in the viewer."""
    if capsules is None:
        from arm_rl.geometry import closest_points_between_segments

        sc = env.scene
        _, _, pc, _, _ = closest_points_between_segments(origins[link], origins[link + 1], sc.obs_q1[obs], sc.obs_q2[obs])
        return pc
    # MuJoCo backend: fromto is not kept per pair; approximate P' by the closest point of the
    # link's capsule axes to the obstacle axis.
    from arm_rl.geometry import closest_points_between_segments

    sc = env.scene
    best, bp = np.inf, origins[link]
    for c in capsules:
        if c["link"] != link:
            continue
        _, _, pc, _, d = closest_points_between_segments(np.asarray(c["a"]), np.asarray(c["b"]), sc.obs_q1[obs], sc.obs_q2[obs])
        if d < best:
            best, bp = d, pc
    return bp


def rollout_episode(env: ArmPathPlanningEnv, agent: DDPGAgent | None, scene: Scene, rng: np.random.Generator | None = None, plugins: list | None = None) -> dict:
    """Run one deterministic episode and record everything the viewer draws."""
    obs, info = env.reset(options={"scene": scene})
    frames = []
    ret = 0.0

    def frame(reward: float, collision: bool, reached: bool):
        sc = env.scene
        origins, clearance, direction, capsules = env._geometry()
        sl = []
        if clearance.shape[1]:
            for i in env.monitored_links:
                j = int(np.argmin(clearance[i]))
                p_close = _closest_point_on_link(env, origins, i, j, direction[i, j], capsules)
                sl.append({"p": _r(p_close), "q": _r(p_close + direction[i, j]), "d": round(float(clearance[i, j]), 4)})
        fr = {
            "q": _r(env.q),
            "origins": _r(origins),
            "ee": _r(env.ee),
            "reward": round(float(reward), 3),
            "return": round(float(ret), 3),
            "distance": round(float(np.linalg.norm(env.ee - sc.target)), 4),
            "collision": bool(collision),
            "reached": bool(reached),
            "sl": sl,
            "t": round(float(sc.t), 3),
        }
        if sc.is_dynamic:
            fr["obstacles"] = [{"p1": _r(a), "p2": _r(b)} for a, b in zip(sc.obs_q1, sc.obs_q2)]
            fr["target"] = _r(sc.target)
        if capsules is not None:
            fr["capsules"] = capsules
        frames.append(fr)

    frame(0.0, False, bool(info["reached"]))
    while True:
        action = agent.act(obs, 0.0) if agent is not None else env.action_space.sample()
        for p in plugins or []:
            action = p.transform_action(obs, action, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        frame(reward, info["collision"], info["reached"])
        if terminated or truncated:
            break
    collided = info["n_collisions"] > 0
    sc0 = env.scene
    return {
        "scene": {
            "name": sc0.name,
            "obstacles": [{"p1": _r(a), "p2": _r(b), "r": round(float(r), 4)} for a, b, r in zip(sc0.base_q1, sc0.base_q2, sc0.radii)],
            "obs_radius": float(np.mean(sc0.radii)) if sc0.n_obstacles else 0.04,
            "target": _r(sc0.base_target),
            "target_radius": sc0.target_radius,
            "dynamic": bool(sc0.is_dynamic),
        },
        "frames": frames,
        "steps": int(info["step"]),
        "success": bool(info["reached"]) and not collided,
        "collided": bool(collided),
        "return": round(float(ret), 3),
        "final_distance": round(float(info["distance"]), 4),
        "path_length": round(float(info["ee_path_length"]), 4),
    }


def export_checkpoint(env: ArmPathPlanningEnv, agent: DDPGAgent | None, scenes: list[Scene], episode: int, out_dir: Path, plugins: list | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = [rollout_episode(env, agent, sc, plugins=plugins) for sc in scenes]
    payload = {"episode_trained": int(episode), "episodes": episodes}
    fname = f"ckpt_{episode:06d}.json"
    (out_dir / fname).write_text(json.dumps(payload, separators=(",", ":")))
    return {
        "episode": int(episode),
        "file": fname,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "mean_final_distance": float(np.mean([e["final_distance"] for e in episodes])),
    }


def training_curve(train_log: Path, bin_size: int = 50) -> list[dict]:
    if not train_log.exists():
        return []
    rows = list(csv.DictReader(open(train_log)))
    curve = []
    for i in range(0, len(rows), bin_size):
        chunk = rows[i:i + bin_size]
        if len(chunk) < bin_size and i > 0:
            break
        curve.append(
            {
                "episode": int(chunk[-1]["episode"]),
                "return": float(np.mean([float(r["return"]) for r in chunk])),
                "success": float(np.mean([int(r["reached"]) for r in chunk])),
            }
        )
    return curve


def write_manifest(out_dir: Path, run_name: str, link_radius: float, checkpoints: list[dict], train_log: Path, done: bool = False, backend: str = "analytic", reach: float = 0.612) -> None:
    manifest = {
        "run": run_name,
        "backend": backend,
        "link_radius": float(link_radius),
        "reach": float(reach),
        "checkpoints": sorted(checkpoints, key=lambda c: c["episode"]),
        "curve": training_curve(train_log),
        "done": bool(done),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
