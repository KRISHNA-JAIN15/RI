"""Evaluate DDPG and RRT over N rounds and reproduce Table I.

Usage:
    python scripts/evaluate.py --checkpoint runs/ddpg/ddpg_final.pt --episodes 1000 --out runs/ddpg/eval
    python scripts/evaluate.py --checkpoint ... --no-rrt      # DDPG only
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.agents import make_agent  # noqa: E402
from arm_rl.plugins import load_plugins  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.metrics import PAPER_TABLE_I, EpisodeResult, format_table, summarize  # noqa: E402
from arm_rl.rrt import RRTPlanner  # noqa: E402
from arm_rl.utils import load_config, pick_device, set_seed, write_json  # noqa: E402
from arm_rl.visualize import plot_ee_path, render_scene  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="defaults to <checkpoint dir>/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345, help="evaluation seed (scenes are shared by both planners)")
    p.add_argument("--out", default=None)
    p.add_argument("--no-rrt", action="store_true")
    p.add_argument("--rrt-episodes", type=int, default=None, help="RRT rounds (defaults to --episodes)")
    p.add_argument("--n-figures", type=int, default=3, help="number of Fig. 5 style trajectory plots")
    p.add_argument("--save-paths", action="store_true", help="write paths.npz (per-episode EE trajectories, NaN-padded, and targets) to --out")
    return p.parse_args()


def pad_paths(paths: list) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length (T_i, 3) EE paths into (N, T_max, 3) padded with NaN, plus (N,) lengths."""
    lengths = np.array([len(p) for p in paths], dtype=np.int64)
    out = np.full((len(paths), int(lengths.max()) if len(paths) else 0, 3), np.nan)
    for i, p in enumerate(paths):
        out[i, : len(p)] = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    return out, lengths


def run_ddpg_episode(env: ArmPathPlanningEnv, agent, scene, plugins: list | None = None) -> tuple[EpisodeResult, np.ndarray]:
    obs, info = env.reset(options={"scene": scene})
    infer_time = 0.0
    ep_return = 0.0
    min_clear = float("inf")
    while True:
        t0 = time.perf_counter()
        action = agent.act(obs, 0.0)
        for p in plugins or []:
            action = p.transform_action(obs, action, deterministic=True)
        infer_time += time.perf_counter() - t0
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        min_clear = min(min_clear, env.min_obstacle_clearance())
        if terminated or truncated:
            break
    collided = info["n_collisions"] > 0
    success = bool(info["reached"]) and not collided
    res = EpisodeResult(success, info["ee_path_length"], infer_time * 1000.0, info["distance"], collided, info["step"], ep_return, min_clear)
    return res, np.asarray(env.ee_history)


def execute_path(env: ArmPathPlanningEnv, scene, path: list[np.ndarray]) -> dict:
    """Replay a joint-space path as +-5 deg increments inside the live environment, so moving
    obstacles / targets act on the RRT plan exactly as they act on the policy."""
    obs, info = env.reset(options={"scene": scene})
    for q_next in path[1:]:
        a = (q_next - env.q) / env.max_delta
        obs, r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
        if term or trunc:
            break
    collided = info["n_collisions"] > 0
    return {"success": bool(info["reached"]) and not collided, "collided": collided, "path_length": info["ee_path_length"],
            "final_error": info["distance"], "steps": info["step"]}


def main():
    args = parse_args()
    cfg = load_config(args.config or Path(args.checkpoint).parent / "config.yaml")
    n = int(args.episodes or cfg["eval"]["episodes"])
    n_rrt = int(args.rrt_episodes or n)
    out = Path(args.out or Path(args.checkpoint).parent / "eval")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(1)

    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    agent = make_agent(env.state_dim, env.action_space.shape[0], cfg, device=pick_device(), seed=args.seed, env=env)
    agent.load(args.checkpoint)
    agent.actor.eval()
    plugins = load_plugins(cfg, env, agent, phase="eval")

    # Same scenes for both planners.
    scene_rng = np.random.default_rng(args.seed)
    scenes = [env.scene_gen.sample(scene_rng) for _ in range(max(n, n_rrt))]

    # ---------------- DDPG
    ddpg_results = []
    ddpg_paths = []
    figs_done = 0
    for i in tqdm(range(n), desc="DDPG eval", dynamic_ncols=True):
        res, ee_hist = run_ddpg_episode(env, agent, scenes[i], plugins)
        ddpg_results.append(res)
        ddpg_paths.append(ee_hist)
        if res.success and figs_done < args.n_figures:
            plot_ee_path(ee_hist, scenes[i].target, out / f"fig5_ddpg_path_{figs_done}.png", f"DDPG round {i}")
            render_scene(env.arm, scenes[i], env.q, ee_hist, f"DDPG round {i}", out / f"scene_ddpg_{figs_done}.png")
            figs_done += 1
    rows = [summarize(ddpg_results, cfg.get("method", {}).get("label", "DDPG (ours)"))]
    write_json([asdict(r) for r in ddpg_results], out / "ddpg_episodes.json")

    # ---------------- RRT
    if not args.no_rrt:
        planner = RRTPlanner(env, cfg["eval"]["rrt"], seed=args.seed)
        rrt_results = []
        rrt_paths = []
        for i in tqdm(range(n_rrt), desc="RRT eval", dynamic_ncols=True):
            env.reset(options={"scene": scenes[i]})  # plan on the t = 0 snapshot
            r = planner.plan(env.arm.home, scenes[i].target)
            if r.success:
                ex = execute_path(env, scenes[i], r.path)
                rrt_results.append(EpisodeResult(ex["success"], ex["path_length"], r.time_ms, ex["final_error"], ex["collided"], ex["steps"], 0.0))
                rrt_paths.append(np.asarray(env.ee_history))  # the executed (replayed) EE trajectory
            else:
                rrt_results.append(EpisodeResult(False, r.ee_path_length, r.time_ms, r.final_error, False, len(r.path), 0.0))
                rrt_paths.append(np.asarray(r.ee_path).reshape(-1, 3))
            if r.success and i < args.n_figures:
                plot_ee_path(r.ee_path, scenes[i].target, out / f"fig5_rrt_path_{i}.png", f"RRT round {i}")
        rows.append(summarize(rrt_results, "RRT (ours)"))
        write_json([asdict(r) for r in rrt_results], out / "rrt_episodes.json")

    if args.save_paths:
        arrays = {"targets": np.asarray([sc.target for sc in scenes]), "seed": np.array(args.seed)}
        arrays["ddpg_paths"], arrays["ddpg_lengths"] = pad_paths(ddpg_paths)
        if not args.no_rrt:
            arrays["rrt_paths"], arrays["rrt_lengths"] = pad_paths(rrt_paths)
        np.savez_compressed(out / "paths.npz", **arrays)

    rows += PAPER_TABLE_I
    table = format_table(rows)
    print("\nTABLE I. ALGORITHM PERFORMANCE\n" + table)
    (out / "table1.md").write_text("TABLE I. ALGORITHM PERFORMANCE\n\n" + table + "\n")
    write_json(rows, out / "table1.json")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
