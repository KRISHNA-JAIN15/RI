"""Train DDPG on the multi-obstacle arm environment (paper Sec. IV-B).

Usage:
    python scripts/train.py --config configs/default.yaml --out runs/ddpg
    python scripts/train.py --episodes 500 --out runs/smoke      # quick check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.ddpg import DDPGAgent  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.export import export_checkpoint, viewer_scenes, write_manifest  # noqa: E402
from arm_rl.utils import CSVLogger, load_config, pick_device, save_config, set_seed, write_json  # noqa: E402
from arm_rl.visualize import plot_training_curve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="runs/ddpg")
    p.add_argument("--episodes", type=int, default=None, help="override train.episodes")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume weights from")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value, e.g. --set ddpg.gamma=0.9 (repeatable)")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    import yaml

    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.episodes is not None:
        cfg["train"]["episodes"] = args.episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg["seed"])
    set_seed(seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    env.reset(seed=seed)
    agent = DDPGAgent(env.state_dim, env.action_space.shape[0], cfg["ddpg"], device=device, seed=seed)
    if args.resume:
        agent.load(args.resume)

    ddpg_cfg = cfg["ddpg"]
    tr = cfg["train"]
    n_episodes = int(tr["episodes"])
    max_delta_deg = float(cfg["env"]["max_delta_deg"])
    noise_start = float(ddpg_cfg.get("action_noise_deg_start", 2.0)) / max_delta_deg
    noise_end = float(ddpg_cfg.get("action_noise_deg_end", 0.0)) / max_delta_deg

    # 3-D viewer: fixed scenes replayed at every checkpoint
    n_viewer = int(tr.get("viewer_episodes", 4))
    viewer_dir = out / "viewer"
    viewer_entries = []
    scenes_for_viewer = viewer_scenes(env, n_viewer) if n_viewer > 0 else []

    def export_for_viewer(episode: int, done: bool = False):
        if n_viewer <= 0:
            return
        agent.actor.eval()
        viewer_entries.append(export_checkpoint(env, agent, scenes_for_viewer, episode, viewer_dir))
        agent.actor.train()
        write_manifest(viewer_dir, out.name, env.arm.link_radius, viewer_entries, out / "train_log.csv", done, env.backend.name, env.arm.reach)

    logger = CSVLogger(
        out / "train_log.csv",
        ["episode", "return", "steps", "reached", "collisions", "final_distance", "noise_deg",
         "critic_loss", "actor_loss", "q_mean", "updates", "wall_time_s"],
    )
    t_start = time.time()
    total_steps = 0
    recent_returns, recent_success = [], []
    pbar = tqdm(range(1, n_episodes + 1), desc="train", dynamic_ncols=True)
    for episode in pbar:
        frac = (episode - 1) / max(1, n_episodes - 1)
        noise_std = noise_start + (noise_end - noise_start) * frac
        obs, info = env.reset()
        ep_return, ep_steps, reached = 0.0, 0, False
        losses = []
        while True:
            if agent.ready():
                action = agent.act(obs, noise_std)
            else:
                action = agent.random_action()
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Only true terminal states (target touched / terminal collision) stop bootstrapping.
            agent.store(obs, action, reward, next_obs, terminated)
            obs = next_obs
            ep_return += reward
            ep_steps += 1
            total_steps += 1
            if agent.ready():
                losses.append(agent.update())
            if terminated or truncated:
                reached = bool(info["reached"])
                break
        recent_returns.append(ep_return)
        recent_success.append(float(reached))
        row = {
            "episode": episode,
            "return": ep_return,
            "steps": ep_steps,
            "reached": int(reached),
            "collisions": info["n_collisions"],
            "final_distance": info["distance"],
            "noise_deg": noise_std * max_delta_deg,
            "updates": agent.n_updates,
            "wall_time_s": time.time() - t_start,
        }
        if losses:
            row["critic_loss"] = float(np.mean([l["critic_loss"] for l in losses]))
            row["actor_loss"] = float(np.mean([l["actor_loss"] for l in losses]))
            row["q_mean"] = float(np.mean([l["q_mean"] for l in losses]))
        logger.log(row)
        if episode % int(tr.get("log_every", 50)) == 0:
            k = int(tr.get("log_every", 50))
            pbar.set_postfix(
                ret=f"{np.mean(recent_returns[-k:]):.1f}",
                succ=f"{100 * np.mean(recent_success[-k:]):.0f}%",
                noise=f"{noise_std * max_delta_deg:.2f}",
                steps=total_steps,
            )
        if episode % int(tr.get("checkpoint_every", 500)) == 0:
            agent.save(out / "checkpoints" / f"ddpg_ep{episode:06d}.pt")
            agent.save(out / "ddpg_final.pt")
            export_for_viewer(episode)
    agent.save(out / "ddpg_final.pt")
    logger.close()
    if n_episodes % int(tr.get("checkpoint_every", 500)) != 0:
        export_for_viewer(n_episodes)
    if viewer_entries:
        write_manifest(viewer_dir, out.name, env.arm.link_radius, viewer_entries, out / "train_log.csv", done=True, backend=env.backend.name, reach=env.arm.reach)
    plot_training_curve(out / "train_log.csv", out / "fig4_training_curve.png", int(tr.get("reward_curve_bin", 100)))
    write_json(
        {
            "episodes": n_episodes,
            "total_env_steps": total_steps,
            "updates": agent.n_updates,
            "wall_time_s": time.time() - t_start,
            "final_100_mean_return": float(np.mean(recent_returns[-100:])),
            "final_100_success_rate": float(np.mean(recent_success[-100:])),
        },
        out / "train_summary.json",
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
