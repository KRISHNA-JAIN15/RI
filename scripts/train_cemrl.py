"""Train CEM-RL (C4: population of actors + gradient learner) on the arm environment.

Produces the same artifacts as scripts/train.py (config.yaml, train_log.csv with one row per
environment episode consumed by the population, train_summary.json, checkpoints/ddpg_epXXXXXX.pt,
ddpg_final.pt, viewer/), so scripts/evaluate.py and the benchmark consume the run unchanged.

Usage:
    python scripts/train_cemrl.py --config configs/default.yaml --out runs/cemrl
    python scripts/train_cemrl.py --config configs/default.yaml --set method.cem.pop_size=20 --set method.algo=td3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.agents import make_agent  # noqa: E402
from arm_rl.cem_rl import CEMRLTrainer, cem_config  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.export import export_checkpoint, viewer_scenes, write_manifest  # noqa: E402
from arm_rl.plugins import load_plugins  # noqa: E402
from arm_rl.utils import CSVLogger, load_config, pick_device, save_config, set_seed, write_json  # noqa: E402
from arm_rl.visualize import plot_training_curve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="runs/cemrl")
    p.add_argument("--episodes", type=int, default=None, help="override train.episodes (environment episodes for the whole population)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None, help="checkpoint whose actor initialises the CEM mean")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value, e.g. --set method.cem.pop_size=20 (repeatable)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
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
    cfg.setdefault("method", {})
    cfg["method"]["cem"] = cem_config(cfg)  # record the resolved CEM settings in config.yaml
    if not any(o.startswith("method.label=") for o in args.set):
        cfg["method"]["label"] = f"CEM-RL ({str(cfg['method'].get('algo', 'ddpg')).upper()})"
    seed = int(cfg["seed"])
    set_seed(seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    env.reset(seed=seed)
    agent = make_agent(env.state_dim, env.action_space.shape[0], cfg, device=device, seed=seed, env=env)
    if args.resume:
        agent.load(args.resume)
    plugins = load_plugins(cfg, env, agent, phase="train")

    tr = cfg["train"]
    logger = CSVLogger(out / "train_log.csv", CEMRLTrainer.CSV_FIELDS)
    trainer = CEMRLTrainer(cfg, env, agent, plugins, out, seed=seed, quiet=args.quiet, logger=logger)

    # 3-D viewer: fixed scenes replayed at every checkpoint with the checkpoint actor
    n_viewer = int(tr.get("viewer_episodes", 4))
    viewer_dir = out / "viewer"
    viewer_entries: list[dict] = []
    scenes_for_viewer = viewer_scenes(env, n_viewer) if n_viewer > 0 else []

    def export_for_viewer(episode: int):
        if n_viewer <= 0:
            return
        trainer._load_into_agent(trainer.checkpoint_params())
        agent.actor.eval()
        viewer_entries.append(export_checkpoint(env, agent, scenes_for_viewer, episode, viewer_dir, plugins=plugins))
        agent.actor.train()
        write_manifest(viewer_dir, out.name, env.arm.link_radius, viewer_entries, out / "train_log.csv", False, env.backend.name, env.arm.reach)

    trainer.on_checkpoint = export_for_viewer

    t0 = time.time()
    summary = trainer.train()
    logger.close()
    if viewer_entries:
        write_manifest(viewer_dir, out.name, env.arm.link_radius, viewer_entries, out / "train_log.csv", done=True, backend=env.backend.name, reach=env.arm.reach)
    plot_training_curve(out / "train_log.csv", out / "fig4_training_curve.png", int(tr.get("reward_curve_bin", 100)))
    summary["wall_time_s"] = time.time() - t0
    write_json(summary, out / "train_summary.json")
    if not args.quiet:
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
