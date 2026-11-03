"""Build viewer data (rollouts per checkpoint + manifest) for an existing run.

Usage: python scripts/export_viewer.py --run runs/ddpg_paper [--episodes 4] [--random-baseline]
Then:  python scripts/serve_viewer.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.ddpg import DDPGAgent  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.export import export_checkpoint, viewer_scenes, write_manifest  # noqa: E402
from arm_rl.utils import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=None, help="rollouts per checkpoint (default train.viewer_episodes)")
    p.add_argument("--random-baseline", action="store_true", help="also export an untrained (episode 0) random policy")
    args = p.parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    n = int(args.episodes or cfg["train"].get("viewer_episodes", 4))
    torch.set_num_threads(1)

    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    env.reset(seed=0)
    scenes = viewer_scenes(env, n)
    agent = DDPGAgent(env.state_dim, env.action_space.shape[0], cfg["ddpg"], seed=0)
    out = run / "viewer"
    entries = []
    if args.random_baseline:
        entries.append(export_checkpoint(env, None, scenes, 0, out))
    ckpts = sorted(run.glob("checkpoints/ddpg_ep*.pt"))
    for ck in tqdm(ckpts, desc="export", dynamic_ncols=True):
        ep = int(re.search(r"ep(\d+)", ck.name).group(1))
        agent.load(ck)
        agent.actor.eval()
        entries.append(export_checkpoint(env, agent, scenes, ep, out))
    write_manifest(out, run.name, env.arm.link_radius, entries, run / "train_log.csv", done=True, backend=env.backend.name, reach=env.arm.reach)
    print(f"{len(entries)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
