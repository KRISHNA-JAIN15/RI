"""Render one episode of a trained policy as a sequence of PNG frames + GIF (Fig. 5).

Usage: python scripts/demo.py --checkpoint runs/ddpg/ddpg_final.pt --out runs/ddpg/demo --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.ddpg import DDPGAgent  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.utils import load_config  # noqa: E402
from arm_rl.visualize import render_scene  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default=None, help="omit for a random policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--every", type=int, default=5, help="render every k-th step")
    args = p.parse_args()

    cfg = load_config(args.config)
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    obs, info = env.reset(seed=args.seed)
    agent = None
    if args.checkpoint:
        agent = DDPGAgent(env.state_dim, env.action_space.shape[0], cfg["ddpg"], seed=args.seed)
        agent.load(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    step = 0
    while True:
        if step % args.every == 0:
            frames.append(render_scene(env.arm, env.scene, env.q, np.asarray(env.ee_history), f"step {step}  d={info['distance']:.3f} m"))
        action = agent.act(obs) if agent else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            frames.append(render_scene(env.arm, env.scene, env.q, np.asarray(env.ee_history),
                                       f"step {step}  d={info['distance']:.3f} m  reached={info['reached']}", out / "final.png"))
            break
    try:
        from matplotlib import animation
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(frames[0])
        plt.axis("off")
        anim = animation.FuncAnimation(fig, lambda i: im.set_data(frames[i]), frames=len(frames), interval=200)
        anim.save(out / "episode.gif", writer="pillow", fps=5)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print("gif export skipped:", exc)
    print(f"steps={step} reached={info['reached']} collisions={info['n_collisions']} final_distance={info['distance']:.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
