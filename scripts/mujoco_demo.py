"""Play a trained (or random) policy in the native MuJoCo viewer, or record a video.

Interactive (macOS needs mjpython, Linux/Windows plain python):
    mjpython scripts/mujoco_demo.py --config configs/ur5e_industrial.yaml --checkpoint runs/ur5e_industrial/ddpg_final.pt
Offscreen video / frames:
    python scripts/mujoco_demo.py --config configs/ur5e_industrial.yaml --checkpoint ... --video runs/ur5e_industrial/demo.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.ddpg import DDPGAgent  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.utils import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ur5e_industrial.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--video", default=None, help="write an .mp4 (needs imageio[ffmpeg]) or .gif instead of opening the viewer")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--realtime", type=float, default=1.0, help="playback speed factor for the interactive viewer")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["env"]["sim"]["backend"] = "mujoco"
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    agent = None
    if args.checkpoint:
        agent = DDPGAgent(env.state_dim, env.action_space.shape[0], cfg["ddpg"], seed=args.seed)
        agent.load(args.checkpoint)
        agent.actor.eval()
    be = env.backend
    model, data = be.model, be.data

    def run_episode(on_frame):
        obs, info = env.reset(seed=args.seed + run_episode.k)
        run_episode.k += 1
        on_frame(info)
        while True:
            a = agent.act(obs) if agent else env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            on_frame(info)
            if term or trunc:
                print(f"episode {run_episode.k}: steps={info['step']} reached={info['reached']} collisions={info['n_collisions']} d={info['distance']:.3f}")
                return
    run_episode.k = 0

    if args.video:
        renderer = mujoco.Renderer(model, height=720, width=1080)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.45, 0.0, 0.35]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 205, -25
        frames = []

        def grab(info):
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        for _ in range(args.episodes):
            run_episode(grab)
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".gif":
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
        else:
            import imageio.v2 as imageio

            imageio.mimwrite(out, frames, fps=args.fps)
        print(f"wrote {len(frames)} frames -> {out}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.35]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 205, -25

        def show(info):
            viewer.sync()
            time.sleep(env.dt / max(args.realtime, 1e-3))

        while viewer.is_running():
            run_episode(show)
            time.sleep(0.8)


if __name__ == "__main__":
    main()
