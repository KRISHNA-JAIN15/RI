"""Matplotlib stand-in for the ROS RViz visualisation (Fig. 3-5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from arm_rl.kinematics import SixAxisArm  # noqa: E402
from arm_rl.scene import Scene  # noqa: E402


def _set_axes_equal(ax, lim=0.8):
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.1, 1.1, 1.0))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def draw_scene(ax, arm: SixAxisArm, scene: Scene, q: np.ndarray, ee_history: np.ndarray | None = None,
               title: str | None = None):
    # obstacles (green cylinders drawn as thick lines)
    for a, b in zip(scene.obs_q1, scene.obs_q2):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="green", linewidth=8, solid_capstyle="round", alpha=0.8)
    # target (red sphere)
    ax.scatter(*scene.target, color="red", s=200, depthshade=False)
    # arm (blue cylinders)
    origins, _ = arm.forward(q)
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], color="royalblue", linewidth=7, solid_capstyle="round")
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], color="navy", s=25)
    if ee_history is not None and len(ee_history) > 1:
        ax.plot(ee_history[:, 0], ee_history[:, 1], ee_history[:, 2], color="orange", linewidth=1.5, marker=".", markersize=3)
    _set_axes_equal(ax)
    if title:
        ax.set_title(title)


def render_scene(arm: SixAxisArm, scene: Scene, q: np.ndarray, ee_history: np.ndarray | None = None,
                 title: str | None = None, save_path: str | Path | None = None) -> np.ndarray:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene(ax, arm, scene, q, ee_history, title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def plot_training_curve(csv_path: str | Path, save_path: str | Path, bin_size: int = 100) -> None:
    """Fig. 4: average (cumulative) reward per round, averaged over bins of rounds."""
    import csv

    episodes, returns = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
    ep = np.asarray(episodes)
    ret = np.asarray(returns)
    n_bins = max(1, len(ret) // bin_size)
    xs = [ep[i * bin_size] for i in range(n_bins)]
    ys = [ret[i * bin_size:(i + 1) * bin_size].mean() for i in range(n_bins)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, label="DDPG")
    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Average reward")
    ax.set_title("Deep reinforcement learning")
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ee_path(ee_history: np.ndarray, target: np.ndarray, save_path: str | Path, title: str = "") -> None:
    """Fig. 5 style end-effector trajectory."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_history[:, 0], ee_history[:, 1], ee_history[:, 2], color="tab:blue", marker="o", markersize=3)
    ax.scatter(*ee_history[0], color="tab:green", s=60, label="start")
    ax.scatter(*target, color="red", s=80, label="target")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
