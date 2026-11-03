"""Forward kinematics of a six-axis arm (standard Denavit-Hartenberg convention).

Each joint rotates about the z-axis of the previous frame.  The arm is simplified to
six cylinders (Fig. 1): link ``i`` is the segment between frame origins ``o_i`` and
``o_{i+1}``.  Link 0 is the fixed base pedestal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass
class SixAxisArm:
    """Six revolute joints; UR-style DH table scaled to a 61.2 cm reach."""

    d1: float = 0.150
    a2: float = 0.260
    a3: float = 0.230
    d4: float = 0.100
    d5: float = 0.080
    d6: float = 0.0782
    link_radius: float = 0.04
    joint_limits_deg: float = 170.0
    home_deg: list[float] = field(default_factory=lambda: [0.0, 90.0, -135.0, 45.0, 90.0, 0.0])

    n_joints: int = 6

    def __post_init__(self) -> None:
        self.alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
        self.a = np.array([0.0, self.a2, self.a3, 0.0, 0.0, 0.0])
        self.d = np.array([self.d1, 0.0, 0.0, self.d4, self.d5, self.d6])
        lim = np.deg2rad(self.joint_limits_deg)
        self.joint_low = -np.full(self.n_joints, lim)
        self.joint_high = np.full(self.n_joints, lim)
        self.home = np.deg2rad(np.asarray(self.home_deg, dtype=np.float64))
        assert self.home.shape == (self.n_joints,)

    @classmethod
    def from_config(cls, cfg: dict) -> "SixAxisArm":
        return cls(**cfg)

    # ------------------------------------------------------------------ kinematics
    @property
    def reach(self) -> float:
        """Maximum shoulder-to-end-effector distance over all joint configurations.

        With the arm stretched (q3 = 0) the elbow chain spans sqrt((a2+a3)^2 + d4^2) and
        the wrist can align its d5/d6 offsets with that direction, adding sqrt(d5^2+d6^2).
        """
        return float(np.hypot(self.a2 + self.a3, self.d4) + np.hypot(self.d5, self.d6))

    @property
    def shoulder(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.d1])

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(origins, T_ee)``: frame origins ``(7, 3)`` and the EE pose ``(4, 4)``."""
        q = np.asarray(q, dtype=np.float64)
        T = np.eye(4)
        origins = np.zeros((self.n_joints + 1, 3))
        for i in range(self.n_joints):
            T = T @ dh_transform(q[i], self.d[i], self.a[i], self.alpha[i])
            origins[i + 1] = T[:3, 3]
        return origins, T

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        return self.forward(q)[1][:3, 3]

    def link_segments(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Axis end points of the six link cylinders: ``(p1 (6,3), p2 (6,3))``."""
        origins, _ = self.forward(q)
        return origins[:-1], origins[1:]

    def position_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerical ``(3, 6)`` Jacobian of the end-effector position."""
        q = np.asarray(q, dtype=np.float64)
        base = self.ee_position(q)
        J = np.zeros((3, self.n_joints))
        for i in range(self.n_joints):
            dq = np.zeros_like(q)
            dq[i] = eps
            J[:, i] = (self.ee_position(q + dq) - base) / eps
        return J

    def clip_joints(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_low, self.joint_high)

    def sample_joints(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.joint_low, self.joint_high)
