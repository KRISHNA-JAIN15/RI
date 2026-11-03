"""Closest points between cylinder axes (paper eq. 5-7).

Each arm link and each obstacle is simplified to a cylinder (Fig. 1).  The shortest
distance between two cylinders is the shortest distance between their axis segments
minus the two radii, and the *direction vector* of that shortest distance is
V = Q' - P' where P' lies on the link axis and Q' on the obstacle axis.

The bivariate minimisation of D(t, s) = ||P(t) - Q(s)|| over t, s in [0, 1] is solved
analytically (clamped least squares, Ericson "Real-Time Collision Detection" 5.1.9)
instead of by gradient descent; the result is identical and exact.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def closest_points_between_segments(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closest points between segments P(t)=p1+t(p2-p1) and Q(s)=q1+s(q2-q1).

    All inputs broadcast against each other over leading dimensions and end with a
    trailing axis of size 3.  Returns ``(t, s, P_closest, Q_closest, distance)``.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    d1 = p2 - p1  # direction of segment P
    d2 = q2 - q1  # direction of segment Q
    r = p1 - q1
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)

    a, e, f, c, b = np.broadcast_arrays(a, e, f, c, b)
    a_deg = a < _EPS
    e_deg = e < _EPS
    a_safe = np.where(a_deg, 1.0, a)
    e_safe = np.where(e_deg, 1.0, e)

    denom = a_safe * e_safe - b * b
    # General (non-parallel) case: unconstrained minimiser of t, then clamp.
    t = np.where(denom > _EPS, (b * f - c * e_safe) / np.where(denom > _EPS, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    # Closest point on Q to P(t)
    s = (b * t + f) / e_safe
    # If s is outside [0, 1], clamp it and recompute t for the clamped s.
    s_lo = s < 0.0
    s_hi = s > 1.0
    t = np.where(s_lo, np.clip(-c / a_safe, 0.0, 1.0), t)
    t = np.where(s_hi, np.clip((b - c) / a_safe, 0.0, 1.0), t)
    s = np.clip(s, 0.0, 1.0)
    # Degenerate segments (zero length) collapse to a point.
    t = np.where(a_deg, 0.0, t)
    s = np.where(e_deg, 0.0, s)
    if np.any(a_deg & ~e_deg):
        s = np.where(a_deg & ~e_deg, np.clip(f / e_safe, 0.0, 1.0), s)

    p_closest = p1 + t[..., None] * d1
    q_closest = q1 + s[..., None] * d2
    dist = np.linalg.norm(q_closest - p_closest, axis=-1)
    return t, s, p_closest, q_closest, dist


def segment_distance_matrix(
    link_p1: np.ndarray, link_p2: np.ndarray, obs_q1: np.ndarray, obs_q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise axis distances and direction vectors between L links and M obstacles.

    Args:
        link_p1, link_p2: ``(L, 3)`` end points of the link axes.
        obs_q1, obs_q2:   ``(M, 3)`` end points of the obstacle axes.
    Returns:
        ``dist`` of shape ``(L, M)`` and ``direction`` of shape ``(L, M, 3)`` where
        ``direction[i, j] = Q' - P'`` (from link i towards obstacle j).
    """
    p1 = link_p1[:, None, :]
    p2 = link_p2[:, None, :]
    q1 = obs_q1[None, :, :]
    q2 = obs_q2[None, :, :]
    _, _, pc, qc, dist = closest_points_between_segments(p1, p2, q1, q2)
    return dist, qc - pc


def point_segment_distance(point: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Distance from ``point`` ``(3,)`` to segments ``q1,q2`` of shape ``(M, 3)``."""
    d = q2 - q1
    e = np.sum(d * d, axis=-1)
    s = np.sum((point[None, :] - q1) * d, axis=-1) / np.where(e < _EPS, 1.0, e)
    s = np.clip(np.where(e < _EPS, 0.0, s), 0.0, 1.0)
    closest = q1 + s[:, None] * d
    return np.linalg.norm(closest - point[None, :], axis=-1)
