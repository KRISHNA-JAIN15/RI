"""Artificial potential field (Khatib 1986) computed from the paper's state (C2).

Attractive term (parabolic well near the target, conic beyond ``d_switch`` so the step
stays bounded):

    F_att = -k_att * (EE - T)                       if |EE - T| <= d_switch
    F_att = -k_att * d_switch * (EE - T) / |EE - T|  otherwise

Repulsive term per monitored link ``i`` with surface clearance ``rho_i`` to its nearest
obstacle and direction ``V_i`` (link -> obstacle):

    F_rep,i = k_rep * (1/rho_i - 1/rho0) * (1/rho_i^2) * (-V_i / |V_i|)   if rho_i < rho0
    F_rep,i = 0                                                           otherwise

Joint increment: ``dq = J_ee^T F_att + sum_i J_i^T F_rep,i`` where ``J_i`` is the position
Jacobian of the closest point on link ``i`` (finite differences through ``arm.forward``,
keeping the fractional position along the link).  ``dq`` is scaled down so that
``max |dq| <= dq_max``.

Potential for reward shaping (Ng, Harada & Russell 1999):

    Phi(s) = -k_phi * |EE - T| * distance_scale - k_phi_rep * sum_i max(0, 1/rho_i - 1/rho0)

evaluated from the observation vector alone (slices ``se``, ``st``, ``sl``).
"""

from __future__ import annotations

import numpy as np

from arm_rl.geometry import closest_points_between_segments

_EPS = 1e-9


# ---------------------------------------------------------------------------- forces
def attractive_force(ee: np.ndarray, target: np.ndarray, k_att: float, d_switch: float) -> np.ndarray:
    """Khatib's attractive force with a conic (constant magnitude) region beyond ``d_switch``."""
    e = np.asarray(ee, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    d = float(np.linalg.norm(e))
    if d <= d_switch or d < _EPS:
        return -k_att * e
    return -k_att * d_switch * e / d


def repulsive_force(rho: float, direction: np.ndarray, k_rep: float, rho0: float) -> np.ndarray:
    """Force pushing a link point away from its nearest obstacle (zero outside ``rho0``)."""
    v = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if rho >= rho0 or n < _EPS:
        return np.zeros(3)
    rho = max(float(rho), 1e-3)  # inside the surface: cap the magnitude instead of blowing up
    mag = k_rep * (1.0 / rho - 1.0 / rho0) / (rho * rho)
    return -mag * v / n


def repulsive_potential(rho: np.ndarray, rho0: float) -> float:
    """sum_i max(0, 1/rho_i - 1/rho0); ``rho`` entries <= 0 are clamped to 1 mm."""
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 1e-3)
    return float(np.sum(np.maximum(0.0, 1.0 / rho - 1.0 / rho0)))


# ---------------------------------------------------------------------------- Jacobians
def origins_jacobian(arm, q: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobian of every frame origin: ``(origins (K,3), J (K,3,n))``.

    One forward pass per joint, shared by every link point we need afterwards.
    """
    q = np.asarray(q, dtype=np.float64)
    base, _ = arm.forward(q)
    J = np.zeros((base.shape[0], 3, q.shape[0]))
    for j in range(q.shape[0]):
        dq = np.zeros_like(q)
        dq[j] = eps
        o, _ = arm.forward(q + dq)
        J[:, :, j] = (o - base) / eps
    return base, J


def link_point_jacobian(J_origins: np.ndarray, link: int, t: float, n_links: int) -> np.ndarray:
    """Jacobian of the point ``o_a + t (o_b - o_a)`` on link ``link`` (``a``, ``b`` its axis ends).

    Analytic arm: 7 origins, link ``i`` spans origins ``i -> i+1``.  MuJoCo arm: 8 origins
    (base, 6 link bodies, tool site), link ``i`` spans ``i+1 -> i+2``.  Both are the generic
    ``i + off -> i + off + 1`` with ``off = n_origins - n_links - 1`` (0 analytic, 1 MuJoCo).
    """
    off = J_origins.shape[0] - n_links - 1
    a = J_origins[link + off]
    b = J_origins[link + off + 1]
    return (1.0 - t) * a + t * b


# ---------------------------------------------------------------------------- joint-space APF step
def apf_joint_step(env, k_att: float, k_rep: float, rho0: float, d_switch: float, dq_max: float) -> tuple[np.ndarray, dict]:
    """APF joint increment (radians) at the environment's current configuration.

    Returns ``(dq (n,), details)``; ``details`` carries the forces for diagnostics.
    """
    q = np.asarray(env.q, dtype=np.float64)
    arm = env.arm
    n = q.shape[0]
    target = env.scene.target
    ee = env.ee

    f_att = attractive_force(ee, target, k_att, d_switch)
    J_ee = arm.position_jacobian(q)
    dq = J_ee.T @ f_att

    origins, clearance, direction, _ = env._geometry()
    f_rep_total = np.zeros(3)
    if clearance.size:
        n_links = clearance.shape[0]
        J_or = None
        nearest = np.argmin(clearance, axis=1)
        for i in env.monitored_links:
            j = int(nearest[i])
            rho = float(clearance[i, j])
            if rho >= rho0:
                continue
            f = repulsive_force(rho, direction[i, j], k_rep, rho0)
            if not np.any(f):
                continue
            if J_or is None:
                _, J_or = origins_jacobian(arm, q)
            t = _closest_point_fraction(env, origins, i, j)
            J_i = link_point_jacobian(J_or, i, t, n_links)
            dq = dq + J_i.T @ f
            f_rep_total += f

    scale = 1.0
    m = float(np.max(np.abs(dq))) if n else 0.0
    if m > dq_max and m > _EPS:
        scale = dq_max / m
        dq = dq * scale
    return dq, {"f_att": f_att, "f_rep": f_rep_total, "scale": scale}


def _closest_point_fraction(env, origins: np.ndarray, link: int, obstacle: int) -> float:
    """Fractional position of the closest point along link ``link``'s axis.

    Analytic backend: recomputed exactly from the capsule axes.  Other backends (MuJoCo
    collision meshes/capsules): fall back to the distal frame origin (t = 1).
    """
    scene = env.scene
    if getattr(env.backend, "name", "") != "analytic" or scene is None:
        return 1.0
    p1, p2 = origins[link], origins[link + 1]
    t, _, _, _, _ = closest_points_between_segments(p1, p2, scene.obs_q1[obstacle], scene.obs_q2[obstacle])
    return float(np.clip(t, 0.0, 1.0))


# ---------------------------------------------------------------------------- potential from the observation
def obs_clearances(obs: np.ndarray, layout: dict, margin: float) -> np.ndarray:
    """Surface clearance per monitored link from the ``sl`` vectors.

    ``margin`` = link radius + obstacle radius for the analytic backend (its vectors join
    the capsule *axes*), 0 for MuJoCo (``mj_geomDistance`` already joins the surfaces).
    A zero vector means "no obstacle" -> infinite clearance.
    """
    a, b = layout["sl"]
    v = np.asarray(obs[a:b], dtype=np.float64).reshape(-1, 3)
    d = np.linalg.norm(v, axis=1)
    rho = d - margin
    rho[d < _EPS] = np.inf
    return rho


def potential(obs: np.ndarray, layout: dict, k_phi: float, k_phi_rep: float, rho0: float, distance_scale: float, margin: float) -> float:
    """Phi(s) = -k_phi * |EE - T| * distance_scale - k_phi_rep * sum_i max(0, 1/rho_i - 1/rho0)."""
    se = np.asarray(obs[slice(*layout["se"])], dtype=np.float64)
    st = np.asarray(obs[slice(*layout["st"])], dtype=np.float64)
    phi = -k_phi * float(np.linalg.norm(se - st)) * distance_scale
    if k_phi_rep != 0.0:
        phi -= k_phi_rep * repulsive_potential(obs_clearances(obs, layout, margin), rho0)
    return phi
