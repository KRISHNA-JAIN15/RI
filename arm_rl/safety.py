"""Control-barrier-function (CBF) safety shield for the arm (research contribution C1).

For every monitored link ``i`` and each of its ``k_nearest`` closest obstacles ``j`` we keep
the barrier

    h_ij(q) = clearance_ij(q) - d_safe >= 0        (clearance is surface-to-surface)

forward invariant under the joint increment ``dq`` chosen per control step.  With
``n_ij`` the unit vector from the closest point ``P'`` on the link towards the closest
point ``Q'`` on the obstacle, ``J_ij = dP'/dq`` (position Jacobian of the link at ``P'``)
and ``v_j`` the obstacle velocity, a first-order expansion gives

    h_ij(q + dq) ~= h_ij(q) - n_ij^T J_ij dq + n_ij^T v_j dt

and the discrete-time CBF condition ``h(q + dq) >= (1 - alpha) h(q)`` becomes the linear
constraint (motion *towards* the obstacle is limited, the velocity term sits on the
right-hand side)

    n_ij^T J_ij dq <= alpha * h_ij + n_ij^T v_j dt .

The shield solves  min ||dq - dq_RL||^2  subject to these constraints and the box
``|dq| <= max_delta`` (intersected with the joint limits).  If no feasible increment
exists (the arm already sits inside ``d_safe`` or an obstacle closes in faster than the
arm can retreat) the already-violated constraints are dropped and the arm is steered along
the direction that increases those barriers fastest, still respecting the remaining
constraints (least-violating increment).

Backends
--------
* analytic (``arm_rl.env.AnalyticBackend``): links are the segments between consecutive
  frame origins.  ``P' = o_i + t (o_{i+1} - o_i)`` with the fractional position ``t`` of the
  closest point held fixed, so ``J = (1 - t) J_{o_i} + t J_{o_{i+1}}`` where the origin
  Jacobians come from finite differences through ``arm.forward``.
* mujoco (``arm_rl.mujoco_sim.MujocoBackend``): the closest point is found on the link's
  collision capsules (4th return value of ``backend.geometry``) and its Jacobian is taken
  from ``mujoco.mj_jac`` on the live ``MjData`` for the link body.

Two QP solvers are provided: ``qpsolvers``/OSQP and a dependency-free Hildreth dual
coordinate-ascent solver (``hildreth_qp``); both return the same minimiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm_rl.geometry import closest_points_between_segments

try:  # optional
    import qpsolvers as _qps

    _HAVE_QPSOLVERS = "osqp" in _qps.available_solvers
except Exception:  # pragma: no cover - import guard
    _qps = None
    _HAVE_QPSOLVERS = False

_EPS = 1e-12


# ============================================================================ QP solvers
def hildreth_qp(x0: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int = 2000, tol: float = 1e-9):
    """Projection of ``x0`` onto ``{x : A x <= b}`` by Hildreth's dual coordinate ascent.

    The dual of ``min 0.5||x - x0||^2 s.t. A x <= b`` is ``min_{lam >= 0} 0.5 lam^T H lam - c^T lam``
    with ``H = A A^T`` and ``c = A x0 - b``; the primal is recovered as ``x = x0 - A^T lam``.
    Returns ``(x, converged, max_violation)``.  An infeasible problem makes the multipliers
    diverge; the caller detects it from ``max_violation``.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = A.shape[0]
    if m == 0:
        return x0.copy(), True, 0.0
    H = A @ A.T
    c = A @ x0 - b
    diag = np.diag(H).copy()
    active = diag > _EPS  # zero rows constrain nothing (feasible iff b >= 0)
    lam = np.zeros(m)
    converged = False
    idx = np.flatnonzero(active)
    Hrows = [H[i] for i in idx]
    for _ in range(int(max_iter)):
        delta = 0.0
        for k, i in enumerate(idx):
            new = lam[i] - (Hrows[k] @ lam - c[i]) / diag[i]
            if new < 0.0:
                new = 0.0
            d = abs(new - lam[i])
            if d > delta:
                delta = d
            lam[i] = new
        if delta < tol:
            converged = True
            break
    x = x0 - A.T @ lam
    viol = float(np.max(A @ x - b)) if m else 0.0
    return x, converged, max(viol, 0.0)


def _osqp_qp(x0: np.ndarray, G: np.ndarray, h: np.ndarray, lb: np.ndarray, ub: np.ndarray, max_iter: int):
    import scipy.sparse as sp

    n = x0.shape[0]
    P = sp.identity(n, format="csc")
    q = -np.asarray(x0, dtype=np.float64)
    Gm = sp.csc_matrix(G) if G.shape[0] else None
    hv = h if G.shape[0] else None
    x = _qps.solve_qp(P, q, Gm, hv, lb=lb, ub=ub, solver="osqp", eps_abs=1e-9, eps_rel=1e-9,
                      max_iter=int(max(max_iter, 4000)), polish=False, verbose=False)
    return x


def solve_box_qp(x0: np.ndarray, G: np.ndarray, h: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                 solver: str = "auto", max_iter: int = 2000, tol: float = 1e-6):
    """``min 0.5||x - x0||^2  s.t.  G x <= h,  lb <= x <= ub``.

    Returns ``(x, feasible)``; ``x`` is ``None`` when the problem is infeasible.
    ``solver``: ``"auto"`` (qpsolvers/OSQP if installed, else numpy), ``"osqp"``/``"qpsolvers"``, ``"numpy"``.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64).reshape(-1, x0.shape[0])
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    if np.any(lb > ub + 1e-12):
        return None, False
    use_osqp = solver in ("osqp", "qpsolvers") or (solver == "auto" and _HAVE_QPSOLVERS)
    if use_osqp and not _HAVE_QPSOLVERS:
        raise RuntimeError("qpsolvers/osqp not installed; use solver='numpy'")
    if use_osqp:
        x = _osqp_qp(x0, G, h, lb, ub, max_iter)
        if x is None:
            return None, False
        x = np.clip(x, lb, ub)  # OSQP satisfies bounds to solver tolerance only
        viol = float(np.max(G @ x - h)) if G.shape[0] else 0.0
        return x, viol <= tol
    n = x0.shape[0]
    A = np.vstack([G, np.eye(n), -np.eye(n)])
    b = np.concatenate([h, ub, -lb])
    x, _, viol = hildreth_qp(x0, A, b, max_iter=max_iter)
    if viol > tol:
        return None, False
    return np.clip(x, lb, ub), True


# ============================================================================ constraints
@dataclass
class BarrierConstraint:
    link: int
    obstacle: int           # obstacle index, -1 for the floor
    h: float                # barrier value clearance - d_safe
    normal: np.ndarray      # (3,) unit vector from the link towards the obstacle
    point: np.ndarray       # (3,) closest point P' on the link
    jac: np.ndarray         # (3, 6) dP'/dq
    obs_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def row(self) -> np.ndarray:
        return self.normal @ self.jac

    def rhs(self, alpha: float, dt: float) -> float:
        return alpha * self.h + float(self.normal @ self.obs_vel) * dt


class CBFShield:
    """Builds the barrier constraints at the current configuration and filters ``dq``."""

    def __init__(self, env, d_safe: float = 0.02, alpha: float = 0.5, k_nearest: int = 3, use_velocity: bool = True,
                 solver: str = "auto", max_iter: int = 2000, d_active: float | None = 0.25, floor: bool | None = None,
                 margin: float = 0.0, fd_eps: float = 1e-6):
        self.env = env
        self.d_safe = float(d_safe) + float(margin)  # margin absorbs the linearisation error of one step
        self.alpha = float(alpha)
        self.k_nearest = int(k_nearest)
        self.use_velocity = bool(use_velocity)
        self.solver = str(solver)
        self.max_iter = int(max_iter)
        self.d_active = None if d_active is None else float(d_active)
        self.floor = bool(env.floor_collision) if floor is None else bool(floor)
        self.fd_eps = float(fd_eps)
        self.is_mujoco = getattr(env.backend, "name", "") == "mujoco"
        self.n = int(env.arm.n_joints)
        self.last: dict = {}

    # ------------------------------------------------------------------ Jacobians
    def _origin_jacobians(self, q: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobians (n_origins, 3, n) of every frame origin w.r.t. q."""
        arm = self.env.arm
        base, _ = arm.forward(q)
        J = np.zeros((base.shape[0], 3, self.n))
        for k in range(self.n):
            qk = q.copy()
            qk[k] += self.fd_eps
            o, _ = arm.forward(qk)
            J[:, :, k] = (o - base) / self.fd_eps
        return J

    def _mj_point_jacobian(self, point: np.ndarray, link: int) -> np.ndarray:
        import mujoco

        be = self.env.backend
        jacp = np.zeros((3, be.model.nv))
        mujoco.mj_jac(be.model, be.data, jacp, None, np.asarray(point, dtype=np.float64), int(be.link_body_ids[link]))
        return jacp[:, : self.n].copy()

    # ------------------------------------------------------------------ closest points
    def _closest_on_link(self, link: int, j: int, origins: np.ndarray, capsules, scene):
        """Closest point P' on link ``link`` to obstacle ``j`` and (analytic) its fraction t."""
        q1, q2 = scene.obs_q1[j], scene.obs_q2[j]
        if not self.is_mujoco or not capsules:
            t, _, pc, _, _ = closest_points_between_segments(origins[link], origins[link + 1], q1, q2)
            return pc, float(t)
        best, bp = np.inf, None
        r_obs = scene.radii[j]
        for cap in capsules:
            if cap["link"] != link:
                continue
            _, _, pc, _, dist = closest_points_between_segments(np.asarray(cap["a"]), np.asarray(cap["b"]), q1, q2)
            d = float(dist) - cap["r"] - r_obs
            if d < best:
                best, bp = d, pc
        if bp is None:  # link without collision geometry: fall back to the frame segment
            t, _, pc, _, _ = closest_points_between_segments(origins[link], origins[link + 1], q1, q2)
            return pc, float(t)
        return bp, float("nan")

    # ------------------------------------------------------------------ constraint set
    def constraints(self, q: np.ndarray | None = None) -> list[BarrierConstraint]:
        env = self.env
        q = env.q if q is None else np.asarray(q, dtype=np.float64)
        same = np.array_equal(q, env.q)
        geom = env._geometry() if same else env.backend.geometry(q)
        origins, clearance, direction, capsules = geom
        scene = env.scene
        M = clearance.shape[1] if clearance.ndim == 2 else 0
        vel = scene.obs_vel if (self.use_velocity and scene is not None and scene.obs_vel is not None) else None
        Jo = None if self.is_mujoco else self._origin_jacobians(q)
        out: list[BarrierConstraint] = []
        for i in env.monitored_links:
            if M:
                order = np.argsort(clearance[i])[: self.k_nearest]
                for j in order:
                    c = float(clearance[i, j])
                    if self.d_active is not None and c > self.d_active:
                        break
                    d = direction[i, j]
                    nn = float(np.linalg.norm(d))
                    if nn < 1e-9 or not np.isfinite(c):
                        continue  # axes intersect: no defined normal (deep collision)
                    normal = d / nn
                    point, t = self._closest_on_link(i, int(j), origins, capsules, scene)
                    if self.is_mujoco:
                        J = self._mj_point_jacobian(point, i)
                    else:
                        J = (1.0 - t) * Jo[i] + t * Jo[i + 1]
                    v = vel[j] if vel is not None else np.zeros(3)
                    out.append(BarrierConstraint(i, int(j), c - self.d_safe, normal, point, J, np.asarray(v, dtype=np.float64)))
            if self.floor:
                out += self._floor_constraints(i, origins, capsules, Jo)
        return out

    def _floor_constraints(self, i: int, origins, capsules, Jo) -> list[BarrierConstraint]:
        down = np.array([0.0, 0.0, -1.0])
        out = []
        if self.is_mujoco and capsules:
            for cap in capsules:
                if cap["link"] != i:
                    continue
                for key in ("a", "b"):
                    p = np.asarray(cap[key], dtype=np.float64)
                    h = float(p[2]) - cap["r"] - self.d_safe
                    if self.d_active is not None and h > self.d_active:
                        continue
                    out.append(BarrierConstraint(i, -1, h, down, p, self._mj_point_jacobian(p, i)))
        else:
            p = origins[i + 1]
            h = float(p[2]) - self.d_safe  # the analytic backend flags a collision when an axis end point goes below z = 0
            if self.d_active is None or h <= self.d_active:
                out.append(BarrierConstraint(i, -1, h, down, p, Jo[i + 1]))
        return out

    # ------------------------------------------------------------------ filter
    def filter(self, dq_rl: np.ndarray, q: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Return the safe joint increment closest to ``dq_rl`` (radians) and a status dict."""
        env = self.env
        q = env.q if q is None else np.asarray(q, dtype=np.float64)
        delta = float(env.max_delta)
        lb = np.maximum(-delta, env.arm.joint_low - q)
        ub = np.minimum(delta, env.arm.joint_high - q)
        dq_rl = np.clip(np.asarray(dq_rl, dtype=np.float64), lb, ub)
        cons = self.constraints(q)
        info = {"n_constraints": len(cons), "feasible": True, "infeasible": False, "min_h": float("inf"), "qp": False}
        if not cons:
            self.last = info
            return dq_rl, info
        G = np.stack([c.row for c in cons])
        rhs = np.array([c.rhs(self.alpha, env.dt) for c in cons])
        info["min_h"] = float(min(c.h for c in cons))
        # Skip the QP when no constraint can bind: the worst-case decrease of each barrier
        # over the box is ||G_row||_1 * delta.
        if np.all(np.abs(G).sum(axis=1) * delta <= rhs):
            self.last = info
            return dq_rl, info
        info["qp"] = True
        x, ok = solve_box_qp(dq_rl, G, rhs, lb, ub, solver=self.solver, max_iter=self.max_iter)
        if ok:
            self.last = info
            return x, info
        # ---- infeasible: least-violating increment ------------------------------------
        info["feasible"] = False
        info["infeasible"] = True
        bad = rhs < 0.0  # constraints that cannot be satisfied by standing still
        if not np.any(bad):
            bad = np.ones(len(cons), dtype=bool)
        # steepest barrier increase for the violated set, scaled to the box
        g = -G[bad].sum(axis=0)
        scale = float(np.max(np.abs(g)))
        esc = np.clip(delta * g / scale, lb, ub) if scale > _EPS else np.zeros(self.n)
        keep = ~bad
        if np.any(keep):
            x, ok = solve_box_qp(esc, G[keep], rhs[keep], lb, ub, solver=self.solver, max_iter=self.max_iter)
            if ok:
                self.last = info
                return x, info
        self.last = info
        return esc, info
