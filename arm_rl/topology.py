"""Topological data analysis of scenes and end-effector paths (research direction C5).

Two tools:

* :func:`scene_complexity` - a scalar difficulty ``tau`` of a scene from the persistent
  homology of the *clearance field* of the free workspace around the target, computed
  with ``ripser``.  ``tau`` is a total persistence (sum of bar lengths, in metres of
  clearance) over H0 (pockets / narrow passages), H1 (loops: obstacles that must be gone
  around) and optionally H2 (voids).  Results are cached per scene geometry, so the
  curriculum plugin and the stratification script never compute the same scene twice.

* :func:`path_homotopy_signature` - a cheap, discrete signature of *which way* a path
  goes around every obstacle capsule.  See its docstring for what it does and does not
  capture; it is an approximation of the homotopy class, not the class itself.

Why a clearance filtration and not a plain Vietoris-Rips complex on free-space samples.
A Rips complex on a few hundred uniform samples of the free workspace only sees holes
that are wider than the sample spacing (~5-10 cm here).  Capsule obstacles of radius
4 cm are invisible at that scale: the persistence diagram is pure sampling noise and its
total persistence grows with the number of accepted points, i.e. it *decreases* with
clutter.  We therefore sample the region of interest on a regular grid, evaluate the
signed clearance ``c(p)`` (distance to the nearest capsule surface, negative inside) at
every grid point, and build the lower-star filtration of ``f = -c`` on the clique complex
of the 26-neighbourhood grid graph (a sparse distance matrix ``d(i, j) = max(f_i, f_j)``
with vertex births ``f_i`` on the diagonal; ``ripser`` accepts exactly that).  Sub-level
sets of ``f`` are super-level sets of the clearance: "the part of the workspace at least
``c`` metres away from every obstacle".  Bars of this filtration are in metres of
clearance and have a direct meaning:

* **H0** - a bar ``[c_birth, c_death]`` is a pocket of free space that exists at
  clearance ``c_birth`` and merges with a bigger pocket through a passage of half-width
  ``c_death``.  Narrow passages = long bars.
* **H0, start-target pair** - with ``start`` given, the level ``c*`` at which the start's
  and the target's pockets merge is the *bottleneck clearance* (maximin clearance over all
  paths through the region).  Its bar ``[c_cap, c*]`` is added to ``tau``: the deeper the
  target sits below open space, the harder the scene.
* **H1** - a loop that cannot be contracted without crossing an obstacle: an obstacle
  that traverses the region (the trunk), or a handle formed by touching obstacles (a shelf
  frame, a ring of posts).  Born when the free space first surrounds the obstacle, dies
  when the obstacle is "filled in" at negative clearance, so its length ~ obstacle
  thickness + surrounding clearance.
* **H2** - a void (an enclosed cavity).  Weighted 2x.

Clearance is capped at ``c_cap`` so that open space is a single flat plateau and an
obstacle-free scene has ``tau = 0`` exactly.  ``tau`` is in metres; values are comparable
across scenes computed with the same ``pad``/``n_samples``.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

import numpy as np

_CACHE_MAX = 4096
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()

# 13 of the 26 grid neighbour offsets (the other 13 are their negatives)
_NEIGHBOURS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]


# ============================================================================ helpers
def scene_geometry_hash(scene, decimals: int = 4) -> str:
    """Hash of the *current* obstacle axes, radii and target (rounded to ``decimals``)."""
    h = hashlib.sha1()
    for arr in (scene.obs_q1, scene.obs_q2, scene.radii, scene.target):
        a = np.ascontiguousarray(np.round(np.asarray(arr, dtype=np.float64), decimals))
        h.update(a.tobytes())
        h.update(str(a.shape).encode())
    return h.hexdigest()


def points_segments_clearance(points: np.ndarray, q1: np.ndarray, q2: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Signed clearance of ``points`` ``(N, 3)`` to ``M`` capsules; ``(N,)`` = min over capsules.

    Vectorised version of :func:`arm_rl.geometry.point_segment_distance` (same clamped
    projection), minus the capsule radius: negative inside a capsule.  An empty scene
    returns ``+inf``.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(q1) == 0:
        return np.full(len(points), np.inf)
    d = q2 - q1  # (M, 3)
    e = np.sum(d * d, axis=-1)  # (M,)
    e_safe = np.where(e < 1e-12, 1.0, e)
    rel = points[:, None, :] - q1[None, :, :]  # (N, M, 3)
    s = np.sum(rel * d[None, :, :], axis=-1) / e_safe[None, :]
    s = np.clip(np.where(e[None, :] < 1e-12, 0.0, s), 0.0, 1.0)
    closest = q1[None, :, :] + s[..., None] * d[None, :, :]
    dist = np.linalg.norm(closest - points[:, None, :], axis=-1) - np.asarray(radii)[None, :]
    return dist.min(axis=1)


def _region_grid(anchors: np.ndarray, pad: float, n_samples: int, rng: np.random.Generator):
    """Regular grid over the padded bounding box of ``anchors`` with ~``n_samples`` points.

    The grid origin is jittered by a random fraction of the spacing (seeded) so that no
    scene is systematically aliased.  Returns ``(points, shape, spacing)``.
    """
    lo = anchors.min(axis=0) - pad
    hi = anchors.max(axis=0) + pad
    lo[2] = max(lo[2], 0.0)
    vol = float(np.prod(np.maximum(hi - lo, 1e-6)))
    h = float(np.clip(np.cbrt(vol / max(8, n_samples)), 0.012, 0.08))
    jitter = rng.uniform(0.0, h, size=3)
    axes = [np.arange(l + j, u + 1e-9, h) for l, u, j in zip(lo, hi, jitter)]
    axes = [a if len(a) >= 2 else np.array([l, u]) for a, l, u in zip(axes, lo, hi)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts, tuple(len(a) for a in axes), h


def _grid_edges(shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs ``(a, b)`` of all 26-neighbour edges of a grid of ``shape``."""
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    rows, cols = [], []
    for d in _NEIGHBOURS:
        sl_a = tuple(slice(max(0, -dd), shape[k] - max(0, dd)) for k, dd in enumerate(d))
        sl_b = tuple(slice(max(0, dd), shape[k] + min(0, dd)) for k, dd in enumerate(d))
        rows.append(idx[sl_a].ravel())
        cols.append(idx[sl_b].ravel())
    return np.concatenate(rows), np.concatenate(cols)


def _bottleneck_clearance(n: int, a: np.ndarray, b: np.ndarray, c: np.ndarray, i_start: int, i_target: int) -> float:
    """Maximin clearance between two vertices on the edge list ``(a, b)`` with edge
    clearance ``min(c_a, c_b)`` (Kruskal on edges sorted by decreasing clearance)."""
    w = np.minimum(c[a], c[b])
    order = np.argsort(-w, kind="stable")
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in order:
        ra, rb = find(a[k]), find(b[k])
        if ra != rb:
            parent[ra] = rb
            if find(i_start) == find(i_target):
                return float(w[k])
    return float("-inf")


# ============================================================================ complexity
def scene_complexity(scene, arm_shoulder, reach: float, n_samples: int = 2000, seed: int = 0, max_dim: int = 1,
                     start=None, pad: float = 0.15, c_cap: float | None = None, margin: float = 0.0,
                     eps: float | None = None, r_min: float = 0.2, use_cache: bool = True) -> dict:
    """Persistent-homology difficulty of a scene (see the module docstring).

    Args:
        scene: :class:`arm_rl.scene.Scene` (current geometry, i.e. at its current ``t``).
        arm_shoulder: ``(3,)`` shoulder position; ``reach``: max shoulder-EE distance (m).
            Grid points outside the reachable shell ``r_min <= |p - shoulder| <= reach`` or
            with ``z <= 0`` are discarded.
        n_samples: approximate number of grid points in the region of interest (sets the
            grid spacing; clamped to 1.2-8 cm).
        seed: jitter seed of the grid origin (``tau`` is deterministic given the seed).
        max_dim: 1 computes H0/H1; 2 adds H2 (voids; several times slower).
        start: optional ``(3,)`` start EE position.  Extends the region of interest to the
            start-target corridor and adds the bottleneck bar ``c_cap - c*`` to ``tau``.
        pad: padding (m) of the region of interest around the target (and start).
        c_cap: clearance cap (default ``pad``); bars are measured up to this level.
        margin: extra radius added to every capsule (e.g. the link radius).
        eps: minimum bar length to count in ``n_h1`` (default ``0.1 * c_cap``).

    Returns a dict: ``tau`` (metres: sum of finite H0 bars + bottleneck bar + sum of H1
    bars + 2 x sum of H2 bars), ``tau_h0``, ``tau_h1``, ``tau_h2``, ``bottleneck``
    (``c*`` maximin clearance start->target, or ``None``), ``n_h1`` (H1 bars longer than
    ``eps``), ``h0_lifetimes``, ``h1_lifetimes``, ``h2_lifetimes`` (descending),
    ``free_fraction`` (grid points with positive clearance / grid points), ``n_points``,
    ``spacing``, ``time_s``, ``cached``.

    Caveats.  The grid resolution bounds what can be seen: a bar shorter than the spacing
    is unreliable.  The region is the free *workspace* (a point robot), not the free
    configuration space of the six-link arm; ``margin`` can inflate obstacles by a link
    radius but the arm's kinematic reachability is ignored.  Values depend on ``pad``,
    ``c_cap`` and ``n_samples``: compare scenes computed with the same settings.
    """
    import scipy.sparse as sp
    from ripser import ripser  # local import: keeps arm_rl importable without ripser

    start_arr = None if start is None else np.asarray(start, dtype=np.float64).reshape(3)
    key = (scene_geometry_hash(scene), int(n_samples), int(seed), int(max_dim),
           None if start_arr is None else tuple(np.round(start_arr, 4)), float(pad), c_cap, float(margin), eps, float(r_min))
    if use_cache and key in _CACHE:
        out = dict(_CACHE[key])
        out["cached"] = True
        _CACHE.move_to_end(key)
        return out

    t0 = time.perf_counter()
    shoulder = np.asarray(arm_shoulder, dtype=np.float64)
    reach = float(reach)
    cap = float(pad if c_cap is None else c_cap)
    eps_v = 0.1 * cap if eps is None else float(eps)
    rng = np.random.default_rng(seed)

    anchors = np.asarray([scene.target] if start_arr is None else [scene.target, start_arr], dtype=np.float64)
    pts, shape, h = _region_grid(anchors, pad, int(n_samples), rng)
    radial = np.linalg.norm(pts - shoulder, axis=1)
    keep = (pts[:, 2] > 0.0) & (radial >= r_min) & (radial <= reach)
    clearance = points_segments_clearance(pts, scene.obs_q1, scene.obs_q2, scene.radii) - float(margin)
    clearance = np.minimum(clearance, cap)

    a, b = _grid_edges(shape)
    m = keep[a] & keep[b]
    a, b = a[m], b[m]
    new_id = -np.ones(len(pts), dtype=np.int64)
    new_id[keep] = np.arange(int(keep.sum()))
    a, b = new_id[a], new_id[b]
    c = clearance[keep]
    n = int(keep.sum())

    h0 = h1 = h2 = np.zeros(0)
    bottleneck = None
    if n >= 2 and len(a):
        f = -c
        f = f - f.min() + 1e-9  # ripser wants non-negative filtration values
        w = np.maximum(f[a], f[b])
        D = sp.coo_matrix((np.concatenate([w, w, f]), (np.concatenate([a, b, np.arange(n)]), np.concatenate([b, a, np.arange(n)]))), shape=(n, n))
        dgms = ripser(D, distance_matrix=True, maxdim=int(max_dim))["dgms"]

        def finite_lifetimes(dg):
            if len(dg) == 0:
                return np.zeros(0)
            life = dg[:, 1] - dg[:, 0]
            return np.sort(life[np.isfinite(life)])[::-1]

        h0 = finite_lifetimes(dgms[0])
        h1 = finite_lifetimes(dgms[1]) if len(dgms) > 1 else h1
        h2 = finite_lifetimes(dgms[2]) if max_dim >= 2 and len(dgms) > 2 else h2
        if start_arr is not None:
            kept_pts = pts[keep]
            i_t = int(np.argmin(np.linalg.norm(kept_pts - scene.target, axis=1)))
            i_s = int(np.argmin(np.linalg.norm(kept_pts - start_arr, axis=1)))
            bottleneck = _bottleneck_clearance(n, a, b, c, i_s, i_t)
            if not np.isfinite(bottleneck):
                bottleneck = float(c.min())
    tau_h0 = float(h0.sum())
    tau_h1 = float(h1.sum())
    tau_h2 = float(h2.sum())
    tau_bn = 0.0 if bottleneck is None else float(max(0.0, cap - bottleneck))
    out = {
        "tau": tau_h0 + tau_bn + tau_h1 + 2.0 * tau_h2,
        "tau_h0": tau_h0,
        "tau_bottleneck": tau_bn,
        "tau_h1": tau_h1,
        "tau_h2": tau_h2,
        "bottleneck": bottleneck,
        "n_h0": int(np.sum(h0 > eps_v)),
        "n_h1": int(np.sum(h1 > eps_v)),
        "n_h2": int(np.sum(h2 > eps_v)),
        "h0_lifetimes": [float(x) for x in h0],
        "h1_lifetimes": [float(x) for x in h1],
        "h2_lifetimes": [float(x) for x in h2],
        "free_fraction": float(np.mean(c > 0.0)) if n else 0.0,
        "n_points": n,
        "spacing": h,
        "c_cap": cap,
        "eps": eps_v,
        "time_s": time.perf_counter() - t0,
        "cached": False,
    }
    if use_cache:
        _CACHE[key] = dict(out)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return out


def clear_cache() -> None:
    _CACHE.clear()


def cache_size() -> int:
    return len(_CACHE)


# ============================================================================ homotopy signature
def _swept_angle(xy: np.ndarray) -> float:
    """Total signed angle swept by the 2-D polyline ``xy`` ``(T, 2)`` about the origin."""
    if len(xy) < 2:
        return 0.0
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi  # wrap each step into (-pi, pi]
    return float(d.sum())


def path_homotopy_signature(ee_path: np.ndarray, scene, min_radial: float = 1e-3) -> tuple:
    """Discrete "which side" signature of an EE path with respect to each obstacle capsule.

    For obstacle ``i`` with axis ``q1 -> q2`` (a sphere, ``q1 == q2``, uses the vertical
    axis through its centre) the path and the straight chord start -> end are projected on
    the plane perpendicular to the axis, with the axis as origin.  The winding number of
    the closed loop *path followed by the reversed chord* is
    ``w_i = (swept_angle(path) - swept_angle(chord)) / 2 pi``, an integer when neither
    curve passes through the axis.  The signature entry is ``sign(round(w_i))`` in
    ``{-1, 0, +1}``: ``+1`` = the path goes around the axis counter-clockwise relative to
    the chord (seen from ``q2`` looking down the axis), ``-1`` = clockwise, ``0`` = the
    path and the chord are on the same side.  The tuple has one entry per obstacle, in
    scene order.

    What this captures and what it does not.
    * It is exact for homotopy classes in the plane with *point* obstacles, i.e. it treats
      every capsule as an infinite line (Bhattacharya et al. 2012, "H-signature").  Two
      paths with different signatures are in different homotopy classes of the complement
      of those lines.
    * In 3-D a finite capsule is not an infinite line: a path that passes *over the tip*
      of a branch is homotopic (in the real free space) to one that passes on either side,
      yet it receives a definite side label here.  So equal signatures do not prove the
      same class, and different signatures may be homotopic in 3-D when the detour goes
      over an end of the capsule.  It is a coarse, reproducible descriptor of the route
      choice, adequate for "does DDPG take the same route as RRT" style statistics.
    * The reference is the straight chord, so ``0`` means "same side as the chord", not
      "did not interact with the obstacle".  Obstacles far from the path always get ``0``.
    * Partial windings (e.g. the path ends at the target without closing) are rounded;
      a path that stops on the far side of an obstacle counts as a ``+-1`` if the loop
      closed by the chord encircles the axis.
    * Points closer than ``min_radial`` to the axis (projected) are dropped to avoid an
      undefined angle; a path that pierces an obstacle is not meaningfully classified.
    """
    path = np.asarray(ee_path, dtype=np.float64).reshape(-1, 3)
    if len(path) < 2 or scene.n_obstacles == 0:
        return tuple([0] * scene.n_obstacles)
    chord = np.stack([path[0], path[-1]])
    sig = []
    for i in range(scene.n_obstacles):
        q1, q2 = scene.obs_q1[i], scene.obs_q2[i]
        axis = q2 - q1
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        # orthonormal basis (e1, e2) of the plane perpendicular to the axis
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(axis, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

        def proj(p):
            rel = p - q1[None, :]
            xy = np.stack([rel @ e1, rel @ e2], axis=1)
            return xy[np.linalg.norm(xy, axis=1) > min_radial]

        w = (_swept_angle(proj(path)) - _swept_angle(proj(chord))) / (2.0 * np.pi)
        sig.append(int(np.sign(np.round(w))))
    return tuple(sig)


def signature_stats(signatures: list[tuple]) -> dict:
    """Distribution summary of a list of signatures: distinct classes, most common share."""
    from collections import Counter

    if not signatures:
        return {"n": 0, "n_classes": 0, "top_share": 0.0, "top_class": None, "nontrivial_fraction": 0.0}
    c = Counter(tuple(int(x) for x in s) for s in signatures)
    top, cnt = c.most_common(1)[0]
    nontrivial = sum(1 for s in signatures if any(int(x) != 0 for x in s))
    return {"n": len(signatures), "n_classes": len(c), "top_share": cnt / len(signatures), "top_class": list(top),
            "nontrivial_fraction": nontrivial / len(signatures)}
