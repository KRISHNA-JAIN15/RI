"""Topological data analysis of scenes and end-effector paths (research direction C5).

Two tools:

* :func:`scene_complexity` - a scalar difficulty ``tau`` of a scene from the persistent
  homology (Vietoris-Rips, ``ripser``) of a uniform point sample of the *free* reachable
  workspace.  Obstacles punch holes into the free space; the H1 (loops) and H2 (voids)
  features that live long in the filtration are the holes the end effector has to go
  around.  ``tau`` is the total persistence (sum of finite lifetimes) of H1 plus twice
  that of H2.  Results are cached per scene geometry so the curriculum plugin and the
  stratification script never compute the same scene twice.

* :func:`path_homotopy_signature` - a cheap, discrete signature of *which way* a path
  goes around every obstacle capsule.  See the docstring for what it does and does not
  capture; it is an approximation of the homotopy class, not the class itself.

The scene complexity is *not* meant as a state feature (see docs/research_directions.md):
it is an offline, per-scene quantity that drives a decision (curriculum) and an
evaluation axis (stratified results).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

import numpy as np

_CACHE_MAX = 4096
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()


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
    """Clearance of ``points`` ``(N, 3)`` to ``M`` capsules; returns ``(N,)`` = min over capsules.

    Vectorised version of :func:`arm_rl.geometry.point_segment_distance` (same clamped
    projection), minus the capsule radius.  An empty scene returns ``+inf``.
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


def sample_free_space(scene, shoulder: np.ndarray, reach: float, n_samples: int, rng: np.random.Generator,
                      r_min: float = 0.2, margin: float = 0.0) -> tuple[np.ndarray, int]:
    """Uniform points in the reachable shell ``r_min <= |p - shoulder| <= reach``, ``z > 0``,
    outside every obstacle capsule (clearance ``> margin``).  Returns ``(points, n_drawn)``.

    The draw is uniform in volume (radius sampled with ``r^3`` uniform), and the ``z > 0``
    half is enforced by mirroring, which keeps the distribution uniform on the half shell.
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    u = rng.normal(size=(n_samples, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = np.cbrt(rng.uniform(r_min**3, reach**3, size=n_samples))
    p = shoulder + r[:, None] * u
    p[:, 2] = shoulder[2] + np.abs(p[:, 2] - shoulder[2])  # mirror into z >= shoulder_z (> 0)
    keep = p[:, 2] > 0.0
    clearance = points_segments_clearance(p, scene.obs_q1, scene.obs_q2, scene.radii)
    keep &= clearance > margin
    return p[keep], n_samples


# ============================================================================ complexity
def scene_complexity(scene, arm_shoulder, reach: float, n_samples: int = 600, seed: int = 0, max_dim: int = 1,
                     thresh: float | None = None, eps: float | None = None, margin: float = 0.0,
                     use_cache: bool = True) -> dict:
    """Persistent-homology difficulty of a scene.

    Args:
        scene: :class:`arm_rl.scene.Scene` (current geometry, i.e. at its current ``t``).
        arm_shoulder: ``(3,)`` shoulder position; reach: max shoulder-EE distance (m).
        n_samples: points drawn in the reachable half shell before rejection.
        seed: sample seed (fixed so that ``tau`` is a deterministic function of the scene).
        max_dim: 1 computes H0/H1; 2 adds H2 (voids; ~3-5x slower).
        thresh: Vietoris-Rips distance cut-off (default ``0.5 * reach``) - features that
            are still alive at the cut-off are *not* counted (their lifetime is infinite).
        eps: minimum lifetime for a feature to count in ``n_h1`` (default ``0.05 * reach``).
        margin: extra clearance subtracted from the capsules (0 = geometric obstacle only).

    Returns dict with ``tau`` (sum of finite H1 lifetimes + 2 x sum of finite H2
    lifetimes), ``n_h1`` (H1 features with lifetime > eps), ``h1_lifetimes`` (sorted,
    descending), ``h2_lifetimes``, ``free_fraction`` (accepted / drawn), ``n_points``,
    ``time_s`` and ``cached``.

    Caveats.  ``tau`` depends on the sample density: noise loops of lifetime ~ the point
    spacing always exist, so compare scenes computed with the same ``n_samples``.  The
    sample ignores the arm's own geometry (the free *workspace*, not the free
    configuration space), so a scene can have a high ``tau`` and still be easy if the
    loops are far from the start-target chord.
    """
    from ripser import ripser  # local import: keeps arm_rl importable without ripser

    key = (scene_geometry_hash(scene), int(n_samples), int(seed), int(max_dim), thresh, eps, float(margin))
    if use_cache and key in _CACHE:
        out = dict(_CACHE[key])
        out["cached"] = True
        _CACHE.move_to_end(key)
        return out

    t0 = time.perf_counter()
    reach = float(reach)
    thresh_v = 0.5 * reach if thresh is None else float(thresh)
    eps_v = 0.05 * reach if eps is None else float(eps)
    rng = np.random.default_rng(seed)
    pts, n_drawn = sample_free_space(scene, arm_shoulder, reach, int(n_samples), rng, margin=margin)

    h1, h2 = np.zeros(0), np.zeros(0)
    if len(pts) >= 4:
        dgms = ripser(pts, maxdim=int(max_dim), thresh=thresh_v)["dgms"]
        if len(dgms) > 1 and len(dgms[1]):
            life = dgms[1][:, 1] - dgms[1][:, 0]
            h1 = np.sort(life[np.isfinite(life)])[::-1]
        if max_dim >= 2 and len(dgms) > 2 and len(dgms[2]):
            life = dgms[2][:, 1] - dgms[2][:, 0]
            h2 = np.sort(life[np.isfinite(life)])[::-1]
    tau = float(h1.sum() + 2.0 * h2.sum())
    out = {
        "tau": tau,
        "n_h1": int(np.sum(h1 > eps_v)),
        "n_h2": int(np.sum(h2 > eps_v)),
        "h1_lifetimes": [float(x) for x in h1],
        "h2_lifetimes": [float(x) for x in h2],
        "free_fraction": float(len(pts) / max(1, n_drawn)),
        "n_points": int(len(pts)),
        "thresh": thresh_v,
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
