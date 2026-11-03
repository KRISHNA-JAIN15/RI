import numpy as np
import pytest

from arm_rl.env import ArmPathPlanningEnv
from arm_rl.scene import Scene
from arm_rl.topology import cache_size, clear_cache, path_homotopy_signature, scene_complexity, signature_stats
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def env():
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], CFG["env"])


TARGET = np.array([0.45, 0.0, 0.30])


def empty_scene():
    return Scene(np.zeros((0, 3)), np.zeros((0, 3)), 0.04, TARGET, 0.03)


def ring_scene(n=8, r=0.12):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    q1 = np.stack([TARGET + [r * np.cos(a), r * np.sin(a), -0.25] for a in ang])
    q2 = q1 + [0.0, 0.0, 0.5]
    return Scene(q1, q2, 0.04, TARGET, 0.03)


def test_ring_harder_than_empty(env):
    clear_cache()
    start = env.arm.ee_position(env.arm.home)
    for st in (None, start):
        e = scene_complexity(empty_scene(), env.arm.shoulder, env.arm.reach, n_samples=1500, start=st)
        r = scene_complexity(ring_scene(), env.arm.shoulder, env.arm.reach, n_samples=1500, start=st)
        assert e["tau"] == 0.0 and e["n_h1"] == 0  # open space is one flat plateau
        assert r["tau"] > 0.1
        assert r["n_h1"] >= 1 and r["tau_h0"] > 0  # loops around the posts + a pocket inside the ring
        assert 0.0 < r["free_fraction"] < 1.0 and e["free_fraction"] == 1.0
    assert r["bottleneck"] < 0.05  # the target sits inside the ring: low bottleneck clearance
    assert e["bottleneck"] == pytest.approx(0.15)


def test_tree_scenes_vary_and_are_fast(env):
    rng = np.random.default_rng(0)
    start = env.arm.ee_position(env.arm.home)
    taus = []
    for _ in range(10):
        c = scene_complexity(env.scene_gen.sample(rng), env.arm.shoulder, env.arm.reach, n_samples=1500, start=start)
        assert c["time_s"] < 1.0
        taus.append(c["tau"])
    assert np.std(taus) > 0.0 and min(taus) > 0.0


def test_cache(env):
    clear_cache()
    s = ring_scene()
    a = scene_complexity(s, env.arm.shoulder, env.arm.reach, n_samples=800)
    assert not a["cached"] and cache_size() == 1
    b = scene_complexity(s.copy(), env.arm.shoulder, env.arm.reach, n_samples=800)
    assert b["cached"] and b["tau"] == a["tau"] and cache_size() == 1
    # a different geometry is a different key
    s2 = ring_scene(r=0.14)
    c = scene_complexity(s2, env.arm.shoulder, env.arm.reach, n_samples=800)
    assert not c["cached"] and cache_size() == 2
    # different settings are a different key too
    d = scene_complexity(s, env.arm.shoulder, env.arm.reach, n_samples=801)
    assert not d["cached"] and cache_size() == 3


def test_curriculum_plugin_returns_scene_with_increasing_tau(env):
    from arm_rl.plugins.curriculum import Plugin

    env.reset(seed=3)
    plugin = Plugin({"candidates": 4, "n_samples": 600, "episodes": 40, "frac": 1.0, "q_start": 0.0, "q_end": 1.0}, env, None, phase="train")
    assert plugin.warm_episodes == 40
    chosen = []
    for ep in range(1, 41):
        s = plugin.choose_scene(ep)
        assert isinstance(s, Scene)
        chosen.append(s)
    taus = np.array([h[2] for h in plugin.history])
    assert len(taus) == 40
    assert taus[:10].mean() < taus[-10:].mean()
    assert plugin.cost["time_s"] > 0.0
    assert plugin.choose_scene(41) is None  # past the warm-up: uniform sampling
    st = plugin.stats()
    assert st["tau_first_fifth"] < st["tau_last_fifth"]
    # eval phase is a no-op
    assert Plugin({"episodes": 10}, env, None, phase="eval").choose_scene(1) is None


def test_homotopy_signature_sides():
    cap = Scene(np.array([[0.4, 0.0, 0.0]]), np.array([[0.4, 0.0, 0.6]]), 0.04, np.array([0.6, 0.0, 0.3]), 0.03)
    t = np.linspace(0, 1, 50)[:, None]
    left = np.hstack([0.2 + 0.4 * t, 0.15 * np.sin(np.pi * t), 0.3 + 0 * t])
    right = left.copy()
    right[:, 1] *= -1
    straight = np.hstack([0.2 + 0.4 * t, 0.0 * t + 0.5, 0.3 + 0 * t])  # far from the capsule: no side
    s_left, s_right, s_str = (path_homotopy_signature(p, cap) for p in (left, right, straight))
    assert len(s_left) == 1 and s_left != s_right and {s_left[0], s_right[0]} == {-1, 1}
    assert s_str == (0,)
    assert path_homotopy_signature(left[::-1], cap) == s_right  # reversed path = other side
    # independent of the path resolution
    assert path_homotopy_signature(left[::5], cap) == s_left
    # empty scene -> empty signature; stats
    assert path_homotopy_signature(left, empty_scene()) == ()
    st = signature_stats([s_left, s_left, s_right])
    assert st["n_classes"] == 2 and st["top_share"] == pytest.approx(2 / 3) and st["nontrivial_fraction"] == 1.0
