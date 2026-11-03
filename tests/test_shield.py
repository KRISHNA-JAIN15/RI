"""Tests for the C1 control-barrier-function shield (arm_rl/safety.py, arm_rl/plugins/shield.py)."""

import numpy as np
import pytest

from arm_rl.env import ArmPathPlanningEnv
from arm_rl.plugins.shield import Plugin
from arm_rl.safety import _HAVE_QPSOLVERS, solve_box_qp
from arm_rl.scene import Scene
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")
D_SAFE = 0.02


def make_env(**env_over):
    env_cfg = dict(CFG["env"], **env_over)
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], env_cfg)


def frontal_scene(env, gap: float = 0.15) -> Scene:
    """One vertical capsule ``gap`` metres ahead of the end effector along the last link."""
    env.reset(seed=0)
    origins, _ = env.arm.forward(env.arm.home)
    d = origins[6] - origins[5]
    d /= np.linalg.norm(d)
    c = env.arm.ee_position(env.arm.home) + gap * d
    return Scene(np.array([c + [0, 0, -0.3]]), np.array([c + [0, 0, 0.3]]), 0.04, c + 0.4 * d, 0.03)


def approach_action(plugin: Plugin) -> np.ndarray:
    """Full-speed increment that drives the EE link's closest point into obstacle 0."""
    cons = [c for c in plugin.shield.constraints() if c.link == 5 and c.obstacle == 0]
    assert cons, "no EE-link constraint found"
    row = cons[0].row
    a = np.sign(row)
    a[np.abs(row) < 1e-9] = 0.0
    return a


# ---------------------------------------------------------------------------- 1. far from obstacles
def test_far_from_obstacles_is_identity():
    env = make_env()
    far = Scene(np.array([[1.5, 1.5, 0.0]]), np.array([[1.5, 1.5, 0.8]]), 0.04, np.array([0.3, 0.0, 0.3]), 0.03)
    obs, _ = env.reset(seed=0, options={"scene": far})
    p = Plugin({"d_safe": D_SAFE}, env, None, phase="eval")
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = rng.uniform(-1, 1, 6).astype(np.float32)
        out = p.transform_action(obs, a, False)
        assert np.allclose(out, a, atol=1e-9)
        obs, *_ = env.step(out * 0.1)  # stay far away
    assert p.cost["calls"] == 10 and p.cost["activations"] == 0


# ---------------------------------------------------------------------------- 2. approach an obstacle
@pytest.mark.parametrize("solver", ["auto", "numpy"])
def test_shield_keeps_d_safe_while_unfiltered_collides(solver):
    env = make_env(collision_mode="continue")
    scene = frontal_scene(env)

    # unfiltered: full-speed approach collides
    obs, _ = env.reset(seed=0, options={"scene": scene})
    p = Plugin({"d_safe": D_SAFE, "solver": solver}, env, None, phase="eval")
    a = approach_action(p)
    assert env.min_obstacle_clearance() > D_SAFE
    clear_raw = []
    for _ in range(30):
        _, _, _, _, info = env.step(a)
        clear_raw.append(env.min_obstacle_clearance())
    assert min(clear_raw) < 0.0 and info["n_collisions"] > 0

    # filtered: never below d_safe (1 mm tolerance for the one-step linearisation error)
    obs, _ = env.reset(seed=0, options={"scene": scene})
    p = Plugin({"d_safe": D_SAFE, "solver": solver}, env, None, phase="eval")
    clear = []
    for _ in range(30):
        act = p.transform_action(obs, a, True)
        assert np.all(np.abs(act) <= 1.0 + 1e-9)
        obs, _, _, _, info = env.step(act)
        clear.append(env.min_obstacle_clearance())
    assert min(clear) >= D_SAFE - 1e-3, clear
    assert info["n_collisions"] == 0
    assert p.cost["activations"] > 0 and p.cost["infeasible"] == 0
    assert p.stats()["mean_solve_ms"] < 50.0
    # the shield lets the arm come close to the barrier instead of freezing it
    assert min(clear) < D_SAFE + 0.01


def test_train_time_flag_disables_during_training():
    env = make_env(collision_mode="continue")
    scene = frontal_scene(env)
    obs, _ = env.reset(seed=0, options={"scene": scene})
    p_on = Plugin({"d_safe": D_SAFE}, env, None, phase="eval")
    a = approach_action(p_on)
    p_off = Plugin({"d_safe": D_SAFE, "train_time": False}, env, None, phase="train")
    for _ in range(8):
        obs, *_ = env.step(a)
    assert np.allclose(p_off.transform_action(obs, a, False), a)
    assert not np.allclose(p_on.transform_action(obs, a, True), a)


def test_inside_d_safe_retreats():
    """Starting inside d_safe is still feasible (the CBF demands a retreat of alpha*|h|)."""
    env = make_env(collision_mode="continue")
    scene = frontal_scene(env, gap=0.09)  # surface clearance 1 cm < d_safe
    obs, _ = env.reset(seed=0, options={"scene": scene})
    assert 0.0 < env.min_obstacle_clearance() < D_SAFE
    p = Plugin({"d_safe": D_SAFE}, env, None, phase="eval")
    c0 = env.min_obstacle_clearance()
    a = approach_action(p)
    env.step(p.transform_action(obs, a, True))
    assert p.cost["infeasible"] == 0
    assert env.min_obstacle_clearance() > c0


def test_infeasible_fallback_moves_away():
    """Deep overlap with alpha = 1 asks for a retreat the box cannot deliver: least-violating fallback."""
    env = make_env(collision_mode="continue")
    scene = frontal_scene(env, gap=0.02)  # surface clearance -6 cm
    obs, _ = env.reset(seed=0, options={"scene": scene})
    p = Plugin({"d_safe": D_SAFE, "alpha": 1.0}, env, None, phase="eval")
    c0 = env.min_obstacle_clearance()
    assert c0 < 0.0
    a = approach_action(p)
    act = p.transform_action(obs, a, True)
    assert np.all(np.abs(act) <= 1.0 + 1e-9)
    env.step(act)
    assert p.cost["infeasible"] == 1
    assert env.min_obstacle_clearance() > c0 + 0.01


# ---------------------------------------------------------------------------- 3. solvers agree
@pytest.mark.skipif(not _HAVE_QPSOLVERS, reason="qpsolvers/osqp not installed")
def test_osqp_and_numpy_fallback_agree():
    rng = np.random.default_rng(1)
    n = 6
    for _ in range(5):
        m = int(rng.integers(5, 25))
        G = rng.normal(size=(m, n))
        G[:, rng.integers(n)] = 0.0  # a joint no constraint depends on
        x_feas = rng.uniform(-0.05, 0.05, n)
        h = G @ x_feas + rng.uniform(0.0, 0.02, m)
        lb, ub = -0.087 * np.ones(n), 0.087 * np.ones(n)
        x0 = rng.uniform(-0.2, 0.2, n)
        xa, oka = solve_box_qp(x0, G, h, lb, ub, solver="osqp")
        xb, okb = solve_box_qp(x0, G, h, lb, ub, solver="numpy", max_iter=20000)
        assert oka and okb
        assert np.max(np.abs(xa - xb)) < 1e-4
        assert np.all(G @ xb <= h + 1e-6) and np.all(xb >= lb - 1e-9) and np.all(xb <= ub + 1e-9)

    # end-to-end: same filtered action from both solvers on the frontal scene
    env = make_env(collision_mode="continue")
    scene = frontal_scene(env, gap=0.11)
    obs, _ = env.reset(seed=0, options={"scene": scene})
    pa = Plugin({"d_safe": D_SAFE, "solver": "osqp"}, env, None, phase="eval")
    pb = Plugin({"d_safe": D_SAFE, "solver": "numpy", "max_iter": 20000}, env, None, phase="eval")
    a = approach_action(pa)
    assert np.max(np.abs(pa.transform_action(obs, a, True) - pb.transform_action(obs, a, True))) < 1e-4


# ---------------------------------------------------------------------------- 4. MuJoCo backend
def test_mujoco_backend_runs_and_respects_barrier():
    pytest.importorskip("mujoco")
    cfg = load_config("configs/ur5e_industrial.yaml")
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], dict(cfg["env"], collision_mode="continue"))
    obs, _ = env.reset(seed=2)
    p = Plugin({"d_safe": D_SAFE}, env, None, phase="eval")
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, 6)
    for k in range(20):
        if k % 5 == 0:
            a = rng.uniform(-1, 1, 6)
        act = p.transform_action(obs, a, True)
        assert act.shape == (6,) and np.all(np.abs(act) <= 1.0 + 1e-9)
        feasible = p.last_info["feasible"]
        obs, _, _, _, info = env.step(act)
        if feasible:
            assert env.min_obstacle_clearance() >= D_SAFE - 1e-3, (k, env.min_obstacle_clearance())
            assert not info["collision"]
    assert p.cost["calls"] == 20
    assert p.stats()["mean_solve_ms"] < 50.0
