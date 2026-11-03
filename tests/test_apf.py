"""C2: artificial-potential-field prior and potential-based shaping (arm_rl/apf.py, plugins/apf.py)."""

import numpy as np
import pytest

from arm_rl import apf
from arm_rl.env import ArmPathPlanningEnv
from arm_rl.plugins.apf import Plugin
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")


class _Agent:
    gamma = 0.95


def make_env(**env_over):
    env_cfg = dict(CFG["env"], **env_over)
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], env_cfg)


def run_apf_episode(env, plugin, seed):
    obs, info = env.reset(seed=seed)
    while True:
        a = plugin.transform_action(obs, np.zeros(6, dtype=np.float32), deterministic=True)
        assert a.shape == (6,) and np.all(np.abs(a) <= 1.0 + 1e-9)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return info


def test_pure_apf_controller_reaches_targets():
    """APF-only baseline on the paper's fruit-tree scenes: ~50 % success, few collisions."""
    env = make_env()
    p = Plugin({"prior_only": True, "shaping": False}, env, _Agent(), phase="eval")
    n = 20
    succ = coll = 0
    for ep in range(n):
        info = run_apf_episode(env, p, seed=1000 + ep)
        succ += int(info["reached"] and info["n_collisions"] == 0)
        coll += int(info["n_collisions"] > 0)
    print(f"APF-only: success {succ}/{n}, collisions {coll}/{n}, {1e3 * p.cost['time_s'] / p.cost['calls']:.2f} ms/call")
    assert p.cost["calls"] > 0 and p.cost["time_s"] > 0.0
    assert succ >= 0.2 * n
    assert coll <= n // 2


def test_attractive_step_moves_ee_towards_target():
    """Obstacle-free scene: the APF increment decreases |EE - target| every step."""
    env = make_env()
    env.reset(seed=5)
    env.scene.obs_q1 = np.zeros((0, 3))
    env.scene.obs_q2 = np.zeros((0, 3))
    env.scene.obs_radius = np.zeros(0)
    env.scene.obs_vel = np.zeros((0, 3))
    env.backend.set_scene(env.scene)
    env._geom_cache = None
    d_prev = env.distance_to_target()
    for _ in range(20):
        dq, det = apf.apf_joint_step(env, k_att=20.0, k_rep=1e-4, rho0=0.15, d_switch=0.1, dq_max=env.max_delta)
        assert np.max(np.abs(dq)) <= env.max_delta + 1e-12
        assert np.allclose(det["f_rep"], 0.0)
        obs, r, term, trunc, info = env.step(dq / env.max_delta)
        assert info["distance"] < d_prev
        d_prev = info["distance"]
        if term:
            break


def test_repulsive_force_points_away_and_vanishes_outside_rho0():
    f = apf.repulsive_force(0.05, np.array([1.0, 0.0, 0.0]), k_rep=1e-4, rho0=0.15)
    assert f[0] < 0.0 and np.allclose(f[1:], 0.0)
    assert np.allclose(apf.repulsive_force(0.2, np.array([1.0, 0.0, 0.0]), 1e-4, 0.15), 0.0)
    # closer -> stronger
    assert np.linalg.norm(apf.repulsive_force(0.02, np.ones(3), 1e-4, 0.15)) > np.linalg.norm(f)


def test_shaping_is_potential_based():
    env = make_env()
    agent = _Agent()
    p = Plugin({"prior": False, "shaping": True}, env, agent, phase="train")
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=7)
    for _ in range(3):
        a = rng.uniform(-1, 1, 6)
        next_obs, r, term, trunc, info = env.step(a)
        shaped = p.shape_reward(obs, a, r, next_obs, term, info)
        phi_s, phi_s2 = p.phi(obs), (0.0 if term else p.phi(next_obs))
        assert shaped == pytest.approx(r + agent.gamma * phi_s2 - phi_s, abs=1e-9)
        # the attractive part of Phi is -k_phi * distance_scale * |EE - T| at that state
        d = np.linalg.norm(obs[slice(*env.obs_layout["se"])] - obs[slice(*env.obs_layout["st"])])
        assert p.phi(obs) <= -p.k_phi * env.distance_scale * d + 1e-9
        if term or trunc:
            break
        obs = next_obs
    # terminal transition: Phi(s') = 0
    assert p.shape_reward(obs, a, -1.0, next_obs, True, {}) == pytest.approx(-1.0 - p.phi(obs))
    # shaping disabled -> raw reward
    q = Plugin({"shaping": False}, env, agent, phase="train")
    assert q.shape_reward(obs, a, -1.0, next_obs, False, {}) == -1.0


def test_residual_prior_adds_apf_to_rl_action():
    env = make_env()
    p = Plugin({"prior": True, "shaping": False}, env, _Agent(), phase="train")
    obs, _ = env.reset(seed=3)
    a_rl = np.full(6, 0.3, dtype=np.float32)
    a_apf = p.apf_action()
    out = p.transform_action(obs, a_rl, deterministic=False)
    assert np.allclose(out, np.clip(a_rl + a_apf, -1, 1), atol=1e-6)
    off = Plugin({"prior": False, "shaping": False}, env, _Agent(), phase="train")
    assert np.allclose(off.transform_action(obs, a_rl, False), a_rl)


def test_plugin_runs_on_mujoco_backend():
    pytest.importorskip("mujoco")
    cfg = load_config("configs/ur5e_industrial.yaml")
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    p = Plugin({"prior": True, "shaping": True}, env, _Agent(), phase="train")
    assert p.margin == 0.0
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        a = p.transform_action(obs, np.zeros(6, dtype=np.float32), False)
        assert a.shape == (6,) and np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
        next_obs, r, term, trunc, info = env.step(a)
        shaped = p.shape_reward(obs, a, r, next_obs, term, info)
        assert np.isfinite(shaped)
        obs = next_obs
        if term or trunc:
            break
    assert p.cost["calls"] >= 1
