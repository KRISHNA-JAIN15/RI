"""C6: all-pairs observation (env.all_pairs) and the permutation-invariant SetEncoder."""

import numpy as np
import pytest
import torch

from arm_rl.ddpg import DDPGAgent
from arm_rl.env import ArmPathPlanningEnv
from arm_rl.networks import SetActor, SetCritic, make_actor
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")


def make_env(**env_over):
    env_cfg = dict(CFG["env"], **env_over)
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], env_cfg)


def _slices(layout):
    return sorted(v for v in layout.values() if isinstance(v, tuple))


def _assert_tiles(layout, dim):
    sl = _slices(layout)
    assert sl[0][0] == 0 and sl[-1][1] == dim
    for (a0, b0), (a1, b1) in zip(sl, sl[1:]):
        assert b0 == a1, "slices must tile the state vector"


# ------------------------------------------------------------------ observation layout
def test_default_layout_unchanged():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert env.state_dim == 27 and obs.shape == (27,)
    assert "pairs" not in env.obs_layout and "mask" not in env.obs_layout
    _assert_tiles(env.obs_layout, 27)
    assert env.obs_layout["sl"] == (0, 15) and env.obs_layout["st"] == (24, 27)


def test_all_pairs_layout_and_content():
    env = make_env(all_pairs=True, max_obstacles=8)
    obs, _ = env.reset(seed=0)
    L, K, D = 5, 8, 8
    assert env.obs_layout["pair_dim"] == D and env.obs_layout["max_obstacles"] == K
    assert env.state_dim == 27 + L * K * (D + 1) and obs.shape == (env.state_dim,)
    _assert_tiles(env.obs_layout, env.state_dim)
    # the paper's 27 dims are untouched in front
    base = make_env()
    base_obs, _ = base.reset(seed=0)
    assert np.allclose(obs[:27], base_obs)
    a, b = env.obs_layout["pairs"]
    pairs = obs[a:b].reshape(L, K, D)
    m0, m1 = env.obs_layout["mask"]
    mask = obs[m0:m1].reshape(L, K)
    M = min(env.scene.n_obstacles, K)
    assert M > 0
    assert np.all(mask[:, :M] == 1.0) and np.all(mask[:, M:] == 0.0)
    assert np.all(pairs[:, M:] == 0.0)
    # token = [direction (3), clearance (1), obs velocity (3), radius (1)]
    _, clearance, direction, _ = env._geometry()
    for li, link in enumerate(env.monitored_links):
        for j in range(M):
            assert np.allclose(pairs[li, j, :3], direction[link, j], atol=1e-6)
            assert np.isclose(pairs[li, j, 3], clearance[link, j], atol=1e-6)
            assert np.allclose(pairs[li, j, 4:7], env.scene.obs_vel[j], atol=1e-6)
            assert np.isclose(pairs[li, j, 7], env.scene.radii[j], atol=1e-6)
    # the nearest-obstacle vector Sl is the token with minimal clearance for that link
    sl = obs[:15].reshape(L, 3)
    for li in range(L):
        j = int(np.argmin(pairs[li, :M, 3]))
        assert np.allclose(sl[li], pairs[li, j, :3], atol=1e-6)
    assert np.all(np.isfinite(obs))


def test_all_pairs_with_velocities_layout():
    env = make_env(all_pairs=True, max_obstacles=4, include_velocities=True)
    obs, _ = env.reset(seed=1)
    assert env.state_dim == 45 + 5 * 4 * 9
    _assert_tiles(env.obs_layout, env.state_dim)
    assert env.obs_layout["vt"][1] == env.obs_layout["pairs"][0]
    obs2, *_ = env.step(np.zeros(6))
    assert obs2.shape == obs.shape and np.all(np.isfinite(obs2))


# ------------------------------------------------------------------ encoder invariances
def _permute_obstacles(obs, layout, perm):
    L, K, D = layout["n_links"], layout["max_obstacles"], layout["pair_dim"]
    out = obs.copy()
    a, b = layout["pairs"]
    out[a:b] = obs[a:b].reshape(L, K, D)[:, perm].ravel()
    m0, m1 = layout["mask"]
    out[m0:m1] = obs[m0:m1].reshape(L, K)[:, perm].ravel()
    return out


def test_set_actor_permutation_invariance():
    env = make_env(all_pairs=True, max_obstacles=8)
    obs, _ = env.reset(seed=3)
    torch.manual_seed(0)
    actor = SetActor(env.obs_layout, 6)
    critic = SetCritic(env.obs_layout, 6)
    rng = np.random.default_rng(0)
    x = torch.as_tensor(obs).unsqueeze(0)
    act = torch.zeros(1, 6)
    with torch.no_grad():
        y0 = actor(x)
        q0 = critic(x, act)
        for _ in range(5):
            perm = rng.permutation(8)  # permutes real and padded tokens alike
            xp = torch.as_tensor(_permute_obstacles(obs, env.obs_layout, perm)).unsqueeze(0)
            assert torch.allclose(actor(xp), y0, atol=1e-5)
            assert torch.allclose(critic(xp, act), q0, atol=1e-5)
    # not trivially constant in the tokens: zeroing the mask changes the output
    mask0 = obs.copy()
    m0, m1 = env.obs_layout["mask"]
    mask0[m0:m1] = 0.0
    with torch.no_grad():
        assert not torch.allclose(actor(torch.as_tensor(mask0).unsqueeze(0)), y0, atol=1e-5)


def test_set_actor_padding_invariance():
    """Same scene, more padding slots -> identical output (weights copied across)."""
    small = make_env(all_pairs=True, max_obstacles=8)
    big = make_env(all_pairs=True, max_obstacles=16)
    obs_s, _ = small.reset(seed=4)
    obs_b, _ = big.reset(seed=4)
    assert small.scene.n_obstacles == big.scene.n_obstacles <= 8
    torch.manual_seed(0)
    actor_s = SetActor(small.obs_layout, 6)
    actor_b = SetActor(big.obs_layout, 6)
    actor_b.load_state_dict(actor_s.state_dict())  # parameters do not depend on max_obstacles
    with torch.no_grad():
        ys = actor_s(torch.as_tensor(obs_s).unsqueeze(0))
        yb = actor_b(torch.as_tensor(obs_b).unsqueeze(0))
    assert torch.allclose(ys, yb, atol=1e-5)
    # batched and 1-D inputs agree
    with torch.no_grad():
        batch = torch.as_tensor(np.stack([obs_s, obs_s]))
        assert torch.allclose(actor_s(batch)[1], ys[0], atol=1e-6)
        assert torch.allclose(actor_s(torch.as_tensor(obs_s)), ys[0], atol=1e-6)


def test_set_encoder_all_masked_is_finite():
    env = make_env(all_pairs=True, max_obstacles=4)
    obs, _ = env.reset(seed=5)
    m0, m1 = env.obs_layout["mask"]
    obs[m0:m1] = 0.0
    actor = SetActor(env.obs_layout, 6)
    x = torch.as_tensor(obs).unsqueeze(0).requires_grad_(True)
    y = actor(x)
    assert torch.all(torch.isfinite(y))
    y.sum().backward()
    assert torch.all(torch.isfinite(x.grad))


def test_make_actor_requires_pairs_layout():
    env = make_env()
    with pytest.raises(ValueError, match="all_pairs"):
        make_actor(env.state_dim, 6, {"encoder": "set", "obs_layout": env.obs_layout})
    with pytest.raises(ValueError):
        make_actor(env.state_dim, 6, {"encoder": "set"})


# ------------------------------------------------------------------ DDPG update
def test_ddpg_update_with_set_encoder():
    env = make_env(all_pairs=True, max_obstacles=8)
    cfg = dict(CFG["ddpg"], encoder="set", obs_layout=env.obs_layout, warmup_steps=32, batch_size=16)
    agent = DDPGAgent(env.state_dim, 6, cfg, seed=0)
    assert isinstance(agent.actor, SetActor) and isinstance(agent.critic, SetCritic)
    obs, _ = env.reset(seed=6)
    for _ in range(40):
        a = agent.random_action()
        obs2, r, term, trunc, _ = env.step(a)
        agent.store(obs, a, r, obs2, term)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
    assert agent.ready()
    stats = agent.update()
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["actor_loss"])
    assert agent.act(obs).shape == (6,)


# ------------------------------------------------------------------ MuJoCo backend
def test_mujoco_all_pairs_shapes():
    pytest.importorskip("mujoco")
    cfg = load_config("configs/ur5e_industrial.yaml")
    env_cfg = dict(cfg["env"], all_pairs=True)
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], env_cfg)
    obs, _ = env.reset(seed=0)
    K = env.max_obstacles
    assert K == 32
    assert env.state_dim == 45 + 5 * K * 9 and obs.shape == (env.state_dim,)
    _assert_tiles(env.obs_layout, env.state_dim)
    assert np.all(np.isfinite(obs))
    m0, m1 = env.obs_layout["mask"]
    mask = obs[m0:m1].reshape(5, K)
    M = min(env.scene.n_obstacles, K)
    assert mask.sum() == 5 * M
    obs2, *_ = env.step(np.zeros(6))
    assert obs2.shape == obs.shape and np.all(np.isfinite(obs2))
    actor = make_actor(env.state_dim, 6, {"encoder": "set", "obs_layout": env.obs_layout})
    with torch.no_grad():
        assert actor(torch.as_tensor(obs2).unsqueeze(0)).shape == (1, 6)
