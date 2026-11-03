import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.utils import load_config  # noqa: E402

CFG = load_config("configs/ur5e_industrial.yaml")


@pytest.fixture(scope="module")
def env():
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], CFG["env"])


def test_state_dim_with_velocities(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (45,)


def test_home_is_collision_free(env):
    env.reset(seed=1)
    assert not env.is_collision(env.arm.home)


def test_geometry_matches_contacts(env):
    """A configuration whose closest-point clearance is negative must register a contact."""
    env.reset(seed=2)
    rng = np.random.default_rng(0)
    found = False
    for _ in range(300):
        q = env.arm.sample_joints(rng)
        _, clearance, _, _ = env.backend.geometry(q)
        if clearance.size and clearance[1:].min() < -0.01:
            assert env.backend.is_collision(q)
            found = True
            break
    assert found


def test_obstacles_move_between_steps(env):
    env.reset(seed=3)
    q1_before = env.scene.obs_q1.copy()
    env.step(np.zeros(6))
    assert not np.allclose(q1_before, env.scene.obs_q1)


def test_dynamic_control_lags_command():
    cfg = dict(CFG["env"], sim=dict(CFG["env"]["sim"], control="dynamic"))
    env = ArmPathPlanningEnv(CFG["arm"], CFG["scene"], cfg)
    env.reset(seed=4)
    q0 = env.q.copy()
    env.step(np.ones(6))
    moved = np.rad2deg(np.abs(env.q - q0))
    assert np.all(moved > 0.5) and np.all(moved <= 5.0 + 1e-6)
