import numpy as np

from arm_rl.env import ArmPathPlanningEnv
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")


def make_env(**env_over):
    env_cfg = dict(CFG["env"], **env_over)
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], env_cfg)


def test_state_is_27_dims():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (27,)
    assert env.observation_space.shape == (27,)
    # layout: Sl(15) Sa(6) Se(3) St(3)
    assert np.allclose(obs[15:21], env.q / np.pi if env.normalize_angles else env.q)
    assert np.allclose(obs[21:24], env.ee, atol=1e-6)
    assert np.allclose(obs[24:27], env.scene.target, atol=1e-6)


def test_reward_eq13():
    env = make_env()
    env.reset(seed=0)
    assert env.reward_for(0.5, False) == -5.0        # 50 cm -> d = 5
    assert env.reward_for(0.02, False) == 0.0        # inside 3 cm
    assert env.reward_for(0.5, True) == -50.0


def test_action_increment_bounded():
    env = make_env(collision_mode="continue")
    env.reset(seed=1)
    q0 = env.q.copy()
    env.step(np.ones(6))
    assert np.all(np.abs(env.q - q0) <= np.deg2rad(5.0) + 1e-9)


def test_episode_ends_at_100_steps():
    env = make_env()
    env.reset(seed=2)
    for i in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(6))
        if term:
            break
    assert term or trunc
    assert info["step"] <= 100


def test_collision_terminates_with_penalty():
    env = make_env(collision_mode="terminate")
    env.reset(seed=3)
    # put an obstacle right next to the end effector
    ee = env.ee.copy()
    env.scene.obs_q1 = np.vstack([env.scene.obs_q1, ee + [0.05, 0, -0.2]])
    env.scene.obs_q2 = np.vstack([env.scene.obs_q2, ee + [0.05, 0, 0.2]])
    assert env.is_collision(env.q)  # the obstacle overlaps the EE link
    _, r, term, trunc, info = env.step(np.ones(6) * 0.1)
    assert r == -50.0 and info["collision"]
    assert term
