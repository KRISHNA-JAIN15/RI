import numpy as np

from arm_rl.scene import ConfigurableSceneGenerator, Motion, Scene, make_scene_generator
from arm_rl.kinematics import SixAxisArm


def test_oscillate_motion_period():
    rng = np.random.default_rng(0)
    m = Motion({"type": "oscillate", "axis": [0, 1, 0], "amplitude": 0.1, "period": 2.0, "phase": 0.0}, rng)
    base = np.array([[0.5, 0.0, 0.3], [0.5, 0.0, 0.6]])
    assert np.allclose(m.apply(base, 0.0), base)
    assert np.allclose(m.apply(base, 0.5), base + [0, 0.1, 0])
    assert np.allclose(m.apply(base, 2.0), base, atol=1e-9)


def test_linear_bounce_stays_in_bounds():
    rng = np.random.default_rng(0)
    m = Motion({"type": "linear", "axis": [0, 1, 0], "speed": 0.3, "bounds": [-0.2, 0.2], "phase": 0.0}, rng)
    base = np.zeros((1, 3))
    ys = [m.apply(base, t)[0, 1] for t in np.linspace(0, 10, 500)]
    assert min(ys) >= -0.2 - 1e-9 and max(ys) <= 0.2 + 1e-9
    assert max(ys) - min(ys) > 0.35  # actually travels


def test_conveyor_wraps():
    rng = np.random.default_rng(0)
    m = Motion({"type": "conveyor", "axis": [0, 1, 0], "speed": 0.1, "bounds": [-0.3, 0.3], "phase": 0.0}, rng)
    base = np.zeros((1, 3))
    y0 = m.apply(base, 0.0)[0, 1]
    y6 = m.apply(base, 6.0)[0, 1]  # 0.6 m later -> wrapped to the start
    assert np.isclose(y0, -0.3) and np.isclose(y6, -0.3, atol=1e-9)


def test_scene_advance_velocities():
    rng = np.random.default_rng(0)
    motion = Motion({"type": "linear", "axis": [1, 0, 0], "speed": 0.2, "bounds": [-1, 1], "phase": 0.0}, rng)
    sc = Scene(np.array([[0.5, 0, 0]]), np.array([[0.5, 0, 1.0]]), 0.04, np.array([0.3, 0, 0.3]), 0.03, obs_motion=[motion], target_motion=Motion(None, rng))
    assert sc.is_dynamic
    sc.advance(0.1)
    assert np.allclose(sc.obs_vel[0], [0.2, 0, 0], atol=1e-6)
    assert np.allclose(sc.target_vel, 0)


def test_configurable_generator_from_preset():
    arm = SixAxisArm(home_deg=[0, 90, -135, 45, 90, 0])
    gen = make_scene_generator(arm, {"preset": "industrial_cell", "target_radius": 0.03, "min_target_reach": 0.2, "max_target_reach": 0.6})
    assert isinstance(gen, ConfigurableSceneGenerator)
    sc = gen.sample(np.random.default_rng(1))
    assert sc.n_obstacles >= 19 and sc.is_dynamic
    assert sc.radii.shape == (sc.n_obstacles,)


def test_fruit_tree_wind_is_dynamic():
    arm = SixAxisArm()
    gen = make_scene_generator(arm, {"wind": {"amplitude_deg": 10, "period": 3}})
    sc = gen.sample(np.random.default_rng(0))
    assert sc.is_dynamic
    t0 = sc.target.copy()
    sc.advance(0.75)
    assert np.linalg.norm(sc.target - t0) > 1e-4
