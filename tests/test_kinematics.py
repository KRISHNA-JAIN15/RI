import numpy as np

from arm_rl.kinematics import SixAxisArm


def test_reach_is_61_2_cm():
    arm = SixAxisArm()
    assert np.isclose(arm.reach, 0.612, atol=5e-4)


def test_no_configuration_exceeds_reach():
    arm = SixAxisArm()
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(20000):
        q = rng.uniform(-np.pi, np.pi, 6)
        best = max(best, np.linalg.norm(arm.ee_position(q) - arm.shoulder))
    assert best <= arm.reach + 1e-6
    assert best > 0.95 * arm.reach  # the bound is tight


def test_segments_positive_length():
    arm = SixAxisArm()
    p1, p2 = arm.link_segments(arm.home)
    assert p1.shape == (6, 3)
    assert np.all(np.linalg.norm(p2 - p1, axis=1) > 1e-3)


def test_jacobian_matches_fd():
    arm = SixAxisArm()
    q = arm.home + 0.1
    J = arm.position_jacobian(q)
    dq = np.full(6, 1e-4)
    pred = J @ dq
    actual = arm.ee_position(q + dq) - arm.ee_position(q)
    assert np.allclose(pred, actual, atol=1e-6)
