import numpy as np

from arm_rl.geometry import closest_points_between_segments, point_segment_distance, segment_distance_matrix


def test_parallel_segments():
    t, s, pc, qc, d = closest_points_between_segments([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])
    assert np.isclose(d, 1.0)
    assert np.allclose(qc - pc, [0, 1, 0])


def test_skew_segments():
    _, _, pc, qc, d = closest_points_between_segments([-1, 0, 0], [1, 0, 0], [0, -1, 2], [0, 1, 2])
    assert np.isclose(d, 2.0)
    assert np.allclose(pc, [0, 0, 0]) and np.allclose(qc, [0, 0, 2])


def test_endpoint_clamping():
    _, _, pc, qc, d = closest_points_between_segments([0, 0, 0], [1, 0, 0], [3, 0, 1], [4, 0, 1])
    assert np.isclose(d, np.sqrt(5.0))
    assert np.allclose(pc, [1, 0, 0]) and np.allclose(qc, [3, 0, 1])


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p1, p2, q1, q2 = rng.normal(size=(4, 3))
        _, _, _, _, d = closest_points_between_segments(p1, p2, q1, q2)
        ts = np.linspace(0, 1, 201)
        P = p1[None] + ts[:, None] * (p2 - p1)[None]
        Q = q1[None] + ts[:, None] * (q2 - q1)[None]
        brute = np.min(np.linalg.norm(P[:, None] - Q[None], axis=-1))
        assert d <= brute + 1e-9 and d >= brute - 2e-2


def test_matrix_shapes():
    rng = np.random.default_rng(1)
    dist, direction = segment_distance_matrix(rng.normal(size=(6, 3)), rng.normal(size=(6, 3)), rng.normal(size=(5, 3)), rng.normal(size=(5, 3)))
    assert dist.shape == (6, 5) and direction.shape == (6, 5, 3)
    assert np.allclose(np.linalg.norm(direction, axis=-1), dist)


def test_point_segment():
    d = point_segment_distance(np.array([0.5, 1.0, 0.0]), np.array([[0, 0, 0.0]]), np.array([[1, 0, 0.0]]))
    assert np.isclose(d[0], 1.0)
