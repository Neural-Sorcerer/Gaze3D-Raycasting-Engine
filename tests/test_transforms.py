from __future__ import annotations

import numpy as np

from gaze3d_lab.core.transforms import (
    RigidTransform,
    euler_xyz_to_matrix,
    look_at_rotation,
    matrix_to_euler_xyz,
    transform_ray,
)


def test_transform_inverse_round_trip_point() -> None:
    tf = RigidTransform(
        rotation=euler_xyz_to_matrix(0.1, -0.2, 0.3),
        translation=np.array([0.5, -0.4, 1.2]),
    )

    point = np.array([0.2, -0.1, 0.7])
    mapped = tf.apply_point(point)
    recovered = tf.inverse().apply_point(mapped)

    assert np.allclose(recovered, point, atol=1e-8)


def test_transform_compose_matches_sequential_application() -> None:
    a = RigidTransform(
        rotation=euler_xyz_to_matrix(0.05, 0.1, -0.08),
        translation=np.array([0.2, 0.3, -0.1]),
    )
    b = RigidTransform(
        rotation=euler_xyz_to_matrix(-0.04, 0.2, 0.11),
        translation=np.array([-0.5, 0.1, 0.2]),
    )

    point = np.array([0.1, 0.2, 0.3])
    seq = a.apply_point(b.apply_point(point))
    composed = a.compose(b).apply_point(point)

    assert np.allclose(composed, seq, atol=1e-8)


def test_euler_matrix_round_trip() -> None:
    euler = np.array([0.2, -0.3, 0.25])
    matrix = euler_xyz_to_matrix(*euler)
    recovered = matrix_to_euler_xyz(matrix)

    assert np.allclose(recovered, euler, atol=1e-7)


def test_transform_ray() -> None:
    tf = RigidTransform(
        rotation=euler_xyz_to_matrix(0.0, 0.0, np.pi / 2.0),
        translation=np.array([1.0, 2.0, 0.0]),
    )

    origin_w, direction_w = transform_ray(
        origin=np.array([1.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        transform=tf,
    )

    assert np.allclose(origin_w, [1.0, 3.0, 0.0], atol=1e-8)
    assert np.allclose(direction_w, [0.0, 1.0, 0.0], atol=1e-8)


def test_look_at_rotation_forward_matches_target_direction() -> None:
    eye = np.array([0.0, -1.35, 1.1])
    target = np.array([0.0, 0.0, 0.25])
    rotation = look_at_rotation(eye=eye, target=target, up=np.array([0.0, 0.0, 1.0]))

    expected_forward = (target - eye) / np.linalg.norm(target - eye)
    assert np.allclose(rotation[:, 2], expected_forward, atol=1e-8)
