from __future__ import annotations

import numpy as np

from gaze3d_lab.geometry.intersections import (
    ray_aabb_intersection,
    ray_obb_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
)


def test_ray_plane_intersection_hit() -> None:
    hit = ray_plane_intersection(
        ray_origin=np.array([0.0, 0.0, 1.0]),
        ray_direction=np.array([0.0, 0.0, -1.0]),
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )

    assert hit is not None
    assert np.isclose(hit.t, 1.0)
    assert np.allclose(hit.point, [0.0, 0.0, 0.0])


def test_ray_plane_parallel_returns_none() -> None:
    hit = ray_plane_intersection(
        ray_origin=np.array([0.0, 0.0, 1.0]),
        ray_direction=np.array([1.0, 0.0, 0.0]),
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert hit is None


def test_ray_aabb_intersection_hit() -> None:
    hit = ray_aabb_intersection(
        ray_origin=np.array([-2.0, 0.0, 0.0]),
        ray_direction=np.array([1.0, 0.0, 0.0]),
        min_corner=np.array([-0.5, -0.5, -0.5]),
        max_corner=np.array([0.5, 0.5, 0.5]),
    )

    assert hit is not None
    assert np.isclose(hit.t, 1.5)
    assert np.allclose(hit.point, [-0.5, 0.0, 0.0])


def test_ray_aabb_inside_box_uses_exit_intersection() -> None:
    hit = ray_aabb_intersection(
        ray_origin=np.array([0.0, 0.0, 0.0]),
        ray_direction=np.array([0.0, 1.0, 0.0]),
        min_corner=np.array([-1.0, -1.0, -1.0]),
        max_corner=np.array([1.0, 1.0, 1.0]),
    )

    assert hit is not None
    assert np.isclose(hit.t, 1.0)
    assert np.allclose(hit.point, [0.0, 1.0, 0.0])


def test_ray_sphere_intersection_hit() -> None:
    hit = ray_sphere_intersection(
        ray_origin=np.array([0.0, 0.0, -2.0]),
        ray_direction=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
    )

    assert hit is not None
    assert np.isclose(hit.t, 1.0)
    assert np.allclose(hit.point, [0.0, 0.0, -1.0])


def test_ray_obb_intersection_hit_with_yaw() -> None:
    hit = ray_obb_intersection(
        ray_origin=np.array([0.0, -2.0, 0.0]),
        ray_direction=np.array([0.0, 1.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([2.0, 1.0, 2.0]),
        euler_deg=np.array([0.0, 0.0, 30.0]),
    )

    assert hit is not None
    assert hit.t > 0.0
    assert np.isclose(np.linalg.norm(hit.normal), 1.0)
