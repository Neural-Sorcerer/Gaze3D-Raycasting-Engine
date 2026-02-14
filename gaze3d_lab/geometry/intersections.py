from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from gaze3d_lab.core.scene import AABBObjectConfig, OBBObjectConfig, SceneObjectConfig, SphereObjectConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector

Vector3 = npt.NDArray[np.float64]


@dataclass
class IntersectionHit:
    t: float
    point: Vector3
    normal: Vector3
    object_name: str | None = None


def _as_vec3(value: Sequence[float] | npt.NDArray[np.float64], name: str) -> Vector3:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}")
    return arr


def ray_plane_intersection(
    ray_origin: Sequence[float] | npt.NDArray[np.float64],
    ray_direction: Sequence[float] | npt.NDArray[np.float64],
    plane_point: Sequence[float] | npt.NDArray[np.float64],
    plane_normal: Sequence[float] | npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> IntersectionHit | None:
    origin = _as_vec3(ray_origin, "ray_origin")
    direction = normalize_vector(ray_direction)
    p0 = _as_vec3(plane_point, "plane_point")
    normal = normalize_vector(plane_normal)

    denom = float(np.dot(direction, normal))
    if abs(denom) < eps:
        return None

    t = float(np.dot(p0 - origin, normal) / denom)
    if t < 0.0:
        return None

    point = origin + t * direction
    return IntersectionHit(t=t, point=point, normal=normal, object_name="plane")


def ray_aabb_intersection(
    ray_origin: Sequence[float] | npt.NDArray[np.float64],
    ray_direction: Sequence[float] | npt.NDArray[np.float64],
    min_corner: Sequence[float] | npt.NDArray[np.float64],
    max_corner: Sequence[float] | npt.NDArray[np.float64],
    eps: float = 1e-10,
) -> IntersectionHit | None:
    origin = _as_vec3(ray_origin, "ray_origin")
    direction = _as_vec3(ray_direction, "ray_direction")
    if float(np.linalg.norm(direction)) < eps:
        return None
    min_c = _as_vec3(min_corner, "min_corner")
    max_c = _as_vec3(max_corner, "max_corner")

    t_min = -np.inf
    t_max = np.inf

    for axis in range(3):
        d = direction[axis]
        if abs(float(d)) < eps:
            if origin[axis] < min_c[axis] or origin[axis] > max_c[axis]:
                return None
            continue

        inv_d = 1.0 / d
        t1 = (min_c[axis] - origin[axis]) * inv_d
        t2 = (max_c[axis] - origin[axis]) * inv_d

        enter = min(t1, t2)
        exit_ = max(t1, t2)

        t_min = max(t_min, enter)
        t_max = min(t_max, exit_)

        if t_min > t_max:
            return None

    if t_max < 0.0:
        return None

    t_hit = t_min if t_min >= 0.0 else t_max
    point = origin + t_hit * direction

    # Compute outward normal from hit point proximity to slab boundaries.
    normal = np.zeros(3, dtype=np.float64)
    tol = 1e-5
    for axis in range(3):
        if abs(point[axis] - min_c[axis]) < tol:
            normal[axis] = -1.0
            break
        if abs(point[axis] - max_c[axis]) < tol:
            normal[axis] = 1.0
            break

    if not np.any(normal):
        axis = int(np.argmax(np.abs(direction)))
        normal[axis] = -np.sign(direction[axis])

    return IntersectionHit(t=float(t_hit), point=point, normal=normal, object_name=None)


def ray_sphere_intersection(
    ray_origin: Sequence[float] | npt.NDArray[np.float64],
    ray_direction: Sequence[float] | npt.NDArray[np.float64],
    center: Sequence[float] | npt.NDArray[np.float64],
    radius: float,
) -> IntersectionHit | None:
    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    origin = _as_vec3(ray_origin, "ray_origin")
    direction = normalize_vector(ray_direction)
    c = _as_vec3(center, "center")

    oc = origin - c
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(oc, direction))
    c_term = float(np.dot(oc, oc) - radius * radius)

    disc = b * b - 4.0 * a * c_term
    if disc < 0.0:
        return None

    sqrt_disc = float(np.sqrt(max(0.0, disc)))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t0, t1) if t >= 0.0]
    if not candidates:
        return None

    t_hit = min(candidates)
    point = origin + t_hit * direction
    normal = normalize_vector(point - c)
    return IntersectionHit(t=float(t_hit), point=point, normal=normal, object_name=None)


def ray_obb_intersection(
    ray_origin: Sequence[float] | npt.NDArray[np.float64],
    ray_direction: Sequence[float] | npt.NDArray[np.float64],
    center: Sequence[float] | npt.NDArray[np.float64],
    size: Sequence[float] | npt.NDArray[np.float64],
    euler_deg: Sequence[float] | npt.NDArray[np.float64],
) -> IntersectionHit | None:
    center_v = _as_vec3(center, "center")
    size_v = _as_vec3(size, "size")
    euler = np.radians(_as_vec3(euler_deg, "euler_deg"))
    if np.any(size_v <= 0.0):
        raise ValueError("OBB size must be > 0")

    box_tf = RigidTransform(rotation=euler_xyz_to_matrix(*euler), translation=center_v)
    box_inv = box_tf.inverse()

    origin_local = box_inv.apply_point(ray_origin)
    direction_local = box_inv.apply_vector(ray_direction)
    half = size_v * 0.5

    local_hit = ray_aabb_intersection(origin_local, direction_local, -half, half)
    if local_hit is None:
        return None

    point_world = box_tf.apply_point(local_hit.point)
    normal_world = normalize_vector(box_tf.apply_vector(local_hit.normal))

    return IntersectionHit(t=float(local_hit.t), point=point_world, normal=normal_world, object_name=None)


def intersect_scene_objects(
    ray_origin: Sequence[float] | npt.NDArray[np.float64],
    ray_direction: Sequence[float] | npt.NDArray[np.float64],
    objects: list[SceneObjectConfig],
) -> IntersectionHit | None:
    best_hit: IntersectionHit | None = None

    for obj in objects:
        hit: IntersectionHit | None
        if isinstance(obj, AABBObjectConfig):
            hit = ray_aabb_intersection(ray_origin, ray_direction, obj.min_corner, obj.max_corner)
        elif isinstance(obj, OBBObjectConfig):
            hit = ray_obb_intersection(ray_origin, ray_direction, obj.center, obj.size, obj.euler_deg)
        elif isinstance(obj, SphereObjectConfig):
            hit = ray_sphere_intersection(ray_origin, ray_direction, obj.center, obj.radius)
        else:
            continue

        if hit is None:
            continue

        hit.object_name = obj.name
        if best_hit is None or hit.t < best_hit.t:
            best_hit = hit

    return best_hit
