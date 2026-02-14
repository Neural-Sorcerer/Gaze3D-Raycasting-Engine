"""Geometry and intersection primitives."""

from .intersections import (
    IntersectionHit,
    intersect_scene_objects,
    ray_aabb_intersection,
    ray_obb_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
)

__all__ = [
    "IntersectionHit",
    "intersect_scene_objects",
    "ray_aabb_intersection",
    "ray_obb_intersection",
    "ray_plane_intersection",
    "ray_sphere_intersection",
]
