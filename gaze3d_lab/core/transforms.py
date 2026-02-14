from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

Vector3 = npt.NDArray[np.float64]
Matrix3 = npt.NDArray[np.float64]


def _as_vector3(value: Sequence[float] | npt.NDArray[np.float64], name: str) -> Vector3:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}")
    return arr


def _as_matrix3(value: Sequence[Sequence[float]] | npt.NDArray[np.float64], name: str) -> Matrix3:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be shape (3, 3), got {arr.shape}")
    return arr


def normalize_vector(vec: Sequence[float] | npt.NDArray[np.float64], eps: float = 1e-9) -> Vector3:
    arr = _as_vector3(vec, "vec")
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return arr / norm


@dataclass
class RigidTransform:
    """Rigid transform mapping local frame coordinates into parent frame coordinates."""

    rotation: Matrix3
    translation: Vector3

    def __post_init__(self) -> None:
        self.rotation = _as_matrix3(self.rotation, "rotation")
        self.translation = _as_vector3(self.translation, "translation")

    @staticmethod
    def identity() -> "RigidTransform":
        return RigidTransform(rotation=np.eye(3, dtype=np.float64), translation=np.zeros(3, dtype=np.float64))

    def apply_point(self, point: Sequence[float] | npt.NDArray[np.float64]) -> Vector3:
        p = _as_vector3(point, "point")
        return self.rotation @ p + self.translation

    def apply_vector(self, vector: Sequence[float] | npt.NDArray[np.float64]) -> Vector3:
        v = _as_vector3(vector, "vector")
        return self.rotation @ v

    def inverse(self) -> "RigidTransform":
        rot_t = self.rotation.T
        inv_t = -(rot_t @ self.translation)
        return RigidTransform(rotation=rot_t, translation=inv_t)

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """Return self ∘ other (apply other, then self)."""
        new_rot = self.rotation @ other.rotation
        new_t = self.rotation @ other.translation + self.translation
        return RigidTransform(rotation=new_rot, translation=new_t)


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> Matrix3:
    """Build rotation matrix from roll/pitch/yaw (radians)."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return rz @ ry @ rx


def matrix_to_euler_xyz(rotation: Sequence[Sequence[float]] | npt.NDArray[np.float64]) -> Vector3:
    """Recover roll/pitch/yaw in radians from R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r = _as_matrix3(rotation, "rotation")

    sy = -r[2, 0]
    sy_clamped = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy_clamped))

    if abs(sy_clamped) < 0.999999:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        # Gimbal lock fallback.
        roll = float(np.arctan2(-r[1, 2], r[1, 1]))
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float64)


def look_at_rotation(
    eye: Sequence[float] | npt.NDArray[np.float64],
    target: Sequence[float] | npt.NDArray[np.float64],
    up: Sequence[float] | npt.NDArray[np.float64] = (0.0, 0.0, 1.0),
    camera_y_down: bool = True,
) -> Matrix3:
    """Build camera rotation matrix (camera frame -> world frame) from a look-at target."""
    eye_v = _as_vector3(eye, "eye")
    target_v = _as_vector3(target, "target")
    up_v = normalize_vector(up)

    forward = normalize_vector(target_v - eye_v)
    right = np.cross(forward, up_v)
    if float(np.linalg.norm(right)) < 1e-8:
        # Forward nearly parallel to up; use a fallback up vector.
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = normalize_vector(right)

    camera_up = normalize_vector(np.cross(right, forward))
    y_axis = -camera_up if camera_y_down else camera_up
    rotation = np.column_stack((right, y_axis, forward))
    return _as_matrix3(rotation, "rotation")


def transform_ray(
    origin: Sequence[float] | npt.NDArray[np.float64],
    direction: Sequence[float] | npt.NDArray[np.float64],
    transform: RigidTransform,
) -> tuple[Vector3, Vector3]:
    ray_origin = transform.apply_point(origin)
    ray_direction = normalize_vector(transform.apply_vector(direction))
    return ray_origin, ray_direction
