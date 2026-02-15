from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vector3 = npt.NDArray[np.float64]

# MediaPipe 468-landmark eye contours (without iris refinement landmarks).
LEFT_EYE_LANDMARKS = np.array(
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    dtype=np.int32,
)
RIGHT_EYE_LANDMARKS = np.array(
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    dtype=np.int32,
)
# Align MediaPipe template orientation with viewer head-frame convention.
FACE_TEMPLATE_AXIS_FLIP = np.array([1.0, -1.0, 1.0], dtype=np.float64)


def _empty_face_mesh() -> npt.NDArray[np.float64]:
    return np.empty((0, 3), dtype=np.float64)


def load_face_mesh_template_m() -> npt.NDArray[np.float64]:
    """Load MediaPipe face template in meters from project-level template file."""
    try:
        from face_mesh_template_3D import face_mesh_template_3D_m
    except Exception:
        return _empty_face_mesh()

    points = np.asarray(face_mesh_template_3D_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return _empty_face_mesh()
    return (points * FACE_TEMPLATE_AXIS_FLIP).copy()


def estimate_eye_center_head(face_mesh_points_head: npt.ArrayLike) -> Vector3:
    """Estimate eye-center origin in head coordinates from template landmarks."""
    points = np.asarray(face_mesh_points_head, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)

    centers: list[Vector3] = []
    for eye_indices in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
        valid_indices = eye_indices[eye_indices < points.shape[0]]
        if valid_indices.size == 0:
            continue
        centers.append(np.mean(points[valid_indices], axis=0))

    if not centers:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)
