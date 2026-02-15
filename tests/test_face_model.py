from __future__ import annotations

import numpy as np

from gaze3d_lab.core.face_model import estimate_eye_center_head, load_face_mesh_template_m


def test_load_face_mesh_template_m_returns_points() -> None:
    points = load_face_mesh_template_m()
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 0


def test_estimate_eye_center_head_returns_vector() -> None:
    points = load_face_mesh_template_m()
    eye_center = estimate_eye_center_head(points)
    assert eye_center.shape == (3,)
    assert np.isfinite(eye_center).all()
