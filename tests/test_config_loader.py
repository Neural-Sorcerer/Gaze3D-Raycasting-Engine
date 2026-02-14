from __future__ import annotations

import numpy as np

from gaze3d_lab.io.config_loader import load_app_config


def test_default_camera_look_at_points_to_scene_target() -> None:
    cfg = load_app_config("config/default.yaml")

    camera_pos = cfg.camera_pose_world.translation
    target = np.array([0.0, 0.0, 0.25], dtype=np.float64)

    expected = target - camera_pos
    expected /= np.linalg.norm(expected)

    forward = cfg.camera_pose_world.rotation[:, 2]
    assert float(np.dot(forward, expected)) > 0.999


def test_synthetic_stress_config_overrides_defaults() -> None:
    cfg = load_app_config("config/synthetic_stress.yaml")

    assert cfg.synthetic.motion_scale == 1.2
    assert cfg.synthetic.noise_std == 0.035
    assert cfg.synthetic.micro_saccade_strength == 0.08
