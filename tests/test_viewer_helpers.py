from __future__ import annotations

from gaze3d_lab.viz.viewer import _build_recording_output_path


def test_build_recording_output_path_uses_mode_and_extension(tmp_path) -> None:
    path = _build_recording_output_path(tmp_path, "synthetic", timestamp="20260327_101500")

    assert path == tmp_path / "gaze3d_lab_synthetic_20260327_101500.mov"


def test_build_recording_output_path_avoids_name_collisions(tmp_path) -> None:
    existing = tmp_path / "gaze3d_lab_webcam_20260327_101500.mov"
    existing.write_bytes(b"existing")

    path = _build_recording_output_path(tmp_path, "webcam", timestamp="20260327_101500")

    assert path == tmp_path / "gaze3d_lab_webcam_20260327_101500_01.mov"
