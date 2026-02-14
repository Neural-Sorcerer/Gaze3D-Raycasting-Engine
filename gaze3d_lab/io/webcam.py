from __future__ import annotations

import time

import numpy as np

from gaze3d_lab.core.scene import WebcamConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector
from gaze3d_lab.core.types import GazeSample

from .datasource import DataSource

try:
    import cv2
except ImportError:  # pragma: no cover - exercised in runtime envs only.
    cv2 = None


class WebcamDataSource(DataSource):
    """Webcam ingestion with mock estimator placeholder for real models."""

    def __init__(self, config: WebcamConfig) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for webcam mode")

        self._config = config
        self._cap = cv2.VideoCapture(self._config.camera_index)
        self._start_time = 0.0
        self._luma_ema = 0.5

    def start(self) -> None:
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open webcam index {self._config.camera_index}")
        self._start_time = time.perf_counter()

    def next_sample(self, timestamp: float) -> GazeSample:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam frame capture failed")

        # Placeholder signal extraction. Replace this section with a real model later.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray) / 255.0)
        self._luma_ema = 0.92 * self._luma_ema + 0.08 * luma

        t = timestamp - self._start_time
        brightness_bias = (self._luma_ema - 0.5) * 0.5

        translation = np.array(
            [
                0.04 * np.sin(0.7 * t),
                -0.02 + 0.03 * np.cos(0.9 * t),
                0.68 + 0.02 * np.sin(0.4 * t),
            ],
            dtype=np.float64,
        )

        roll = 0.03 * np.sin(0.6 * t)
        pitch = 0.08 * np.sin(0.5 * t + 0.2)
        yaw = 0.14 * np.cos(0.65 * t) + brightness_bias

        gaze = np.array(
            [
                0.2 * np.sin(1.1 * t) + brightness_bias,
                -0.12 + 0.1 * np.cos(0.8 * t),
                1.0,
            ],
            dtype=np.float64,
        )

        return GazeSample(
            timestamp=timestamp,
            head_transform_cam=RigidTransform(
                rotation=euler_xyz_to_matrix(roll, pitch, yaw),
                translation=translation,
            ),
            gaze_direction_head=normalize_vector(gaze),
        )

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
