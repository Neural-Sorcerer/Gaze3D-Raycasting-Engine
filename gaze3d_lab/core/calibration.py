from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vector3 = npt.NDArray[np.float64]


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def pixel_to_camera(self, u: float, v: float, depth: float) -> Vector3:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth], dtype=np.float64)

    def frustum_corners(self, z_near: float, z_far: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return near/far plane corner points in camera coordinates."""
        near = np.array(
            [
                self.pixel_to_camera(0.0, 0.0, z_near),
                self.pixel_to_camera(float(self.width), 0.0, z_near),
                self.pixel_to_camera(float(self.width), float(self.height), z_near),
                self.pixel_to_camera(0.0, float(self.height), z_near),
            ],
            dtype=np.float64,
        )
        far = np.array(
            [
                self.pixel_to_camera(0.0, 0.0, z_far),
                self.pixel_to_camera(float(self.width), 0.0, z_far),
                self.pixel_to_camera(float(self.width), float(self.height), z_far),
                self.pixel_to_camera(0.0, float(self.height), z_far),
            ],
            dtype=np.float64,
        )
        return near, far
