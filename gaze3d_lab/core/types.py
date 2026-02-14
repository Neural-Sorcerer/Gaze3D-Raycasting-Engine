from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .transforms import RigidTransform, normalize_vector

Vector3 = npt.NDArray[np.float64]


@dataclass
class GazeSample:
    """Single timestamped sample from a data source."""

    timestamp: float
    head_transform_cam: RigidTransform
    gaze_direction_head: Vector3

    def __post_init__(self) -> None:
        self.gaze_direction_head = normalize_vector(self.gaze_direction_head)
