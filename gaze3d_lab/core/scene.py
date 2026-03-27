from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from .calibration import CameraIntrinsics
from .transforms import RigidTransform, normalize_vector

Vector3 = npt.NDArray[np.float64]
ColorRGBA = Tuple[float, float, float, float]


@dataclass
class PlaneConfig:
    point: Vector3
    normal: Vector3
    size: tuple[float, float] = (2.0, 2.0)
    color: ColorRGBA = (0.2, 0.8, 1.0, 0.18)

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64)
        self.normal = normalize_vector(self.normal)


@dataclass
class AABBObjectConfig:
    name: str
    min_corner: Vector3
    max_corner: Vector3
    color: ColorRGBA = (0.9, 0.4, 0.4, 0.35)

    def __post_init__(self) -> None:
        self.min_corner = np.asarray(self.min_corner, dtype=np.float64)
        self.max_corner = np.asarray(self.max_corner, dtype=np.float64)
        if np.any(self.max_corner <= self.min_corner):
            raise ValueError(f"Invalid AABB bounds for {self.name}")


@dataclass
class SphereObjectConfig:
    name: str
    center: Vector3
    radius: float
    color: ColorRGBA = (0.3, 0.9, 0.4, 0.35)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.radius <= 0.0:
            raise ValueError(f"Sphere radius must be > 0 for {self.name}")


@dataclass
class OBBObjectConfig:
    """Oriented box object defined by center, size, and Euler rotation (degrees)."""

    name: str
    center: Vector3
    size: Vector3
    euler_deg: Vector3
    color: ColorRGBA = (0.7, 0.7, 0.9, 0.35)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        self.euler_deg = np.asarray(self.euler_deg, dtype=np.float64)
        if np.any(self.size <= 0.0):
            raise ValueError(f"Invalid OBB size for {self.name}")


SceneObjectConfig = Union[AABBObjectConfig, OBBObjectConfig, SphereObjectConfig]


@dataclass
class FilterConfig:
    mode: str = "one_euro"
    ema_alpha: float = 0.35
    one_euro_min_cutoff: float = 1.2
    one_euro_beta: float = 0.02
    one_euro_d_cutoff: float = 1.0


@dataclass
class SyntheticConfig:
    seed: int = 7
    noise_std: float = 0.02
    motion_scale: float = 1.0
    translation_amplitude: tuple[float, float, float] = (0.12, 0.08, 0.04)
    rotation_amplitude_deg: tuple[float, float, float] = (5.0, 8.5, 12.5)
    gaze_xy_amplitude: tuple[float, float] = (0.28, 0.16)
    gaze_depth: float = 1.1
    micro_saccade_interval: float = 2.2
    micro_saccade_strength: float = 0.05
    micro_saccade_decay: float = 7.0
    focus_targets_world: tuple[tuple[float, float, float], ...] = ()
    focus_switch_interval: float = 1.8
    focus_jitter_std: float = 0.03
    focus_pull: float = 0.9
    head_focus_pull: float = 0.65
    head_parallel_angle_deg: float = -15.0
    head_parallel_angle_variation_deg: float = 5.0
    head_parallel_osc_speed: float = 0.9


@dataclass
class WebcamConfig:
    camera_index: int = 0


@dataclass
class RecordedConfig:
    path: str = "data/recorded_demo.csv"
    loop: bool = True
    speed: float = 1.0


@dataclass
class RenderConfig:
    target_fps: int = 60
    ray_length: float = 3.0
    head_axis_length: float = 0.15


@dataclass
class AppConfig:
    camera_intrinsics: CameraIntrinsics
    camera_pose_world: RigidTransform
    plane: PlaneConfig
    objects: list[SceneObjectConfig]
    filter: FilterConfig
    synthetic: SyntheticConfig
    webcam: WebcamConfig
    recorded: RecordedConfig
    render: RenderConfig
