from __future__ import annotations

import time

import numpy as np

from gaze3d_lab.core.scene import SyntheticConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector
from gaze3d_lab.core.types import GazeSample

from .datasource import DataSource


class SyntheticDataSource(DataSource):
    """Procedural pose/gaze trajectories for camera-free demos."""

    def __init__(self, config: SyntheticConfig, camera_pose_world: RigidTransform | None = None) -> None:
        self._config = config
        if self._config.gaze_depth <= 0.0:
            raise ValueError("SyntheticConfig.gaze_depth must be > 0")
        if self._config.micro_saccade_interval <= 0.0:
            raise ValueError("SyntheticConfig.micro_saccade_interval must be > 0")
        if self._config.micro_saccade_decay <= 0.0:
            raise ValueError("SyntheticConfig.micro_saccade_decay must be > 0")
        if self._config.focus_switch_interval <= 0.0:
            raise ValueError("SyntheticConfig.focus_switch_interval must be > 0")
        if self._config.focus_jitter_std < 0.0:
            raise ValueError("SyntheticConfig.focus_jitter_std must be >= 0")
        self._focus_pull = float(np.clip(self._config.focus_pull, 0.0, 1.0))

        self._camera_pose_world = camera_pose_world
        self._rng = np.random.default_rng(self._config.seed)
        self._start_time = 0.0
        self._last_t = 0.0
        self._next_micro_saccade_t = self._config.micro_saccade_interval
        self._micro_saccade_xy = np.zeros(2, dtype=np.float64)
        self._focus_target_index = 0
        self._focus_direction = 1
        self._next_focus_switch_t = self._config.focus_switch_interval
        self._focus_offset = np.zeros(3, dtype=np.float64)

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._last_t = 0.0
        self._next_micro_saccade_t = self._config.micro_saccade_interval
        self._micro_saccade_xy.fill(0.0)
        self._focus_target_index = 0
        self._focus_direction = 1
        self._next_focus_switch_t = self._config.focus_switch_interval
        self._focus_offset.fill(0.0)

    def next_sample(self, timestamp: float) -> GazeSample:
        t = (timestamp - self._start_time) * self._config.motion_scale
        dt = max(1e-6, t - self._last_t)
        self._last_t = t
        noise_scale = self._config.noise_std

        tx_amp, ty_amp, tz_amp = self._config.translation_amplitude
        translation = np.array(
            [
                tx_amp * np.sin(0.65 * t) + 0.04 * tx_amp * np.sin(1.9 * t + 0.2),
                -0.03 + ty_amp * np.cos(0.55 * t) + 0.06 * ty_amp * np.sin(1.1 * t),
                0.70 + tz_amp * np.sin(0.43 * t) + 0.10 * tz_amp * np.cos(1.3 * t + 0.4),
            ],
            dtype=np.float64,
        )
        translation += self._rng.normal(0.0, noise_scale * 0.25, size=3)

        roll_amp, pitch_amp, yaw_amp = np.radians(np.asarray(self._config.rotation_amplitude_deg, dtype=np.float64))
        roll = roll_amp * np.sin(0.8 * t) + self._rng.normal(0.0, noise_scale * 0.08)
        pitch = pitch_amp * np.sin(0.45 * t + 0.4) + self._rng.normal(0.0, noise_scale * 0.08)
        yaw = yaw_amp * np.cos(0.5 * t) + self._rng.normal(0.0, noise_scale * 0.08)

        head_rotation = euler_xyz_to_matrix(roll, pitch, yaw)

        if t >= self._next_micro_saccade_t:
            angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            self._micro_saccade_xy += self._config.micro_saccade_strength * direction
            jitter = float(self._rng.uniform(-0.35, 0.35))
            self._next_micro_saccade_t += max(0.2, self._config.micro_saccade_interval * (1.0 + jitter))

        decay = np.exp(-self._config.micro_saccade_decay * dt)
        self._micro_saccade_xy *= decay

        gx_amp, gy_amp = self._config.gaze_xy_amplitude
        gaze_head = np.array(
            [
                gx_amp * np.sin(1.2 * t) + 0.08 * np.sin(2.8 * t + 0.9),
                -0.16 + gy_amp * np.cos(0.95 * t) + 0.06 * np.sin(1.75 * t),
                self._config.gaze_depth + 0.06 * np.sin(0.5 * t),
            ],
            dtype=np.float64,
        )
        gaze_head[:2] += self._micro_saccade_xy
        gaze_head += self._rng.normal(0.0, noise_scale * 0.35, size=3)

        if self._camera_pose_world is not None and len(self._config.focus_targets_world) > 0:
            if t >= self._next_focus_switch_t:
                target_count = len(self._config.focus_targets_world)
                if target_count > 1:
                    if self._focus_target_index >= target_count - 1:
                        self._focus_direction = -1
                    elif self._focus_target_index <= 0:
                        self._focus_direction = 1
                    self._focus_target_index = int(
                        np.clip(self._focus_target_index + self._focus_direction, 0, target_count - 1)
                    )

                self._focus_offset = self._rng.normal(0.0, self._config.focus_jitter_std, size=3)
                self._focus_offset[1] *= 0.3
                self._next_focus_switch_t += self._config.focus_switch_interval

            target_world = (
                np.asarray(self._config.focus_targets_world[self._focus_target_index], dtype=np.float64) + self._focus_offset
            )
            head_world = self._camera_pose_world.compose(RigidTransform(rotation=head_rotation, translation=translation))
            focus_dir_world = normalize_vector(target_world - head_world.translation)
            focus_dir_head = normalize_vector(head_world.rotation.T @ focus_dir_world)

            gaze_head = normalize_vector((1.0 - self._focus_pull) * gaze_head + self._focus_pull * focus_dir_head)

        return GazeSample(
            timestamp=timestamp,
            head_transform_cam=RigidTransform(rotation=head_rotation, translation=translation),
            gaze_direction_head=normalize_vector(gaze_head),
        )
