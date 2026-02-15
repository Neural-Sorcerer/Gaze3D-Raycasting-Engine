from __future__ import annotations

import time

import numpy as np

from gaze3d_lab.core.scene import OBBObjectConfig, SceneObjectConfig, SyntheticConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector
from gaze3d_lab.core.types import GazeSample

from .datasource import DataSource


class SyntheticDataSource(DataSource):
    """Procedural pose/gaze trajectories for camera-free demos."""
    _MAX_HEAD_YAW_TO_GAZE_RAD = float(np.radians(10.0))
    _MAX_GAZE_HEAD_ANGLE_RAD = float(np.radians(30.0))

    def __init__(
        self,
        config: SyntheticConfig,
        camera_pose_world: RigidTransform | None = None,
        scene_objects: list[SceneObjectConfig] | None = None,
    ) -> None:
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
        if self._config.head_parallel_angle_variation_deg < 0.0:
            raise ValueError("SyntheticConfig.head_parallel_angle_variation_deg must be >= 0")
        if self._config.head_parallel_osc_speed <= 0.0:
            raise ValueError("SyntheticConfig.head_parallel_osc_speed must be > 0")
        self._focus_pull = float(np.clip(self._config.focus_pull, 0.0, 1.0))
        self._head_focus_pull = float(np.clip(self._config.head_focus_pull, 0.0, 1.0))

        self._camera_pose_world = camera_pose_world
        self._center_focus_target_world: np.ndarray | None = None
        if len(self._config.focus_targets_world) > 0:
            self._center_focus_target_world = np.asarray(
                self._config.focus_targets_world[len(self._config.focus_targets_world) // 2],
                dtype=np.float64,
            )

        self._center_tv_x_world: np.ndarray | None = None
        self._center_tv_normal_world: np.ndarray | None = None
        for obj in self._pick_center_tv_candidates(scene_objects):
            rotation_world = euler_xyz_to_matrix(*np.radians(obj.euler_deg))
            self._center_tv_x_world = normalize_vector(rotation_world[:, 0])
            self._center_tv_normal_world = normalize_vector(rotation_world[:, 1])
            self._center_focus_target_world = np.asarray(obj.center, dtype=np.float64)
            break

        self._rng = np.random.default_rng(self._config.seed)
        self._start_time = 0.0
        self._last_t = 0.0
        self._next_micro_saccade_t = self._config.micro_saccade_interval
        self._micro_saccade_xy = np.zeros(2, dtype=np.float64)
        self._focus_target_index = 0
        self._focus_direction = 1
        self._next_focus_switch_t = self._config.focus_switch_interval
        self._focus_offset = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _pick_center_tv_candidates(scene_objects: list[SceneObjectConfig] | None) -> list[OBBObjectConfig]:
        if scene_objects is None:
            return []

        ranked: list[tuple[int, OBBObjectConfig]] = []
        for obj in scene_objects:
            if not isinstance(obj, OBBObjectConfig):
                continue
            name = obj.name.lower()
            if "tv_center" not in name:
                continue
            rank = 0
            if "screen" in name:
                rank = 2
            elif "frame" in name:
                rank = 1
            ranked.append((rank, obj))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [obj for _, obj in ranked]

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
                0.86 + tz_amp * np.sin(0.43 * t) + 0.10 * tz_amp * np.cos(1.3 * t + 0.4),
            ],
            dtype=np.float64,
        )
        translation += self._rng.normal(0.0, noise_scale * 0.25, size=3)

        roll_amp, pitch_amp, yaw_amp = np.radians(np.asarray(self._config.rotation_amplitude_deg, dtype=np.float64))
        roll = roll_amp * np.sin(0.8 * t) + self._rng.normal(0.0, noise_scale * 0.08)
        pitch = pitch_amp * np.sin(0.45 * t + 0.4) + self._rng.normal(0.0, noise_scale * 0.08)
        yaw = yaw_amp * np.cos(0.5 * t) + self._rng.normal(0.0, noise_scale * 0.08)

        head_rotation = euler_xyz_to_matrix(roll, pitch, yaw)

        if self._camera_pose_world is not None and self._center_focus_target_world is not None:
            head_world_pos = self._camera_pose_world.apply_point(translation)
            to_center_world = normalize_vector(self._center_focus_target_world - head_world_pos)

            if self._center_tv_x_world is not None and self._center_tv_normal_world is not None:
                # Lock head frame to center TV: X/Y in plane, Z normal to plane.
                z_world = self._center_tv_normal_world.copy()
                if float(np.dot(z_world, to_center_world)) < 0.0:
                    z_world = -z_world

                x_base = self._center_tv_x_world - float(np.dot(self._center_tv_x_world, z_world)) * z_world
                if float(np.linalg.norm(x_base)) < 1e-8:
                    x_base = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), z_world)
                x_base = normalize_vector(x_base)
                y_base = normalize_vector(np.cross(z_world, x_base))
            else:
                # Fallback: use center target direction as normal and keep Y axis near world-up.
                z_world = to_center_world
                up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                y_base = up_world - float(np.dot(up_world, z_world)) * z_world
                if float(np.linalg.norm(y_base)) < 1e-8:
                    y_base = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                y_base = normalize_vector(y_base)
                x_base = normalize_vector(np.cross(y_base, z_world))

            # Keep head frame fixed to center-TV frame (X/Y parallel to TV, Z normal to TV).
            x_world = x_base
            y_world = y_base
            y_world = normalize_vector(np.cross(z_world, x_world))
            x_world = normalize_vector(np.cross(y_world, z_world))

            head_world_rotation = np.column_stack((x_world, y_world, z_world))
            head_rotation = self._camera_pose_world.rotation.T @ head_world_rotation

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

        gaze_head = normalize_vector(gaze_head)

        # Turn head around local Y-axis toward gaze direction, limited to +/-10 degrees.
        gaze_yaw = float(np.arctan2(gaze_head[0], gaze_head[2]))
        head_yaw_follow = float(np.clip(gaze_yaw, -self._MAX_HEAD_YAW_TO_GAZE_RAD, self._MAX_HEAD_YAW_TO_GAZE_RAD))
        if abs(head_yaw_follow) > 1e-9:
            rot_y = euler_xyz_to_matrix(0.0, head_yaw_follow, 0.0)
            head_rotation = head_rotation @ rot_y
            # Keep world gaze continuity while head yaw changes.
            gaze_head = normalize_vector(rot_y.T @ gaze_head)

        # Keep gaze within 30 degrees of head Z-axis.
        gaze_angle = float(np.arccos(np.clip(gaze_head[2], -1.0, 1.0)))
        if gaze_angle > self._MAX_GAZE_HEAD_ANGLE_RAD:
            xy = gaze_head[:2]
            xy_norm = float(np.linalg.norm(xy))
            if xy_norm < 1e-9:
                gaze_head = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                xy_dir = xy / xy_norm
                xy_mag = float(np.sin(self._MAX_GAZE_HEAD_ANGLE_RAD))
                gaze_head = np.array(
                    [xy_dir[0] * xy_mag, xy_dir[1] * xy_mag, float(np.cos(self._MAX_GAZE_HEAD_ANGLE_RAD))],
                    dtype=np.float64,
                )

        return GazeSample(
            timestamp=timestamp,
            head_transform_cam=RigidTransform(rotation=head_rotation, translation=translation),
            gaze_direction_head=normalize_vector(gaze_head),
        )
