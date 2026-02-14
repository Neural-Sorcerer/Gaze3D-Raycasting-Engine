from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from gaze3d_lab.core.calibration import CameraIntrinsics
from gaze3d_lab.core.scene import (
    AABBObjectConfig,
    AppConfig,
    FilterConfig,
    OBBObjectConfig,
    PlaneConfig,
    RecordedConfig,
    RenderConfig,
    SceneObjectConfig,
    SphereObjectConfig,
    SyntheticConfig,
    WebcamConfig,
)
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, look_at_rotation


def _vec3(value: Sequence[float], field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{field} must be a 3-vector, got {arr.shape}")
    return arr


def _color_rgba(raw: Sequence[float] | None, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if raw is None:
        return default
    values = tuple(float(x) for x in raw)
    if len(values) != 4:
        raise ValueError("Color must have 4 values (r,g,b,a)")
    return values  # type: ignore[return-value]


def _load_objects(objects_raw: list[dict[str, Any]]) -> list[SceneObjectConfig]:
    objects: list[SceneObjectConfig] = []
    for item in objects_raw:
        kind = str(item["type"]).lower()
        name = str(item.get("name", kind))

        if kind == "box":
            objects.append(
                AABBObjectConfig(
                    name=name,
                    min_corner=_vec3(item["min"], f"scene.objects[{name}].min"),
                    max_corner=_vec3(item["max"], f"scene.objects[{name}].max"),
                    color=_color_rgba(item.get("color"), (0.9, 0.4, 0.4, 0.35)),
                )
            )
        elif kind in {"obb", "oriented_box"}:
            objects.append(
                OBBObjectConfig(
                    name=name,
                    center=_vec3(item["center"], f"scene.objects[{name}].center"),
                    size=_vec3(item["size"], f"scene.objects[{name}].size"),
                    euler_deg=_vec3(item.get("euler_deg", [0.0, 0.0, 0.0]), f"scene.objects[{name}].euler_deg"),
                    color=_color_rgba(item.get("color"), (0.7, 0.7, 0.9, 0.35)),
                )
            )
        elif kind == "sphere":
            objects.append(
                SphereObjectConfig(
                    name=name,
                    center=_vec3(item["center"], f"scene.objects[{name}].center"),
                    radius=float(item["radius"]),
                    color=_color_rgba(item.get("color"), (0.3, 0.9, 0.5, 0.35)),
                )
            )
        else:
            raise ValueError(f"Unsupported object type: {kind}")

    return objects


def _vec2_tuple(value: Sequence[float], field: str) -> tuple[float, float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{field} must be a 2-vector, got {arr.shape}")
    return float(arr[0]), float(arr[1])


def _vec3_tuple(value: Sequence[float], field: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{field} must be a 3-vector, got {arr.shape}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def load_app_config(path: str | Path) -> AppConfig:
    path_obj = Path(path)
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")

    camera_raw = payload.get("camera", {})
    intr_raw = camera_raw.get("intrinsics", {})
    pose_raw = camera_raw.get("pose_world", {})

    intrinsics = CameraIntrinsics(
        fx=float(intr_raw.get("fx", 900.0)),
        fy=float(intr_raw.get("fy", 900.0)),
        cx=float(intr_raw.get("cx", 640.0)),
        cy=float(intr_raw.get("cy", 360.0)),
        width=int(intr_raw.get("width", 1280)),
        height=int(intr_raw.get("height", 720)),
    )

    translation = _vec3(pose_raw.get("translation", [0.0, -1.4, 1.0]), "camera.pose_world.translation")
    look_at_raw = pose_raw.get("look_at")
    if isinstance(look_at_raw, dict) and "target" in look_at_raw:
        target = _vec3(look_at_raw["target"], "camera.pose_world.look_at.target")
        up = _vec3(look_at_raw.get("up", [0.0, 0.0, 1.0]), "camera.pose_world.look_at.up")
        camera_rotation = look_at_rotation(eye=translation, target=target, up=up, camera_y_down=True)
    else:
        euler_deg = _vec3(pose_raw.get("euler_deg", [65.0, 0.0, 0.0]), "camera.pose_world.euler_deg")
        camera_rotation = euler_xyz_to_matrix(*np.radians(euler_deg))

    camera_pose_world = RigidTransform(
        rotation=camera_rotation,
        translation=translation,
    )

    scene_raw = payload.get("scene", {})
    plane_raw = scene_raw.get("plane", {})
    plane = PlaneConfig(
        point=_vec3(plane_raw.get("point", [0.0, 0.0, 0.0]), "scene.plane.point"),
        normal=_vec3(plane_raw.get("normal", [0.0, 0.0, 1.0]), "scene.plane.normal"),
        size=(float(plane_raw.get("size", [2.0, 2.0])[0]), float(plane_raw.get("size", [2.0, 2.0])[1])),
        color=_color_rgba(plane_raw.get("color"), (0.2, 0.8, 1.0, 0.18)),
    )

    objects = _load_objects(scene_raw.get("objects", []))

    filter_raw = payload.get("filters", {})
    one_euro_raw = filter_raw.get("one_euro", {})
    filter_cfg = FilterConfig(
        mode=str(filter_raw.get("mode", "one_euro")),
        ema_alpha=float(filter_raw.get("ema_alpha", 0.35)),
        one_euro_min_cutoff=float(one_euro_raw.get("min_cutoff", 1.2)),
        one_euro_beta=float(one_euro_raw.get("beta", 0.02)),
        one_euro_d_cutoff=float(one_euro_raw.get("d_cutoff", 1.0)),
    )

    ds_raw = payload.get("data_sources", {})
    synthetic_raw = ds_raw.get("synthetic", {})
    webcam_raw = ds_raw.get("webcam", {})
    recorded_raw = ds_raw.get("recorded", {})

    synthetic_cfg = SyntheticConfig(
        seed=int(synthetic_raw.get("seed", 7)),
        noise_std=float(synthetic_raw.get("noise_std", 0.02)),
        motion_scale=float(synthetic_raw.get("motion_scale", 1.0)),
        translation_amplitude=_vec3_tuple(
            synthetic_raw.get("translation_amplitude", [0.12, 0.08, 0.04]),
            "data_sources.synthetic.translation_amplitude",
        ),
        rotation_amplitude_deg=_vec3_tuple(
            synthetic_raw.get("rotation_amplitude_deg", [5.0, 8.5, 12.5]),
            "data_sources.synthetic.rotation_amplitude_deg",
        ),
        gaze_xy_amplitude=_vec2_tuple(
            synthetic_raw.get("gaze_xy_amplitude", [0.28, 0.16]),
            "data_sources.synthetic.gaze_xy_amplitude",
        ),
        gaze_depth=float(synthetic_raw.get("gaze_depth", 1.1)),
        micro_saccade_interval=float(synthetic_raw.get("micro_saccade_interval", 2.2)),
        micro_saccade_strength=float(synthetic_raw.get("micro_saccade_strength", 0.05)),
        micro_saccade_decay=float(synthetic_raw.get("micro_saccade_decay", 7.0)),
        focus_targets_world=tuple(
            _vec3_tuple(item, f"data_sources.synthetic.focus_targets_world[{idx}]")
            for idx, item in enumerate(synthetic_raw.get("focus_targets_world", []))
        ),
        focus_switch_interval=float(synthetic_raw.get("focus_switch_interval", 1.8)),
        focus_jitter_std=float(synthetic_raw.get("focus_jitter_std", 0.03)),
        focus_pull=float(synthetic_raw.get("focus_pull", 0.9)),
        head_focus_pull=float(synthetic_raw.get("head_focus_pull", 0.65)),
        head_parallel_angle_deg=float(synthetic_raw.get("head_parallel_angle_deg", -15.0)),
        head_parallel_angle_variation_deg=float(synthetic_raw.get("head_parallel_angle_variation_deg", 5.0)),
        head_parallel_osc_speed=float(synthetic_raw.get("head_parallel_osc_speed", 0.9)),
    )
    webcam_cfg = WebcamConfig(camera_index=int(webcam_raw.get("camera_index", 0)))
    recorded_cfg = RecordedConfig(
        path=str(recorded_raw.get("path", "data/recorded_demo.csv")),
        loop=bool(recorded_raw.get("loop", True)),
        speed=float(recorded_raw.get("speed", 1.0)),
    )

    render_raw = payload.get("render", {})
    render_cfg = RenderConfig(
        target_fps=int(render_raw.get("target_fps", 60)),
        ray_length=float(render_raw.get("ray_length", 3.0)),
        head_axis_length=float(render_raw.get("head_axis_length", 0.15)),
    )

    return AppConfig(
        camera_intrinsics=intrinsics,
        camera_pose_world=camera_pose_world,
        plane=plane,
        objects=objects,
        filter=filter_cfg,
        synthetic=synthetic_cfg,
        webcam=webcam_cfg,
        recorded=recorded_cfg,
        render=render_cfg,
    )
