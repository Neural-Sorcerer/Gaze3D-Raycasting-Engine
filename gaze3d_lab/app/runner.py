from __future__ import annotations

import time
from pathlib import Path

from gaze3d_lab.core.profiler import FrameProfiler
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, matrix_to_euler_xyz, normalize_vector
from gaze3d_lab.filters.manager import PoseGazeSmoother
from gaze3d_lab.geometry.intersections import intersect_scene_objects, ray_plane_intersection
from gaze3d_lab.io.config_loader import load_app_config
from gaze3d_lab.io.datasource import DataSource
from gaze3d_lab.io.recorded import RecordedDataSource
from gaze3d_lab.io.synthetic import SyntheticDataSource
from gaze3d_lab.viz.viewer import SceneViewer


def _build_data_source(mode: str, config) -> DataSource:
    if mode == "synthetic":
        return SyntheticDataSource(config.synthetic, camera_pose_world=config.camera_pose_world)
    if mode == "webcam":
        # Keep OpenCV import isolated to webcam mode only.
        from gaze3d_lab.io.webcam import WebcamDataSource

        return WebcamDataSource(config.webcam)
    if mode == "recorded":
        return RecordedDataSource(config.recorded)
    raise ValueError(f"Unsupported mode: {mode}")


def run_app(mode: str, config_path: str | Path) -> int:
    cfg = load_app_config(config_path)
    source = _build_data_source(mode, cfg)
    smoother = PoseGazeSmoother(cfg.filter)
    profiler = FrameProfiler()

    def handle_control(action: str) -> None:
        if action == "cycle_filter":
            smoother.cycle_mode()
        elif action == "ema_down":
            smoother.adjust_ema_alpha(-0.02)
        elif action == "ema_up":
            smoother.adjust_ema_alpha(0.02)
        elif action == "beta_down":
            smoother.adjust_one_euro_beta(-0.002)
        elif action == "beta_up":
            smoother.adjust_one_euro_beta(0.002)
        elif action == "min_cutoff_down":
            smoother.adjust_one_euro_min_cutoff(-0.1)
        elif action == "min_cutoff_up":
            smoother.adjust_one_euro_min_cutoff(0.1)

    viewer = SceneViewer(cfg, mode=mode, control_callback=handle_control)

    source.start()
    viewer.qt_app.aboutToQuit.connect(source.close)

    qtcore = viewer.qtcore
    timer = qtcore.QTimer()
    timer.setTimerType(qtcore.Qt.PreciseTimer)
    timer.setInterval(max(1, int(1000 / max(1, cfg.render.target_fps))))

    frame_idx = 0

    def update_frame() -> None:
        nonlocal frame_idx

        try:
            now = time.perf_counter()
            profiler.begin_frame(now)

            sample = source.next_sample(now)
            profiler.mark("source")

            head_world_raw = cfg.camera_pose_world.compose(sample.head_transform_cam)
            face_world_raw = head_world_raw.translation
            head_euler_raw = matrix_to_euler_xyz(head_world_raw.rotation)
            gaze_world_raw = normalize_vector(head_world_raw.apply_vector(sample.gaze_direction_head))

            face_world, head_euler, gaze_world = smoother.update(
                sample.timestamp,
                face_world_raw,
                head_euler_raw,
                gaze_world_raw,
            )

            head_world = RigidTransform(rotation=euler_xyz_to_matrix(*head_euler), translation=face_world)
            profiler.mark("filters")

            plane_hit = ray_plane_intersection(face_world, gaze_world, cfg.plane.point, cfg.plane.normal)
            object_hit = intersect_scene_objects(face_world, gaze_world, cfg.objects)
            profiler.mark("intersections")

            viewer.update_dynamic(
                head_transform_world=head_world,
                face_center_world=face_world,
                gaze_direction_world=gaze_world,
                ray_length=cfg.render.ray_length,
                head_axis_length=cfg.render.head_axis_length,
                plane_hit=plane_hit,
                object_hit=object_hit,
            )
            profiler.mark("render")
            fps = profiler.end_frame()

            frame_idx += 1
            if frame_idx % 8 == 0:
                viewer.update_status(fps=fps, stage_ms=profiler.stage_snapshot_ms(), filter_info=smoother.describe())
        except Exception as exc:
            timer.stop()
            source.close()
            print(f"[gaze3d-lab] runtime error: {exc}")
            viewer.close()

    timer.timeout.connect(update_frame)
    timer.start()

    return viewer.exec()
