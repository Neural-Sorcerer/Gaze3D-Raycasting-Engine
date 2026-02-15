from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from gaze3d_lab.core.scene import AABBObjectConfig, AppConfig, OBBObjectConfig, SphereObjectConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector
from gaze3d_lab.geometry.intersections import IntersectionHit

try:  # pragma: no cover - UI imports not exercised by unit tests.
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph.opengl as gl
except ImportError as exc:  # pragma: no cover
    QtCore = None
    QtWidgets = None
    gl = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

Vector3 = npt.NDArray[np.float64]
ControlCallback = Callable[[str], None]


def _segments_to_line_pos(segments: list[tuple[Vector3, Vector3]]) -> npt.NDArray[np.float64]:
    pos = np.empty((len(segments) * 2, 3), dtype=np.float64)
    for i, (a, b) in enumerate(segments):
        pos[2 * i] = a
        pos[2 * i + 1] = b
    return pos


def _plane_basis(normal: Vector3) -> tuple[Vector3, Vector3]:
    n = normalize_vector(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = normalize_vector(np.cross(n, helper))
    bitangent = normalize_vector(np.cross(n, tangent))
    return tangent, bitangent


def _build_box_mesh(min_corner: Vector3, max_corner: Vector3) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    min_x, min_y, min_z = min_corner
    max_x, max_y, max_z = max_corner

    vertices = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )

    return vertices, faces


def _build_sphere_mesh(
    center: Vector3,
    radius: float,
    lat_steps: int = 12,
    lon_steps: int = 20,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    vertices: list[list[float]] = []
    for i in range(lat_steps + 1):
        phi = np.pi * i / lat_steps
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for j in range(lon_steps + 1):
            theta = 2.0 * np.pi * j / lon_steps
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            x = center[0] + radius * sin_phi * cos_theta
            y = center[1] + radius * sin_phi * sin_theta
            z = center[2] + radius * cos_phi
            vertices.append([x, y, z])

    faces: list[list[int]] = []
    ring = lon_steps + 1
    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * ring + j
            b = a + ring
            faces.append([a, b, a + 1])
            faces.append([a + 1, b, b + 1])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _build_obb_mesh(
    center: Vector3,
    size: Vector3,
    euler_deg: Vector3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    half = size * 0.5
    vertices_local, faces = _build_box_mesh(-half, half)
    rotation = euler_xyz_to_matrix(*np.radians(euler_deg))
    vertices_world = (rotation @ vertices_local.T).T + center
    return vertices_world, faces


if gl is not None:

    class _InteractiveGLView(gl.GLViewWidget):  # type: ignore[misc]
        def __init__(self, key_handler: Callable[[int], bool]) -> None:
            super().__init__()
            self._key_handler = key_handler

        def keyPressEvent(self, event) -> None:  # type: ignore[override]
            if self._key_handler(int(event.key())):
                event.accept()
                return
            super().keyPressEvent(event)

else:

    class _InteractiveGLView:  # pragma: no cover - only used when GUI deps are missing.
        def __init__(self, key_handler: Callable[[int], bool]) -> None:
            raise RuntimeError("OpenGL view unavailable because GUI dependencies are missing")


class SceneViewer:
    """OpenGL scene for gaze ray visualization."""

    def __init__(
        self,
        config: AppConfig,
        mode: str,
        control_callback: ControlCallback | None = None,
        face_mesh_points_head: npt.ArrayLike | None = None,
    ) -> None:
        if gl is None or QtWidgets is None or QtCore is None:
            raise RuntimeError(
                "PyQtGraph OpenGL dependencies are missing. Install requirements.txt first."
            ) from _IMPORT_ERROR

        self._config = config
        self._mode = mode
        self._control_callback = control_callback
        if face_mesh_points_head is None:
            self._face_mesh_points_head = np.empty((0, 3), dtype=np.float64)
        else:
            points = np.asarray(face_mesh_points_head, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                self._face_mesh_points_head = np.empty((0, 3), dtype=np.float64)
            else:
                self._face_mesh_points_head = points.copy()

        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.view = _InteractiveGLView(self._on_key_press)
        self.view.setWindowTitle("gaze3d-lab")
        self.view.resize(1280, 800)
        self.view.setBackgroundColor((10, 16, 25))
        self.view.setCameraPosition(distance=9.0, elevation=20.0, azimuth=-40.0)
        self._smooth_orbit_enabled = False
        self._smooth_orbit_speed_deg_per_sec = 18.0
        self._smooth_orbit_last_ts: float | None = None
        self._smooth_orbit_timer = QtCore.QTimer(self.view)
        self._smooth_orbit_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._smooth_orbit_timer.setInterval(16)
        self._smooth_orbit_timer.timeout.connect(self._tick_smooth_orbit)

        self._empty_pos = np.empty((0, 3), dtype=np.float64)
        self._head_axis_x = np.zeros((2, 3), dtype=np.float64)
        self._head_axis_y = np.zeros((2, 3), dtype=np.float64)
        self._head_axis_z = np.zeros((2, 3), dtype=np.float64)
        self._gaze_line = np.zeros((2, 3), dtype=np.float64)
        self._face_pos = np.zeros((1, 3), dtype=np.float64)
        self._face_mesh_pos = np.zeros_like(self._face_mesh_points_head)
        self._plane_hit_pos = np.zeros((1, 3), dtype=np.float64)
        self._object_hit_pos = np.zeros((1, 3), dtype=np.float64)

        self._groups: dict[str, list] = {}
        self._visibility: dict[str, bool] = {}

        self._build_static_scene()
        self._build_dynamic_items()
        self.view.show()

    @property
    def qtcore(self):
        return QtCore

    def exec(self) -> int:
        return int(self.qt_app.exec_())

    def close(self) -> None:
        self._smooth_orbit_timer.stop()
        self.view.close()

    def _add_group(self, name: str, items: list) -> None:
        self._groups[name] = items
        self._visibility[name] = True

    def _set_group_visible(self, name: str, visible: bool) -> None:
        for item in self._groups.get(name, []):
            item.setVisible(visible)
        self._visibility[name] = visible

    def _toggle_group(self, name: str) -> None:
        self._set_group_visible(name, not self._visibility.get(name, True))

    def _build_static_scene(self) -> None:
        assert gl is not None
        world_axes = gl.GLAxisItem()
        world_axes.setSize(0.6, 0.6, 0.6)
        self.view.addItem(world_axes)

        grid = gl.GLGridItem()
        if hasattr(grid, "setColor"):
            grid.setColor((70, 85, 110, 120))
        grid.scale(0.2, 0.2, 1.0)
        self.view.addItem(grid)
        self._add_group("world_axes", [world_axes, grid])

        camera_axis_items = self._build_camera_axes_items()
        frustum_item = self._build_camera_frustum_item()
        self._add_group("camera_frustum", [frustum_item, *camera_axis_items])

        plane_item = self._build_plane_outline_item()
        self._add_group("plane", [plane_item])

        object_items = self._build_scene_object_items()
        self._add_group("objects", object_items)

    def _build_camera_axes_items(self) -> list:
        assert gl is not None
        cam = self._config.camera_pose_world
        origin = cam.translation
        # Keep camera gizmo compact in dense scenes.
        length = 0.083

        colors = [
            (1.0, 0.1, 0.1, 1.0),
            (0.1, 1.0, 0.1, 1.0),
            (0.1, 0.5, 1.0, 1.0),
        ]
        items = []
        for axis_idx, color in enumerate(colors):
            tip = origin + cam.rotation[:, axis_idx] * length
            pos = np.vstack([origin, tip])
            item = gl.GLLinePlotItem(pos=pos, color=color, width=2.0, mode="lines", antialias=True)
            self.view.addItem(item)
            items.append(item)
        return items

    def _build_camera_frustum_item(self):
        assert gl is not None
        # Scale frustum display down to avoid overpowering nearby scene objects.
        near, far = self._config.camera_intrinsics.frustum_corners(0.027, 0.15)
        cam = self._config.camera_pose_world

        near_w = np.asarray([cam.apply_point(p) for p in near], dtype=np.float64)
        far_w = np.asarray([cam.apply_point(p) for p in far], dtype=np.float64)
        origin = cam.translation

        segments: list[tuple[Vector3, Vector3]] = []
        for i in range(4):
            segments.append((near_w[i], near_w[(i + 1) % 4]))
            segments.append((far_w[i], far_w[(i + 1) % 4]))
            segments.append((near_w[i], far_w[i]))
            segments.append((origin, near_w[i]))

        pos = _segments_to_line_pos(segments)
        item = gl.GLLinePlotItem(
            pos=pos,
            color=(0.8, 0.85, 0.95, 0.65),
            width=1.0,
            mode="lines",
            antialias=True,
        )
        self.view.addItem(item)
        return item

    def _build_plane_outline_item(self):
        assert gl is not None
        plane = self._config.plane
        tangent, bitangent = _plane_basis(plane.normal)
        hx = plane.size[0] * 0.5
        hy = plane.size[1] * 0.5
        p = plane.point

        c0 = p - hx * tangent - hy * bitangent
        c1 = p + hx * tangent - hy * bitangent
        c2 = p + hx * tangent + hy * bitangent
        c3 = p - hx * tangent + hy * bitangent

        segments = [(c0, c1), (c1, c2), (c2, c3), (c3, c0), (c0, c2), (c1, c3)]
        pos = _segments_to_line_pos(segments)

        item = gl.GLLinePlotItem(
            pos=pos,
            color=plane.color,
            width=1.0,
            mode="lines",
            antialias=True,
        )
        self.view.addItem(item)
        return item

    def _build_scene_object_items(self) -> list:
        assert gl is not None
        items = []
        for obj in self._config.objects:
            if isinstance(obj, AABBObjectConfig):
                verts, faces = _build_box_mesh(obj.min_corner, obj.max_corner)
            elif isinstance(obj, OBBObjectConfig):
                verts, faces = _build_obb_mesh(obj.center, obj.size, obj.euler_deg)
            elif isinstance(obj, SphereObjectConfig):
                verts, faces = _build_sphere_mesh(obj.center, obj.radius)
            else:
                continue

            mesh = gl.GLMeshItem(
                vertexes=verts,
                faces=faces,
                color=obj.color,
                smooth=False,
                drawEdges=True,
                drawFaces=True,
                shader="shaded",
            )
            self.view.addItem(mesh)
            items.append(mesh)
        return items

    def _build_dynamic_items(self) -> None:
        assert gl is not None
        self._head_x_item = gl.GLLinePlotItem(
            pos=self._head_axis_x,
            color=(1.0, 0.2, 0.2, 1.0),
            width=2.2,
            mode="lines",
            antialias=True,
        )
        self._head_y_item = gl.GLLinePlotItem(
            pos=self._head_axis_y,
            color=(0.2, 1.0, 0.2, 1.0),
            width=2.2,
            mode="lines",
            antialias=True,
        )
        self._head_z_item = gl.GLLinePlotItem(
            pos=self._head_axis_z,
            color=(0.2, 0.5, 1.0, 1.0),
            width=2.2,
            mode="lines",
            antialias=True,
        )

        self._gaze_item = gl.GLLinePlotItem(
            pos=self._gaze_line,
            color=(1.0, 0.95, 0.1, 1.0),
            width=2.0,
            mode="lines",
            antialias=True,
        )

        self._face_item = gl.GLScatterPlotItem(pos=self._face_pos, color=(1.0, 0.6, 0.2, 1.0), size=10)
        self._face_mesh_item = None
        if self._face_mesh_points_head.shape[0] > 0:
            self._face_mesh_item = gl.GLScatterPlotItem(
                pos=self._face_mesh_pos,
                color=(1.0, 1.0, 1.0, 1.0),
                size=1.0,
            )
        self._plane_hit_item = gl.GLScatterPlotItem(pos=self._empty_pos, color=(0.0, 1.0, 1.0, 1.0), size=12)
        self._object_hit_item = gl.GLScatterPlotItem(pos=self._empty_pos, color=(1.0, 0.2, 1.0, 1.0), size=12)

        for item in (
            self._head_x_item,
            self._head_y_item,
            self._head_z_item,
            self._gaze_item,
            self._face_item,
            self._plane_hit_item,
            self._object_hit_item,
        ):
            self.view.addItem(item)
        if self._face_mesh_item is not None:
            self.view.addItem(self._face_mesh_item)

        self._add_group("head_frame", [self._head_x_item, self._head_y_item, self._head_z_item])
        face_group_items = [self._face_item]
        if self._face_mesh_item is not None:
            face_group_items.append(self._face_mesh_item)
        self._add_group("face_point", face_group_items)
        self._add_group("gaze_ray", [self._gaze_item])
        self._add_group("plane_hit", [self._plane_hit_item])
        self._add_group("object_hit", [self._object_hit_item])

    def _on_key_press(self, key: int) -> bool:
        if QtCore is None:
            return False

        if key == QtCore.Qt.Key_R:
            self._toggle_smooth_orbit()
            return True

        toggles = {
            QtCore.Qt.Key_1: "world_axes",
            QtCore.Qt.Key_2: "camera_frustum",
            QtCore.Qt.Key_3: "head_frame",
            QtCore.Qt.Key_4: "face_point",
            QtCore.Qt.Key_5: "gaze_ray",
            QtCore.Qt.Key_6: "plane_hit",
            QtCore.Qt.Key_7: "object_hit",
            QtCore.Qt.Key_8: "objects",
            QtCore.Qt.Key_9: "plane",
        }
        if key in toggles:
            self._toggle_group(toggles[key])
            return True

        control_actions = {
            QtCore.Qt.Key_F: "cycle_filter",
            QtCore.Qt.Key_0: "toggle_ray_clip",
            QtCore.Qt.Key_BracketLeft: "ema_down",
            QtCore.Qt.Key_BracketRight: "ema_up",
            QtCore.Qt.Key_Minus: "beta_down",
            QtCore.Qt.Key_Equal: "beta_up",
            QtCore.Qt.Key_Comma: "min_cutoff_down",
            QtCore.Qt.Key_Period: "min_cutoff_up",
        }
        action = control_actions.get(key)
        if action is not None:
            if self._control_callback is not None:
                self._control_callback(action)
            return True

        if key == QtCore.Qt.Key_H:
            self.print_controls()
            return True

        return False

    def _toggle_smooth_orbit(self) -> None:
        self._smooth_orbit_enabled = not self._smooth_orbit_enabled
        if self._smooth_orbit_enabled:
            self._smooth_orbit_last_ts = time.perf_counter()
            self._smooth_orbit_timer.start()
        else:
            self._smooth_orbit_timer.stop()
            self._smooth_orbit_last_ts = None

    def _tick_smooth_orbit(self) -> None:
        if not self._smooth_orbit_enabled:
            return
        now = time.perf_counter()
        if self._smooth_orbit_last_ts is None:
            self._smooth_orbit_last_ts = now
            return
        dt = max(0.0, min(0.1, now - self._smooth_orbit_last_ts))
        self._smooth_orbit_last_ts = now
        self.view.orbit(self._smooth_orbit_speed_deg_per_sec * dt, 0.0)

    def print_controls(self) -> None:
        print(
            "Controls: 1 world, 2 frustum, 3 head, 4 eye-center/face, 5 gaze, 6 plane hit, "
            "7 object hit, 8 objects, 9 plane, 0 ray-clip, R smooth-orbit, F filter, "
            "[/] ema, -/= beta, ,/. min cutoff"
        )

    def update_dynamic(
        self,
        head_transform_world: RigidTransform,
        head_origin_world: Vector3,
        gaze_origin_world: Vector3,
        gaze_direction_world: Vector3,
        ray_length: float,
        head_axis_length: float,
        plane_hit: IntersectionHit | None,
        object_hit: IntersectionHit | None,
    ) -> None:
        origin = head_origin_world
        rot = head_transform_world.rotation

        self._head_axis_x[0] = origin
        self._head_axis_x[1] = origin + rot[:, 0] * head_axis_length
        self._head_axis_y[0] = origin
        self._head_axis_y[1] = origin + rot[:, 1] * head_axis_length
        self._head_axis_z[0] = origin
        self._head_axis_z[1] = origin + rot[:, 2] * head_axis_length

        self._head_x_item.setData(pos=self._head_axis_x)
        self._head_y_item.setData(pos=self._head_axis_y)
        self._head_z_item.setData(pos=self._head_axis_z)

        gaze = normalize_vector(gaze_direction_world)
        self._gaze_line[0] = gaze_origin_world
        self._gaze_line[1] = gaze_origin_world + gaze * ray_length
        self._gaze_item.setData(pos=self._gaze_line)

        self._face_pos[0] = gaze_origin_world
        self._face_item.setData(pos=self._face_pos)
        if self._face_mesh_item is not None:
            self._face_mesh_pos[:] = (rot @ self._face_mesh_points_head.T).T + head_origin_world
            self._face_mesh_item.setData(pos=self._face_mesh_pos)

        if plane_hit is None:
            self._plane_hit_item.setData(pos=self._empty_pos)
        else:
            self._plane_hit_pos[0] = plane_hit.point
            self._plane_hit_item.setData(pos=self._plane_hit_pos)

        if object_hit is None:
            self._object_hit_item.setData(pos=self._empty_pos)
        else:
            self._object_hit_pos[0] = object_hit.point
            self._object_hit_item.setData(pos=self._object_hit_pos)

    def update_status(self, fps: float, stage_ms: dict[str, float], filter_info: str) -> None:
        source_ms = stage_ms.get("source", 0.0)
        filter_ms = stage_ms.get("filters", 0.0)
        geom_ms = stage_ms.get("intersections", 0.0)
        render_ms = stage_ms.get("render", 0.0)

        self.view.setWindowTitle(
            f"gaze3d-lab | mode={self._mode} | fps={fps:5.1f} | "
            f"src={source_ms:5.2f}ms filt={filter_ms:5.2f}ms geom={geom_ms:5.2f}ms render={render_ms:5.2f}ms | "
            f"{filter_info}"
        )
