# gaze3d-lab: Real-time 3D Gaze Ray + Scene Intersection Visualizer

`gaze3d-lab` is a production-style Python project for real-time 3D gaze visualization. It renders world/camera/head frames, gaze rays, and point-of-gaze intersections against a plane plus simple scene meshes (AABBs and spheres), with pluggable data sources and runtime smoothing controls.

## Features

- Real-time OpenGL viewer using `pyqtgraph.opengl`.
- Coordinate frames: world, camera frustum, head frame, face center.
- Geometry: ray-plane, ray-AABB, ray-sphere intersections.
- Frame transforms: camera frame -> world frame conversion for pose and rays.
- Data source plugins: `synthetic`, `webcam` (mock estimator placeholder), `recorded` (CSV/JSON).
- Extended synthetic trajectories include multi-frequency motion + micro-saccade bursts for filter stress testing.
- Advanced default scene with 3 TV targets and off-center decorative spheres/cubes.
- Left/right TVs are angled inward (30 deg) with OBB geometry support for realistic layout tests.
- Smoothing: `EMA` and `OneEuroFilter`, tunable at runtime via keyboard.
- Lightweight performance profiling: FPS + per-stage timings.
- Clean modular architecture and pytest coverage for core geometry math.

## Demo GIF Instructions

1. Run synthetic mode (see Quick Start).
2. Record a short clip with your preferred recorder (for Ubuntu, `peek` works well).
3. Convert MP4 to GIF with ffmpeg:

   ```bash
   ffmpeg -i demo.mp4 -vf "fps=20,scale=1280:-1:flags=lanczos" -loop 0 docs/demo.gif
   ```

4. Add to README:

   ```markdown
   ![gaze3d-lab demo](docs/demo.gif)
   ```

## Architecture

```text
gaze3d_lab/
  app/            # runtime orchestration + update loop
  core/           # shared models, transforms, calibration, profiler
  geometry/       # intersections and geometric primitives
  filters/        # EMA + OneEuro + smoothing manager
  io/             # YAML config and data source plugins
  viz/            # pyqtgraph OpenGL scene/view controls
config/           # default app configuration
docs/             # math/frame notes
tests/            # unit tests for geometry/transforms
```

### Runtime Flow

```text
DataSource -> sample(head_pose_cam, gaze_head)
          -> camera_to_world transform
          -> smoother (EMA / OneEuro / none)
          -> intersections (plane + scene objects)
          -> viewer update (ray, frames, hits)
```

## Coordinate Frames and Math

- `W` (world): global rendering frame.
- `C` (camera): camera optical frame.
- `H` (head): local face/head frame.

Given:

- `T_WC`: camera pose in world.
- `T_CH`: head pose in camera.
- `g_H`: gaze direction in head frame.

Compute:

- `T_WH = T_WC ∘ T_CH`
- `o_W = translation(T_WH)` (ray origin at face center)
- `d_W = normalize(R_WH * g_H)` (ray direction)

Intersections:

- Ray-plane: `t = dot(p0 - o, n) / dot(d, n)` with `t >= 0`.
- Ray-AABB: slab test in x/y/z.

More detail: `docs/math_and_frames.md`.

Camera pose in YAML supports either:

- `pose_world.euler_deg` (legacy Euler XYZ)
- `pose_world.look_at` (recommended, avoids direction mistakes)

## Keyboard Controls

- `1`: toggle world axes/grid
- `2`: toggle camera frustum/axes
- `3`: toggle head frame
- `4`: toggle face center point
- `5`: toggle gaze ray
- `6`: toggle plane hit point
- `7`: toggle object hit point
- `8`: toggle objects
- `9`: toggle plane outline
- `0`: toggle gaze-ray clipping at first object hit (`on` by default)
- `F`: cycle filter mode (`none -> ema -> one_euro`)
- `[` / `]`: EMA alpha down/up
- `-` / `=`: OneEuro beta down/up
- `,` / `.`: OneEuro min cutoff down/up
- `H`: print control summary

## Install

Python 3.10+ on Ubuntu 20.04+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python -m gaze3d_lab --mode synthetic --config config/default.yaml
python -m gaze3d_lab --mode synthetic --config config/synthetic_stress.yaml
python -m gaze3d_lab --mode webcam --config config/default.yaml
python -m gaze3d_lab --mode recorded --config config/default.yaml
```

## Tests

```bash
pytest -q
```

## How To Add a Real AI Model Later

1. Keep `DataSource` interface unchanged (`next_sample(timestamp) -> GazeSample`).
2. Replace internals of `WebcamDataSource.next_sample()` with:
   - face detection / tracking
   - head pose estimator
   - gaze estimator
3. Output:
   - `head_transform_cam` (`R_CH`, `t_CH`)
   - `gaze_direction_head` vector
4. Existing smoothing, geometry, and rendering pipeline continues to work unchanged.
