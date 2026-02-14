# Math and Frame Conventions

## Frames

- `W` (World): fixed global scene frame used by renderer.
- `C` (Camera): camera optical frame.
- `H` (Head): local frame attached to face center.

Transform notation:

- `T_AB` maps coordinates from frame `B` into frame `A`.
- `T_AB = [R_AB, t_AB]` where `R_AB` is 3x3, `t_AB` is 3x1.

Camera pose tip:

- Prefer configuring `T_WC` via a look-at target in YAML (`pose_world.look_at`) to avoid accidental reversed orientation from Euler angles.

## Pose and Gaze Conversion

Input sample:

- `T_CH`: head pose in camera frame.
- `g_H`: gaze direction in head frame.

Known calibration:

- `T_WC`: camera pose in world frame.

Compose:

- `T_WH = T_WC ∘ T_CH`

Then:

- Ray origin: `o_W = t_WH`
- Ray direction: `d_W = normalize(R_WH g_H)`

## Ray-Plane Intersection

Plane parameterization:

- point `p0`
- normal `n` (normalized)

Ray:

- `r(t) = o + t d`, `t >= 0`

Solve:

- `dot(r(t) - p0, n) = 0`
- `t = dot(p0 - o, n) / dot(d, n)`

Conditions:

- parallel if `abs(dot(d, n)) < eps`
- valid hit if `t >= 0`

## Ray-AABB Intersection (Slab Method)

For each axis `i`:

- compute entry/exit `t` values for min/max planes
- accumulate global interval `[t_min, t_max]`

No hit if:

- intervals do not overlap (`t_min > t_max`)
- box is behind ray (`t_max < 0`)

## Filtering

### EMA

`y_t = alpha * x_t + (1-alpha) * y_{t-1}`

### One Euro Filter

- smooth derivative first
- adapt cutoff with speed:
  - `cutoff = min_cutoff + beta * |dx_hat|`
- apply low-pass at adaptive cutoff

This gives low jitter at low speed and reduced lag at high speed.

## Practical Notes

- Keep vectors normalized for robust intersection math.
- Maintain transform composition order explicitly; frame mistakes are the most common source of bugs.
- Use smoothing on both head pose signals and gaze direction for stable visualization.
