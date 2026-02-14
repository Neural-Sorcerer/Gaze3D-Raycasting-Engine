from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class FrameProfiler:
    """Lightweight per-frame profiler with EMA-smoothed stats."""

    smoothing: float = 0.2
    _frame_start: float = 0.0
    _last_mark: float = 0.0
    _last_frame_end: float = 0.0
    _fps_ema: float = 0.0
    _stage_ema: dict[str, float] = field(default_factory=dict)

    def begin_frame(self, now: float | None = None) -> None:
        ts = now if now is not None else time.perf_counter()
        self._frame_start = ts
        self._last_mark = ts

    def mark(self, stage_name: str, now: float | None = None) -> None:
        ts = now if now is not None else time.perf_counter()
        dt = max(0.0, ts - self._last_mark)
        prev = self._stage_ema.get(stage_name, dt)
        self._stage_ema[stage_name] = self.smoothing * dt + (1.0 - self.smoothing) * prev
        self._last_mark = ts

    def end_frame(self, now: float | None = None) -> float:
        ts = now if now is not None else time.perf_counter()
        frame_dt = max(1e-6, ts - self._frame_start)
        inst_fps = 1.0 / frame_dt
        if self._fps_ema <= 0.0:
            self._fps_ema = inst_fps
        else:
            self._fps_ema = self.smoothing * inst_fps + (1.0 - self.smoothing) * self._fps_ema

        self._last_frame_end = ts
        return self._fps_ema

    def stage_snapshot_ms(self) -> dict[str, float]:
        return {name: dur * 1000.0 for name, dur in self._stage_ema.items()}

    @property
    def fps(self) -> float:
        return self._fps_ema
