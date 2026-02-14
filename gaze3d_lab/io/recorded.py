from __future__ import annotations

import bisect
import csv
import json
import time
from pathlib import Path

import numpy as np

from gaze3d_lab.core.scene import RecordedConfig
from gaze3d_lab.core.transforms import RigidTransform, euler_xyz_to_matrix, normalize_vector
from gaze3d_lab.core.types import GazeSample

from .datasource import DataSource


class RecordedDataSource(DataSource):
    """Playback for CSV/JSON trajectories with timestamps."""

    def __init__(self, config: RecordedConfig) -> None:
        self._config = config
        source_path = Path(self._config.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Recorded file not found: {source_path}")

        self._timestamps, self._samples = self._load_samples(source_path)
        if not self._samples:
            raise ValueError(f"No records found in {source_path}")

        self._start_wall_time = 0.0
        self._index = 0

    def start(self) -> None:
        self._start_wall_time = time.perf_counter()
        self._index = 0

    def next_sample(self, timestamp: float) -> GazeSample:
        elapsed = (timestamp - self._start_wall_time) * self._config.speed
        start_ts = self._timestamps[0]
        end_ts = self._timestamps[-1]
        duration = max(1e-6, end_ts - start_ts)

        if self._config.loop:
            target_ts = start_ts + (elapsed % duration)
            self._index = max(0, bisect.bisect_right(self._timestamps, target_ts) - 1)
        else:
            target_ts = min(end_ts, start_ts + elapsed)
            self._index = max(0, bisect.bisect_right(self._timestamps, target_ts) - 1)

        return self._samples[self._index]

    @staticmethod
    def _load_samples(path: Path) -> tuple[list[float], list[GazeSample]]:
        ext = path.suffix.lower()
        if ext == ".json":
            return RecordedDataSource._load_json(path)
        if ext in {".csv", ".txt"}:
            return RecordedDataSource._load_csv(path)
        raise ValueError(f"Unsupported recording extension: {ext}")

    @staticmethod
    def _load_json(path: Path) -> tuple[list[float], list[GazeSample]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON recording must be a list of records")

        records = [RecordedDataSource._record_to_sample(record) for record in payload]
        records.sort(key=lambda item: item[0])
        timestamps = [t for t, _ in records]
        samples = [s for _, s in records]
        return timestamps, samples

    @staticmethod
    def _load_csv(path: Path) -> tuple[list[float], list[GazeSample]]:
        records: list[tuple[float, GazeSample]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(RecordedDataSource._record_to_sample(row))

        records.sort(key=lambda item: item[0])
        timestamps = [t for t, _ in records]
        samples = [s for _, s in records]
        return timestamps, samples

    @staticmethod
    def _record_to_sample(record: dict[str, object]) -> tuple[float, GazeSample]:
        timestamp = float(record["timestamp"])

        tx = float(record["head_tx"])
        ty = float(record["head_ty"])
        tz = float(record["head_tz"])

        roll = float(record["head_roll"])
        pitch = float(record["head_pitch"])
        yaw = float(record["head_yaw"])

        gx = float(record["gaze_x"])
        gy = float(record["gaze_y"])
        gz = float(record["gaze_z"])

        sample = GazeSample(
            timestamp=timestamp,
            head_transform_cam=RigidTransform(
                rotation=euler_xyz_to_matrix(roll, pitch, yaw),
                translation=np.array([tx, ty, tz], dtype=np.float64),
            ),
            gaze_direction_head=normalize_vector(np.array([gx, gy, gz], dtype=np.float64)),
        )

        return timestamp, sample
