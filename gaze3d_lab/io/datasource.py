from __future__ import annotations

from abc import ABC, abstractmethod

from gaze3d_lab.core.types import GazeSample


class DataSource(ABC):
    """Abstract streaming source for gaze + head pose samples."""

    def start(self) -> None:
        return None

    @abstractmethod
    def next_sample(self, timestamp: float) -> GazeSample:
        raise NotImplementedError

    def close(self) -> None:
        return None
