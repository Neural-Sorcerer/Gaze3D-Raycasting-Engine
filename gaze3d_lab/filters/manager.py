from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gaze3d_lab.core.scene import FilterConfig
from gaze3d_lab.core.transforms import normalize_vector

from .ema import EMAFilter
from .one_euro import OneEuroFilter

Vector3 = npt.NDArray[np.float64]


@dataclass
class PoseGazeSmoother:
    config: FilterConfig

    def __post_init__(self) -> None:
        self.mode = self.config.mode if self.config.mode in {"none", "ema", "one_euro"} else "one_euro"

        self._ema_position = EMAFilter(self.config.ema_alpha)
        self._ema_euler = EMAFilter(self.config.ema_alpha)
        self._ema_gaze = EMAFilter(self.config.ema_alpha)

        self._oe_position = OneEuroFilter(
            min_cutoff=self.config.one_euro_min_cutoff,
            beta=self.config.one_euro_beta,
            d_cutoff=self.config.one_euro_d_cutoff,
        )
        self._oe_euler = OneEuroFilter(
            min_cutoff=self.config.one_euro_min_cutoff,
            beta=self.config.one_euro_beta,
            d_cutoff=self.config.one_euro_d_cutoff,
        )
        self._oe_gaze = OneEuroFilter(
            min_cutoff=self.config.one_euro_min_cutoff,
            beta=self.config.one_euro_beta,
            d_cutoff=self.config.one_euro_d_cutoff,
        )

    def reset(self) -> None:
        for f in (
            self._ema_position,
            self._ema_euler,
            self._ema_gaze,
            self._oe_position,
            self._oe_euler,
            self._oe_gaze,
        ):
            f.reset()

    def cycle_mode(self) -> str:
        order = ["none", "ema", "one_euro"]
        idx = (order.index(self.mode) + 1) % len(order)
        self.mode = order[idx]
        self.config.mode = self.mode
        return self.mode

    def set_mode(self, mode: str) -> str:
        if mode in {"none", "ema", "one_euro"}:
            self.mode = mode
            self.config.mode = self.mode
        return self.mode

    def update(
        self,
        timestamp: float,
        head_position: Vector3,
        head_euler: Vector3,
        gaze_direction_world: Vector3,
    ) -> tuple[Vector3, Vector3, Vector3]:
        position = np.asarray(head_position, dtype=np.float64)
        euler = np.asarray(head_euler, dtype=np.float64)
        gaze = normalize_vector(gaze_direction_world)

        if self.mode == "none":
            return position, euler, gaze

        if self.mode == "ema":
            return (
                self._ema_position.update(position),
                self._ema_euler.update(euler),
                normalize_vector(self._ema_gaze.update(gaze)),
            )

        return (
            self._oe_position.update(timestamp, position),
            self._oe_euler.update(timestamp, euler),
            normalize_vector(self._oe_gaze.update(timestamp, gaze)),
        )

    def adjust_ema_alpha(self, delta: float) -> float:
        return self.set_ema_alpha(self._ema_position.alpha + delta)

    def set_ema_alpha(self, value: float) -> float:
        new_alpha = float(np.clip(value, 0.01, 1.0))
        for filt in (self._ema_position, self._ema_euler, self._ema_gaze):
            filt.set_alpha(new_alpha)
        self.config.ema_alpha = new_alpha
        return new_alpha

    def adjust_one_euro_beta(self, delta: float) -> float:
        return self.set_one_euro_beta(self.config.one_euro_beta + delta)

    def set_one_euro_beta(self, value: float) -> float:
        new_beta = max(0.0, float(value))
        self.config.one_euro_beta = new_beta
        for filt in (self._oe_position, self._oe_euler, self._oe_gaze):
            filt.set_params(beta=new_beta)
        return new_beta

    def adjust_one_euro_min_cutoff(self, delta: float) -> float:
        return self.set_one_euro_min_cutoff(self.config.one_euro_min_cutoff + delta)

    def set_one_euro_min_cutoff(self, value: float) -> float:
        new_cutoff = max(1e-4, float(value))
        self.config.one_euro_min_cutoff = new_cutoff
        for filt in (self._oe_position, self._oe_euler, self._oe_gaze):
            filt.set_params(min_cutoff=new_cutoff)
        return new_cutoff

    def describe(self) -> str:
        return (
            f"mode={self.mode} "
            f"ema_alpha={self.config.ema_alpha:.2f} "
            f"oe_min={self.config.one_euro_min_cutoff:.2f} "
            f"oe_beta={self.config.one_euro_beta:.3f}"
        )
