from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

VectorN = npt.NDArray[np.float64]


def _alpha(cutoff: float, dt: float) -> float:
    cutoff_clamped = max(1e-4, float(cutoff))
    dt_clamped = max(1e-6, float(dt))
    tau = 1.0 / (2.0 * np.pi * cutoff_clamped)
    return 1.0 / (1.0 + tau / dt_clamped)


@dataclass
class OneEuroFilter:
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0
    _prev_time: float | None = None
    _x_prev: VectorN | None = None
    _x_hat: VectorN | None = None
    _dx_hat: VectorN | None = None

    def reset(self) -> None:
        self._prev_time = None
        self._x_prev = None
        self._x_hat = None
        self._dx_hat = None

    def set_params(
        self,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        if min_cutoff is not None:
            self.min_cutoff = max(1e-4, float(min_cutoff))
        if beta is not None:
            self.beta = max(0.0, float(beta))
        if d_cutoff is not None:
            self.d_cutoff = max(1e-4, float(d_cutoff))

    def update(self, timestamp: float, value: Sequence[float] | VectorN) -> VectorN:
        x = np.asarray(value, dtype=np.float64)

        if self._prev_time is None or self._x_prev is None:
            self._prev_time = float(timestamp)
            self._x_prev = x.copy()
            self._x_hat = x.copy()
            self._dx_hat = np.zeros_like(x)
            return x.copy()

        dt = max(1e-6, float(timestamp - self._prev_time))
        dx = (x - self._x_prev) / dt

        alpha_d = _alpha(self.d_cutoff, dt)
        assert self._dx_hat is not None
        self._dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_hat

        speed = float(np.linalg.norm(self._dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        alpha_x = _alpha(cutoff, dt)

        assert self._x_hat is not None
        self._x_hat = alpha_x * x + (1.0 - alpha_x) * self._x_hat

        self._x_prev = x.copy()
        self._prev_time = float(timestamp)
        return self._x_hat.copy()
