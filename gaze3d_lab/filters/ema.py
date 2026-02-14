from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

VectorN = npt.NDArray[np.float64]


@dataclass
class EMAFilter:
    alpha: float
    _state: VectorN | None = None

    def __post_init__(self) -> None:
        self.set_alpha(self.alpha)

    def reset(self) -> None:
        self._state = None

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.001, 1.0))

    def update(self, value: Sequence[float] | VectorN) -> VectorN:
        current = np.asarray(value, dtype=np.float64)
        if self._state is None:
            self._state = current.copy()
            return self._state.copy()

        self._state = self.alpha * current + (1.0 - self.alpha) * self._state
        return self._state.copy()
