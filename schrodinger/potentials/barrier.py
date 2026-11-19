import math
from typing import Literal
import numpy as np
from .base import Potential


class BarrierPotential(Potential):
    """
    Finite rectangular potential barrier:
        V(x) = height    for center - width/2 <= x <= center + width/2
        V(x) = baseline  otherwise
    """

    def __init__(
        self,
        height: float = 1.0,
        width: float = 1.0,
        center: float = 0.0,
        baseline: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(height):
            raise ValueError(f"height must be finite. Received: {height}")
        if not math.isfinite(baseline):
            raise ValueError(f"baseline must be finite. Received: {baseline}")
        if height <= baseline:
            raise ValueError(f"Barrier height ({height}) must be strictly greater than baseline ({baseline}).")
        if not math.isfinite(width) or width <= 0:
            raise ValueError(f"width must be finite and strictly positive. Received: {width}")
        if not math.isfinite(center):
            raise ValueError(f"center must be finite. Received: {center}")

        self.height = float(height)
        self.width = float(width)
        self.center = float(center)
        self.baseline = float(baseline)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        v = np.full_like(x, self.baseline, dtype=np.float64)
        left = self.center - 0.5 * self.width
        right = self.center + 0.5 * self.width
        inside = (x >= left) & (x <= right)
        v[inside] = self.height
        return v
