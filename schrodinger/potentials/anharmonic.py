import math
from typing import Literal
import numpy as np
from .base import Potential


class QuarticAnharmonicOscillator(Potential):
    """
    One-dimensional Quartic Anharmonic Oscillator potential:
        V(x) = a * (x - center)^2 + b * (x - center)^4
    """

    def __init__(
        self,
        quadratic_coeff: float = 0.5,
        quartic_coeff: float = 0.1,
        center: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(quadratic_coeff) or quadratic_coeff < 0:
            raise ValueError(f"quadratic_coeff (a) must be finite and >= 0. Received: {quadratic_coeff}")
        if not math.isfinite(quartic_coeff) or quartic_coeff < 0:
            raise ValueError(f"quartic_coeff (b) must be finite and >= 0. Received: {quartic_coeff}")
        if quadratic_coeff + quartic_coeff <= 0:
            raise ValueError("At least one of quadratic_coeff or quartic_coeff must be strictly positive.")
        if not math.isfinite(center):
            raise ValueError(f"center must be finite. Received: {center}")

        self.quadratic_coeff = float(quadratic_coeff)
        self.quartic_coeff = float(quartic_coeff)
        self.center = float(center)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        dx = x - self.center
        dx2 = dx ** 2
        v = self.quadratic_coeff * dx2 + self.quartic_coeff * (dx2 ** 2)
        if not np.all(np.isfinite(v)):
            raise OverflowError("QuarticAnharmonicOscillator evaluation overflowed float64 range.")
        return v
