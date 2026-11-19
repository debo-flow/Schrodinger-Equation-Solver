import math
from typing import Literal
import numpy as np
from .base import Potential


class CoulombLikePotential(Potential):
    """
    One-dimensional soft-core regularized Coulomb-like potential:
        V(x) = -strength / sqrt((x - center)^2 + softening^2)

    Singularity Treatment:
        The bare 1D Coulomb interaction -k/|x - x_c| possesses a non-integrable
        singularity at x = x_c causing unphysical ground-state collapse. Introducing
        the positive soft-core length `softening` (a > 0) caps the minimum value at
        V(center) = -strength / softening while preserving the asymptotic -k/|x - x_c|
        tail for |x - x_c| >> softening.
    """

    def __init__(
        self,
        strength: float = 1.0,
        softening: float = 0.1,
        center: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(strength) or strength == 0.0:
            raise ValueError(f"strength must be finite and non-zero. Received: {strength}")
        if not math.isfinite(softening) or softening <= 0.0:
            raise ValueError(f"softening parameter (a) must be finite and strictly positive. Received: {softening}")
        if not math.isfinite(center):
            raise ValueError(f"center must be finite. Received: {center}")

        self.strength = float(strength)
        self.softening = float(softening)
        self.center = float(center)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        distance = np.hypot(x - self.center, self.softening)
        return -self.strength / distance
