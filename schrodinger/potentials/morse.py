import math
from typing import Literal
import numpy as np
from .base import Potential


class MorsePotential(Potential):
    """
    One-dimensional Morse anharmonic molecular potential:
        V(x) = D_e * (1 - exp(-a * (x - x_e)))^2
    """

    _MAX_EXPONENT = 350.0  # Prevents IEEE-754 float64 overflow when squared

    def __init__(
        self,
        dissociation_energy: float = 10.0,
        width_parameter: float = 1.0,
        equilibrium_position: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(dissociation_energy) or dissociation_energy <= 0:
            raise ValueError(f"dissociation_energy (D_e) must be finite and strictly positive. Received: {dissociation_energy}")
        if not math.isfinite(width_parameter) or width_parameter <= 0:
            raise ValueError(f"width_parameter (a) must be finite and strictly positive. Received: {width_parameter}")
        if not math.isfinite(equilibrium_position):
            raise ValueError(f"equilibrium_position (x_e) must be finite. Received: {equilibrium_position}")

        self.dissociation_energy = float(dissociation_energy)
        self.width_parameter = float(width_parameter)
        self.equilibrium_position = float(equilibrium_position)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        exponent = -self.width_parameter * (x - self.equilibrium_position)
        safe_exponent = np.clip(exponent, -self._MAX_EXPONENT, self._MAX_EXPONENT)
        term = 1.0 - np.exp(safe_exponent)
        return self.dissociation_energy * (term ** 2)
