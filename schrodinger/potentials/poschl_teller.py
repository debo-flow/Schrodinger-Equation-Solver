import math
from typing import Literal
import numpy as np
from .base import Potential


class PoschlTellerPotential(Potential):
    """
    Symmetric Pöschl–Teller reflectionless potential well:
        V(x) = -V_0 * sech^2(alpha * (x - center))
    """

    _MAX_ARG = 350.0

    def __init__(
        self,
        depth: float = 1.0,
        alpha: float = 1.0,
        center: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError(f"depth (V_0) must be finite and strictly positive. Received: {depth}")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be finite and strictly positive. Received: {alpha}")
        if not math.isfinite(center):
            raise ValueError(f"center must be finite. Received: {center}")

        self.depth = float(depth)
        self.alpha = float(alpha)
        self.center = float(center)

    @classmethod
    def from_lambda(
        cls,
        lam: float,
        alpha: float = 1.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ) -> "PoschlTellerPotential":
        """
        Constructs a Pöschl–Teller potential with depth V_0 = (hbar^2 * alpha^2 / 2m) * lam * (lam - 1),
        supporting exact integer/real depth parameter lambda > 1.
        """
        if lam <= 1.0:
            raise ValueError(f"Depth parameter lambda must be > 1 for bound states. Received: {lam}")
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        depth = ((hbar ** 2) * (alpha ** 2) / (2.0 * mass)) * lam * (lam - 1.0)
        return cls(depth=depth, alpha=alpha, center=center, unit_system=unit_system)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        arg = np.clip(self.alpha * (x - self.center), -self._MAX_ARG, self._MAX_ARG)
        sech = 1.0 / np.cosh(arg)
        return -self.depth * (sech ** 2)
