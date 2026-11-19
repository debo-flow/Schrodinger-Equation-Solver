import math
from typing import Literal
import numpy as np
from .base import Potential


class StepPotential(Potential):
    """
    Piecewise constant potential step:
        V(x) = left_height   for x < step_position
        V(x) = right_height  for x >= step_position
    """

    def __init__(
        self,
        step_position: float = 0.0,
        left_height: float = 0.0,
        right_height: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(step_position):
            raise ValueError(f"step_position must be finite. Received: {step_position}")
        if not math.isfinite(left_height):
            raise ValueError(f"left_height must be finite. Received: {left_height}")
        if not math.isfinite(right_height):
            raise ValueError(f"right_height must be finite. Received: {right_height}")

        self.step_position = float(step_position)
        self.left_height = float(left_height)
        self.right_height = float(right_height)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < self.step_position, self.left_height, self.right_height).astype(np.float64)
