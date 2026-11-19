import math
from typing import Literal
import numpy as np
from .base import Potential


class PeriodicPotential(Potential):
    """
    Configurable sinusoidal periodic lattice potential:
        V(x) = amplitude * cos(2 * pi * (x - phase_shift) / period) + offset
    or sine waveform.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        period: float = 1.0,
        phase_shift: float = 0.0,
        offset: float = 0.0,
        waveform: Literal["cosine", "sine"] = "cosine",
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite. Received: {amplitude}")
        if not math.isfinite(period) or period <= 0:
            raise ValueError(f"period must be finite and strictly positive. Received: {period}")
        if not math.isfinite(phase_shift):
            raise ValueError(f"phase_shift must be finite. Received: {phase_shift}")
        if not math.isfinite(offset):
            raise ValueError(f"offset must be finite. Received: {offset}")
        if waveform not in ("cosine", "sine"):
            raise ValueError("waveform must be 'cosine' or 'sine'.")

        self.amplitude = float(amplitude)
        self.period = float(period)
        self.phase_shift = float(phase_shift)
        self.offset = float(offset)
        self.waveform = waveform

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        arg = (2.0 * math.pi / self.period) * (x - self.phase_shift)
        trig = np.cos(arg) if self.waveform == "cosine" else np.sin(arg)
        return self.amplitude * trig + self.offset
