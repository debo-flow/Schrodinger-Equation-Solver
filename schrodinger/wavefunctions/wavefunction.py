import numpy as np
from .probability_density import ProbabilityDensity
from .quantum_observables import QuantumObservables
from .normalization import NormalizationEngine

class Wavefunction:
    """
    Represents a continuous physical quantum eigenstate.
    Validates on initialization and lazily computes properties.
    """

    def __init__(self, x: np.ndarray, psi: np.ndarray, energy: float):
        if len(x) != len(psi):
            raise ValueError(f"Grid size ({len(x)}) and wavefunction size ({len(psi)}) must match.")
        if not np.all(np.isfinite(psi)):
            raise ValueError("Wavefunction contains NaN or Inf.")
            
        self._x = x
        self._psi = psi.astype(complex)
        self._energy = float(energy)

        # Self-verify normalization
        if not NormalizationEngine.verify_normalization(self._psi, self._x):
            raise ValueError("Wavefunction is not continuously normalized.")

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def psi(self) -> np.ndarray:
        return self._psi

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def probability_density(self) -> np.ndarray:
        return ProbabilityDensity.calculate(self._psi)

    @property
    def expectation_x(self) -> float:
        return QuantumObservables.expectation_x(self._psi, self._x)

    @property
    def variance_x(self) -> float:
        return QuantumObservables.variance_x(self._psi, self._x)

    @property
    def nodes(self) -> int:
        return QuantumObservables.count_nodes(self._psi)
