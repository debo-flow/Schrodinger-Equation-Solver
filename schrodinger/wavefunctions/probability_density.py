import numpy as np

class ProbabilityDensity:
    """Evaluates the Born probability density of a quantum state."""

    @staticmethod
    def calculate(psi: np.ndarray) -> np.ndarray:
        """
        Calculates rho(x) = |psi(x)|^2
        Always returns a strictly real, non-negative array.
        """
        return np.abs(psi)**2
