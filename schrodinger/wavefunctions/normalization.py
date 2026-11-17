import numpy as np
from scipy import integrate
from enum import Enum
from typing import Tuple

class IntegrationMethod(Enum):
    SIMPSON = "simpson"
    TRAPEZOIDAL = "trapezoidal"


class NormalizationEngine:
    """Handles continuous numerical normalization of wavefunctions."""

    @staticmethod
    def calculate_norm(
        psi: np.ndarray, 
        x: np.ndarray, 
        method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> float:
        """
        Calculates the continuous L2 norm of the wavefunction: sqrt(int |psi|^2 dx).
        """
        density = np.abs(psi)**2
        
        if method == IntegrationMethod.SIMPSON:
            integral = integrate.simpson(y=density, x=x)
        else:
            integral = integrate.trapezoid(y=density, x=x)
            
        if integral <= 0:
            raise ValueError(f"Wavefunction integral non-positive ({integral}). Cannot normalize.")
            
        return float(np.sqrt(integral))

    @staticmethod
    def normalize(
        psi: np.ndarray, 
        x: np.ndarray, 
        method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Tuple[np.ndarray, float]:
        """
        Returns the normalized wavefunction and its original normalization constant.
        """
        norm_constant = NormalizationEngine.calculate_norm(psi, x, method)
        normalized_psi = psi / norm_constant
        return normalized_psi, norm_constant

    @staticmethod
    def verify_normalization(psi: np.ndarray, x: np.ndarray, tolerance: float = 1e-5) -> bool:
        """Verifies if the wavefunction integral evaluates to 1.0 within tolerance."""
        density = np.abs(psi)**2
        integral = integrate.simpson(y=density, x=x)
        return abs(integral - 1.0) < tolerance
