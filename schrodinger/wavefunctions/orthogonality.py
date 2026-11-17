import numpy as np
from scipy import integrate

class OrthogonalityAnalyzer:
    """Validates orthogonality between distinct quantum eigenstates."""

    @staticmethod
    def calculate_overlap(psi_1: np.ndarray, psi_2: np.ndarray, x: np.ndarray) -> complex:
        """
        Calculates the inner product <psi_1 | psi_2> = integral(psi_1* * psi_2 dx).
        """
        if len(psi_1) != len(psi_2) or len(psi_1) != len(x):
            raise ValueError("Dimension mismatch between wavefunctions and spatial grid.")
            
        integrand = np.conj(psi_1) * psi_2
        return complex(integrate.simpson(y=integrand, x=x))

    @staticmethod
    def check_orthogonality(
        psi_1: np.ndarray, 
        psi_2: np.ndarray, 
        x: np.ndarray, 
        tolerance: float = 1e-5
    ) -> bool:
        """
        Returns True if the two states are numerically orthogonal.
        """
        overlap = OrthogonalityAnalyzer.calculate_overlap(psi_1, psi_2, x)
        return abs(overlap) < tolerance
