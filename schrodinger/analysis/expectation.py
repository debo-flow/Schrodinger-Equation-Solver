import numpy as np
from scipy import sparse
from typing import Union

class ExpectationCalculator:
    """Calculates physical expectation values <psi|H|psi>."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def evaluate(self, matrix: Union[np.ndarray, sparse.spmatrix], psi: np.ndarray, normalize: bool = True) -> float:
        """
        Computes <psi|H|psi>. Returns the strictly real part if valid.
        
        Args:
            matrix: The Hamiltonian operator.
            psi: The wavefunction vector.
            normalize: If True, divides by <psi|psi>.
            
        Returns:
            float: The real-valued energy expectation.
            
        Raises:
            ValueError: If dimensions mismatch or the result is unphysically complex.
        """
        if matrix.shape[0] != psi.shape[0]:
            raise ValueError(f"Dimension mismatch: Matrix is {matrix.shape}, Wavefunction is {psi.shape}")

        # Compute denominator <psi|psi>
        norm_squared = np.vdot(psi, psi)
        if norm_squared == 0:
            raise ValueError("Wavefunction norm is zero.")

        # Compute numerator <psi|H|psi>
        if sparse.issparse(matrix):
            h_psi = matrix.dot(psi)
        else:
            h_psi = matrix @ psi
            
        expectation_val = np.vdot(psi, h_psi)

        if normalize:
            expectation_val /= norm_squared

        # Verify physicality
        if abs(np.imag(expectation_val)) > self.tolerance:
            raise ValueError(f"Expectation value has significant imaginary component: {expectation_val}")

        return float(np.real(expectation_val))
