import numpy as np
from scipy import sparse
from typing import Literal

class OperatorValidator:
    """Provides validation logic for physical constants and quantum matrices."""

    @staticmethod
    def validate_physical_constants(mass: float, hbar: float) -> None:
        """Ensures mass and hbar are strictly positive and physically valid."""
        if mass <= 0:
            raise ValueError(f"Particle mass must be strictly positive. Received: {mass}")
        if hbar <= 0:
            raise ValueError(f"Reduced Planck constant (hbar) must be strictly positive. Received: {hbar}")

    @staticmethod
    def validate_unit_consistency(unit_a: str, unit_b: str) -> None:
        """Ensures operators belong to the same unit system before addition."""
        if unit_a != unit_b:
            raise ValueError(f"Unit mismatch: cannot mix '{unit_a}' with '{unit_b}'.")

    @staticmethod
    def check_hermiticity(matrix: sparse.spmatrix, tol: float = 1e-10) -> None:
        """
        Validates that a real matrix is Hermitian (Symmetric: H = H.T).
        Raises an exception if the matrix violates symmetry beyond the given tolerance.
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Hamiltonian matrix must be square.")
            
        # For real matrices, H.conjugate().transpose() is just H.transpose()
        diff = matrix - matrix.T
        
        # Calculate the max absolute difference safely for sparse matrices
        max_diff = abs(diff).max()
        
        if max_diff > tol:
            raise ValueError(f"Matrix is not Hermitian (Symmetric). Max asymmetry: {max_diff} > tolerance {tol}")
