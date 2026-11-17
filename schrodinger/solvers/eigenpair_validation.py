import numpy as np
from scipy import sparse
from typing import Union, Tuple, List

class EigenpairValidator:
    """Validates the mathematical correctness of computed (E, psi) pairs."""

    @staticmethod
    def check_finiteness(energies: np.ndarray, wavefunctions: np.ndarray) -> None:
        """Ensures no NaN or Inf values contaminated the solution."""
        if not np.all(np.isfinite(energies)):
            raise ValueError("Computed energy eigenvalues contain non-finite values.")
        if not np.all(np.isfinite(wavefunctions)):
            raise ValueError("Computed wavefunctions contain non-finite values.")

    @staticmethod
    def calculate_residual(
        matrix: Union[np.ndarray, sparse.spmatrix], 
        energy: float, 
        wavefunction: np.ndarray
    ) -> float:
        """
        Calculates r = || H*psi - E*psi ||_2
        """
        if sparse.issparse(matrix):
            h_psi = matrix.dot(wavefunction)
        else:
            h_psi = matrix @ wavefunction
            
        e_psi = energy * wavefunction
        residual_vector = h_psi - e_psi
        return float(np.linalg.norm(residual_vector))

    @staticmethod
    def validate_all_pairs(
        matrix: Union[np.ndarray, sparse.spmatrix], 
        energies: np.ndarray, 
        wavefunctions: np.ndarray,
        tolerance: float = 1e-5
    ) -> List[float]:
        """
        Validates finiteness and residuals for all eigenpairs.
        Returns the list of residuals. Raises error if any exceed tolerance.
        """
        EigenpairValidator.check_finiteness(energies, wavefunctions)
        
        residuals = []
        for i in range(len(energies)):
            psi = wavefunctions[:, i]
            res = EigenpairValidator.calculate_residual(matrix, energies[i], psi)
            residuals.append(res)
            
            if res > tolerance:
                raise ValueError(
                    f"Eigenpair {i} failed residual validation: r={res:.2e} > tol={tolerance:.2e}"
                )
                
        return residuals
