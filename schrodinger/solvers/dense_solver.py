import numpy as np
from scipy import linalg, sparse
from typing import Tuple

class DenseSolver:
    """Solves the eigenvalue problem using dense linear algebra (eigh)."""

    @staticmethod
    def solve(matrix: np.ndarray, num_eigenvalues: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves H*psi = E*psi for dense Hermitian matrices.
        
        Args:
            matrix (np.ndarray): Dense Hamiltonian matrix.
            num_eigenvalues (int): Number of lowest eigenstates to return.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Energies, Wavefunctions [columns]).
        """
        # eigh is optimized for Hermitian/symmetric matrices
        energies, wavefunctions = linalg.eigh(matrix)
        
        # Ensure proper sorting (eigh normally sorts, but we guarantee it)
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]
        
        # Slice to requested number of eigenvalues
        limit = min(num_eigenvalues, len(energies))
        return energies[:limit], wavefunctions[:, :limit]
