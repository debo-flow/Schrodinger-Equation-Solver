import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional
from .solver_configuration import SolverConfig

class SparseSolver:
    """Solves the eigenvalue problem using Lanczos/Arnoldi iteration (eigsh)."""

    @staticmethod
    def solve(matrix: sparse.spmatrix, config: SolverConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves H*psi = E*psi for sparse Hermitian matrices.
        
        Args:
            matrix (sparse.spmatrix): Sparse Hamiltonian matrix.
            config (SolverConfig): Solver settings.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Energies, Wavefunctions [columns]).
        """
        # eigsh requires k < N - 1
        n_dim = matrix.shape[0]
        k = min(config.num_eigenvalues, n_dim - 2)
        
        if k <= 0:
            raise ValueError("Matrix dimension too small for sparse solver. Use dense solver.")
            
        energies, wavefunctions = eigsh(
            matrix, 
            k=k, 
            which='SA',  # Smallest Algebraic (lowest energy states)
            tol=config.tolerance,
            maxiter=config.max_iterations
        )
        
        # Sort results
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]
        
        return energies, wavefunctions
