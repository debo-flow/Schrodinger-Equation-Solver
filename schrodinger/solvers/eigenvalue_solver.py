import numpy as np
from scipy import sparse
from typing import Union, Tuple, Dict, Any
from schrodinger.analysis.hermiticity import HermiticityAnalyzer
from .solver_configuration import SolverConfig, SolverMethod
from .dense_solver import DenseSolver
from .sparse_solver import SparseSolver
from .eigenpair_validation import EigenpairValidator
from .degeneracy import DegeneracyAnalyzer

class EigenvalueSolver:
    """
    Main orchestration engine for quantum eigenvalue computation.
    Routes between dense/sparse, enforces validation, and analyzes degeneracies.
    """

    def __init__(self, config: SolverConfig = SolverConfig()):
        self.config = config
        self.hermiticity_analyzer = HermiticityAnalyzer(tolerance=1e-8)

    def _select_method(self, n_dim: int) -> SolverMethod:
        """Decides whether to use dense or sparse solvers based on size and config."""
        if self.config.method != SolverMethod.AUTO:
            return self.config.method
            
        # If requesting too many eigenvalues for ARPACK, force Dense
        if self.config.num_eigenvalues >= n_dim - 1:
            return SolverMethod.DENSE
            
        if n_dim > self.config.sparse_threshold:
            return SolverMethod.SPARSE
        return SolverMethod.DENSE

    def solve(self, matrix: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
        """
        Computes valid energy eigenpairs for the given Hamiltonian matrix.
        
        Returns:
            Dict containing:
                - 'energies': 1D array of eigenvalues.
                - 'wavefunctions': 2D array (columns are eigenstates).
                - 'residuals': List of residual norms.
                - 'degeneracies': List of degenerate state indices.
        """
        # 1. Validation before solving
        is_herm, dev = self.hermiticity_analyzer.is_hermitian(matrix)
        if not is_herm:
            raise ValueError(f"Matrix is not Hermitian (Deviation: {dev:.2e}). Cannot compute real physical energies.")
            
        n_dim = matrix.shape[0]
        method = self._select_method(n_dim)

        # 2. Routing to Engine
        if method == SolverMethod.DENSE:
            dense_mat = matrix.toarray() if sparse.issparse(matrix) else matrix
            energies, wavefunctions = DenseSolver.solve(dense_mat, self.config.num_eigenvalues)
        else:
            sparse_mat = matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
            energies, wavefunctions = SparseSolver.solve(sparse_mat, self.config)

        # 3. Eigenpair Validation (H*psi = E*psi)
        # Using an inherently slightly higher tolerance for sparse iterative results
        val_tol = max(self.config.tolerance * 100, 1e-6)
        residuals = EigenpairValidator.validate_all_pairs(matrix, energies, wavefunctions, tolerance=val_tol)

        # 4. Analysis
        degeneracies = DegeneracyAnalyzer.find_degeneracies(
            energies, 
            tolerance=self.config.degeneracy_tolerance
        )

        return {
            "energies": energies,
            "wavefunctions": wavefunctions,
            "residuals": residuals,
            "degeneracies": degeneracies,
            "method_used": method.value
        }
