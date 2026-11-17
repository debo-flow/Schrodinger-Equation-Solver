import numpy as np
from scipy import sparse
from typing import Union, Tuple

class HermiticityAnalyzer:
    """Analyzes the Hermiticity of dense or sparse quantum operators."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def measure_deviation(self, matrix: Union[np.ndarray, sparse.spmatrix]) -> float:
        """
        Calculates the maximum absolute deviation between H and H^dagger.
        """
        if sparse.issparse(matrix):
            diff = matrix - matrix.conjugate().transpose()
            if diff.nnz == 0:
                return 0.0
            return float(np.max(np.abs(diff.data)))
        else:
            diff = matrix - matrix.conj().T
            return float(np.max(np.abs(diff)))

    def is_hermitian(self, matrix: Union[np.ndarray, sparse.spmatrix]) -> Tuple[bool, float]:
        """
        Returns a boolean indicating if the matrix is Hermitian, and the deviation.
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False, float('inf')
            
        deviation = self.measure_deviation(matrix)
        return deviation <= self.tolerance, deviation
