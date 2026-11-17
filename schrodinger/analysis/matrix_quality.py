import numpy as np
from scipy import sparse
from typing import Union, List

class MatrixQualityAnalyzer:
    """Checks the underlying numerical stability and validity of the matrix data."""

    @staticmethod
    def check_finiteness(matrix: Union[np.ndarray, sparse.spmatrix]) -> bool:
        """Ensures the matrix contains no NaN or Inf values."""
        if sparse.issparse(matrix):
            return bool(np.all(np.isfinite(matrix.data)))
        return bool(np.all(np.isfinite(matrix)))

    @staticmethod
    def find_invalid_entries(matrix: Union[np.ndarray, sparse.spmatrix]) -> List[str]:
        """Returns a list of warnings regarding the matrix quality."""
        warnings = []
        if not MatrixQualityAnalyzer.check_finiteness(matrix):
            warnings.append("Matrix contains NaN or Inf values.")
            
        if matrix.shape[0] != matrix.shape[1]:
            warnings.append(f"Matrix is not square. Shape: {matrix.shape}")
            
        return warnings
