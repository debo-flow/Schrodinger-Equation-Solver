import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from typing import Union, Dict, Any, Optional

class SpectralAnalyzer:
    """Computes preliminary spectral invariants of the Hamiltonian."""

    @staticmethod
    def calculate_trace(matrix: Union[np.ndarray, sparse.spmatrix]) -> float:
        """Computes the trace (sum of diagonal elements) of the matrix."""
        if sparse.issparse(matrix):
            return float(matrix.diagonal().sum())
        return float(np.trace(matrix))

    @staticmethod
    def calculate_norm(matrix: Union[np.ndarray, sparse.spmatrix]) -> float:
        """Computes the Frobenius norm of the matrix."""
        if sparse.issparse(matrix):
            return float(sparse_norm(matrix, ord='fro'))
        return float(np.linalg.norm(matrix, ord='fro'))

    @staticmethod
    def analyze_dense_properties(matrix: Union[np.ndarray, sparse.spmatrix], max_size: int = 2000) -> Dict[str, Optional[float]]:
        """
        Computes computationally expensive properties like determinant and rank.
        Only executed if the matrix is small enough to avoid memory/time exhaustion.
        """
        result: Dict[str, Optional[float]] = {"determinant": None, "rank": None}
        
        if matrix.shape[0] > max_size:
            return result
            
        dense_mat = matrix.toarray() if sparse.issparse(matrix) else matrix
        
        # Calculate Rank
        result["rank"] = int(np.linalg.matrix_rank(dense_mat))
        
        # Calculate Determinant (can overflow/underflow easily for large matrices)
        try:
            # Using slogdet to prevent standard overflow
            sign, logdet = np.linalg.slogdet(dense_mat)
            if sign == 0:
                result["determinant"] = 0.0
            else:
                result["determinant"] = float(sign * np.exp(logdet))
        except Exception:
            pass 
            
        return result
