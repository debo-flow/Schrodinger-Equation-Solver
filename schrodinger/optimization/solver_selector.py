from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np
from scipy import sparse


class OptimizedSolverMethod(Enum):
    """Supported numerical eigensolver backends."""
    TRIDIAGONAL = "tridiagonal"
    BANDED = "banded"
    DENSE = "dense"
    SPARSE_SHIFT_INVERT = "sparse_shift_invert"
    SPARSE_DIRECT = "sparse_direct"


@dataclass
class MatrixStructure:
    """Structural summary of a Hamiltonian matrix used for solver routing."""
    dimension: int
    nnz: int
    sparsity_ratio: float
    is_complex: bool
    bandwidth: Optional[int]
    is_tridiagonal: bool


@dataclass
class SolverSelectionCriteria:
    """Input criteria governing automatic solver selection."""
    dimension: int
    num_eigenvalues: int
    sparsity_ratio: float = 0.0
    is_complex: bool = False
    is_symmetric_tridiagonal: bool = False
    bandwidth: Optional[int] = None
    available_memory_mb: float = 512.0
    accuracy_tolerance: float = 1e-10


@dataclass
class SolverRecommendation:
    """Result of automatic solver selection with justification and fallback."""
    method: OptimizedSolverMethod
    fallback_method: OptimizedSolverMethod
    estimated_memory_mb: float
    reason: str


class SolverSelector:
    """
    Intelligent solver selection system that chooses the optimal Hermitian
    eigensolver based on matrix structure, dimensions, sparsity, memory budget,
    and accuracy requirements.
    """

    DENSE_SIZE_LIMIT = 600
    BANDED_SIZE_LIMIT = 4000
    MAX_BANDWIDTH_FOR_BANDED = 4

    @staticmethod
    def analyze_matrix_structure(matrix: Union[np.ndarray, sparse.spmatrix]) -> MatrixStructure:
        """Inspects matrix sparsity, dtype, and bandwidth without dense blowup."""
        n = int(matrix.shape[0])
        is_complex = bool(np.iscomplexobj(matrix))

        if sparse.issparse(matrix):
            coo = matrix.tocoo()
            nnz = int(coo.nnz)
            sparsity_ratio = 1.0 - (nnz / float(n * n)) if n > 0 else 0.0
            bandwidth = int(np.max(np.abs(coo.row - coo.col))) if nnz > 0 else 0
        else:
            nz_rows, nz_cols = np.nonzero(matrix)
            nnz = int(len(nz_rows))
            sparsity_ratio = 1.0 - (nnz / float(n * n)) if n > 0 else 0.0
            bandwidth = int(np.max(np.abs(nz_rows - nz_cols))) if nnz > 0 else 0

        is_tridiagonal = (bandwidth <= 1) and (not is_complex)
        return MatrixStructure(
            dimension=n,
            nnz=nnz,
            sparsity_ratio=sparsity_ratio,
            is_complex=is_complex,
            bandwidth=bandwidth,
            is_tridiagonal=is_tridiagonal
        )

    @staticmethod
    def estimate_dense_memory_mb(dimension: int, is_complex: bool = False) -> float:
        """Estimates peak memory in MB required for dense LAPACK eigh (matrix + workspace)."""
        bytes_per_elem = 16 if is_complex else 8
        # Dense matrix + eigenvectors + LAPACK workspace (~3x matrix footprint)
        total_bytes = 3.0 * bytes_per_elem * (dimension ** 2)
        return total_bytes / (1024.0 * 1024.0)

    @staticmethod
    def estimate_sparse_memory_mb(dimension: int, nnz: int, num_eigenvalues: int, is_complex: bool = False) -> float:
        """Estimates memory in MB for sparse CSR storage + Krylov basis vectors."""
        bytes_per_elem = 16 if is_complex else 8
        csr_bytes = nnz * (bytes_per_elem + 4) + (dimension + 1) * 4
        ncv = min(dimension, max(2 * num_eigenvalues + 1, 20))
        krylov_bytes = dimension * ncv * bytes_per_elem * 2
        return (csr_bytes + krylov_bytes) / (1024.0 * 1024.0)

    @classmethod
    def select(cls, criteria: SolverSelectionCriteria) -> SolverRecommendation:
        """Selects the optimal solver method and fallback strategy."""
        n = criteria.dimension
        k = criteria.num_eigenvalues
        if n <= 0:
            raise ValueError(f"Matrix dimension must be positive. Received: {n}")
        if k <= 0 or k > n:
            raise ValueError(f"Requested eigenvalues ({k}) must be within [1, {n}].")

        dense_mem_mb = cls.estimate_dense_memory_mb(n, criteria.is_complex)
        nnz_est = max(n, int(round((1.0 - criteria.sparsity_ratio) * n * n)))
        sparse_mem_mb = cls.estimate_sparse_memory_mb(n, nnz_est, k, criteria.is_complex)

        # 1. Exact symmetric real tridiagonal solver: O(N) storage, fast & exact
        if criteria.is_symmetric_tridiagonal and not criteria.is_complex and n >= 16:
            tri_mem_mb = (n * (k + 4) * 8) / (1024.0 * 1024.0)
            fallback = (
                OptimizedSolverMethod.SPARSE_SHIFT_INVERT
                if (n > cls.DENSE_SIZE_LIMIT and k < n - 1)
                else OptimizedSolverMethod.DENSE
            )
            return SolverRecommendation(
                method=OptimizedSolverMethod.TRIDIAGONAL,
                fallback_method=fallback,
                estimated_memory_mb=tri_mem_mb,
                reason="Real symmetric tridiagonal structure enables O(N)-memory eigh_tridiagonal."
            )

        # 2. Small-bandwidth real symmetric banded solver (e.g., 4th-order FD)
        if (
            criteria.bandwidth is not None
            and 2 <= criteria.bandwidth <= cls.MAX_BANDWIDTH_FOR_BANDED
            and not criteria.is_complex
            and n <= cls.BANDED_SIZE_LIMIT
            and k <= n
        ):
            band_mem_mb = ((criteria.bandwidth + 1 + k) * n * 8) / (1024.0 * 1024.0)
            if band_mem_mb <= criteria.available_memory_mb:
                fallback = (
                    OptimizedSolverMethod.SPARSE_SHIFT_INVERT
                    if (n > cls.DENSE_SIZE_LIMIT and k < n - 1)
                    else OptimizedSolverMethod.DENSE
                )
                return SolverRecommendation(
                    method=OptimizedSolverMethod.BANDED,
                    fallback_method=fallback,
                    estimated_memory_mb=band_mem_mb,
                    reason=f"Real symmetric banded matrix (bandwidth={criteria.bandwidth}) fits eig_banded."
                )

        # 3. If k is close to N, ARPACK cannot run -> Dense required
        if k >= n - 1:
            return SolverRecommendation(
                method=OptimizedSolverMethod.DENSE,
                fallback_method=OptimizedSolverMethod.DENSE,
                estimated_memory_mb=dense_mem_mb,
                reason="Requested eigenvalue count approaches matrix dimension; dense solver required."
            )

        # 4. Large sparse matrix or memory-constrained dense matrix -> Sparse solver
        is_sparse_enough = criteria.sparsity_ratio >= 0.7
        exceeds_dense_budget = dense_mem_mb > criteria.available_memory_mb
        if (n > cls.DENSE_SIZE_LIMIT and is_sparse_enough) or exceeds_dense_budget:
            if k <= max(20, n // 10):
                return SolverRecommendation(
                    method=OptimizedSolverMethod.SPARSE_SHIFT_INVERT,
                    fallback_method=OptimizedSolverMethod.SPARSE_DIRECT,
                    estimated_memory_mb=sparse_mem_mb,
                    reason=(
                        f"Large sparse matrix (N={n}, sparsity={criteria.sparsity_ratio:.2%}); "
                        "shift-invert ARPACK accelerates lowest-state convergence."
                    )
                )
            return SolverRecommendation(
                method=OptimizedSolverMethod.SPARSE_DIRECT,
                fallback_method=OptimizedSolverMethod.DENSE,
                estimated_memory_mb=sparse_mem_mb,
                reason="Large sparse matrix requesting a moderate fraction of spectrum."
            )

        # 5. Default for small or dense matrices -> Dense eigh
        return SolverRecommendation(
            method=OptimizedSolverMethod.DENSE,
            fallback_method=OptimizedSolverMethod.SPARSE_DIRECT,
            estimated_memory_mb=dense_mem_mb,
            reason=f"Small or dense matrix (N={n}, est={dense_mem_mb:.2f} MB) is optimal for dense eigh."
        )

    @classmethod
    def select_for_matrix(
        cls,
        matrix: Union[np.ndarray, sparse.spmatrix],
        num_eigenvalues: int,
        available_memory_mb: float = 512.0,
        accuracy_tolerance: float = 1e-10
    ) -> SolverRecommendation:
        """Convenience helper that inspects a concrete matrix and returns a recommendation."""
        struct = cls.analyze_matrix_structure(matrix)
        criteria = SolverSelectionCriteria(
            dimension=struct.dimension,
            num_eigenvalues=num_eigenvalues,
            sparsity_ratio=struct.sparsity_ratio,
            is_complex=struct.is_complex,
            is_symmetric_tridiagonal=struct.is_tridiagonal,
            bandwidth=struct.bandwidth,
            available_memory_mb=available_memory_mb,
            accuracy_tolerance=accuracy_tolerance
        )
        return cls.select(criteria)
