import numpy as np
from scipy import sparse
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine

class MomentumOperator:
    """Constructs momentum operators using finite difference derivatives."""

    @staticmethod
    def build_p(fd_engine: FiniteDifferenceEngine, hbar: float = 1.0, accuracy_order: int = 2) -> sparse.csr_matrix:
        """
        Constructs the momentum operator p = -i * hbar * d/dx.
        Returns a complex sparse matrix.
        """
        d1_matrix = fd_engine.build_differentiation_matrix(
            derivative_order=1, 
            accuracy_order=accuracy_order,
            interior_only=False
        )
        return d1_matrix.multiply(-1j * hbar)

    @staticmethod
    def build_p_squared(fd_engine: FiniteDifferenceEngine, hbar: float = 1.0, accuracy_order: int = 2) -> sparse.csr_matrix:
        """
        Constructs the momentum-squared operator p^2 = -hbar^2 * d^2/dx^2.
        Returns a real sparse matrix.
        """
        d2_matrix = fd_engine.build_differentiation_matrix(
            derivative_order=2, 
            accuracy_order=accuracy_order,
            interior_only=False
        )
        return d2_matrix.multiply(-(hbar**2))
