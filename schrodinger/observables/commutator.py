from scipy import sparse
import numpy as np
from .observable_engine import ObservableEngine
from schrodinger.wavefunctions.wavefunction import Wavefunction

class CommutatorEngine:
    """Evaluates algebraic operator commutators [A, B] = AB - BA."""

    @staticmethod
    def evaluate_matrix(A: sparse.spmatrix, B: sparse.spmatrix) -> sparse.csr_matrix:
        """
        Calculates the explicit commutator matrix [A, B].
        """
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same dimensions to commute.")
        
        # Sparse matrix multiplication
        AB = A.dot(B)
        BA = B.dot(A)
        return (AB - BA).tocsr()

    @staticmethod
    def expectation(A: sparse.spmatrix, B: sparse.spmatrix, wf: Wavefunction) -> complex:
        """
        Evaluates the expectation value of the commutator <psi|[A, B]|psi>.
        Note: The expectation of a commutator of Hermitian operators is purely imaginary.
        """
        commutator = CommutatorEngine.evaluate_matrix(A, B)
        # We set is_hermitian=False because [A, B] is anti-Hermitian if A and B are Hermitian.
        return ObservableEngine.expectation(commutator, wf, is_hermitian=False)
