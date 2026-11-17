import numpy as np
from scipy import sparse
from .kinetic_operator import KineticEnergyOperator
from .potential_operator import PotentialEnergyOperator
from .operator_validation import OperatorValidator


class Hamiltonian:
    """
    Orchestrates the assembly of the full Quantum Hamiltonian H = T + V.
    Provides strict Hermiticity validation and matrix formatting.
    """

    def __init__(self, kinetic_op: KineticEnergyOperator, potential_op: PotentialEnergyOperator):
        # Prevent addition of incompatible physical systems
        OperatorValidator.validate_unit_consistency(kinetic_op.unit_system, potential_op.unit_system)
        
        self.kinetic = kinetic_op
        self.potential = potential_op

    def build_sparse(self, accuracy_order: int = 2, interior_only: bool = False) -> sparse.csr_matrix:
        """
        Constructs and returns the sparse Hamiltonian matrix H = T + V.
        Validates dimensions and Hermiticity before returning.
        """
        t_mat = self.kinetic.build_matrix(accuracy_order=accuracy_order, interior_only=interior_only)
        v_mat = self.potential.build_matrix(interior_only=interior_only)
        
        if t_mat.shape != v_mat.shape:
            raise ValueError(f"Matrix dimension mismatch: Kinetic {t_mat.shape} != Potential {v_mat.shape}")
            
        h_mat = t_mat + v_mat
        
        # Verify physical validity (Symmetry)
        OperatorValidator.check_hermiticity(h_mat)
        
        return h_mat

    def build_dense(self, accuracy_order: int = 2, interior_only: bool = False) -> np.ndarray:
        """
        Constructs and returns the dense representation of the Hamiltonian.
        Useful for analytical debugging or small-scale full diagonalization.
        """
        sparse_h = self.build_sparse(accuracy_order=accuracy_order, interior_only=interior_only)
        return sparse_h.toarray()
