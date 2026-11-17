from scipy import sparse
from typing import Literal
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from .operator_validation import OperatorValidator


class KineticEnergyOperator:
    """
    Constructs the sparse matrix representation of the kinetic energy operator T.
    T = -(hbar^2 / 2m) * (d^2/dx^2)
    """

    def __init__(
        self, 
        fd_engine: FiniteDifferenceEngine, 
        mass: float = 1.0, 
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        OperatorValidator.validate_physical_constants(mass, hbar)
        
        self.fd_engine = fd_engine
        self.mass = mass
        self.hbar = hbar
        self.unit_system = unit_system
        self.prefactor = -(self.hbar**2) / (2.0 * self.mass)

    def build_matrix(self, accuracy_order: int = 2, interior_only: bool = False) -> sparse.csr_matrix:
        """
        Constructs the kinetic energy matrix using finite differences.
        
        Args:
            accuracy_order (int): 2 for O(dx^2), 4 for O(dx^4).
            interior_only (bool): If True, returns (N-2 x N-2) matrix for Dirichlet boundaries.
            
        Returns:
            sparse.csr_matrix: The sparse kinetic energy matrix.
        """
        d2_matrix = self.fd_engine.build_differentiation_matrix(
            derivative_order=2, 
            accuracy_order=accuracy_order, 
            interior_only=interior_only
        )
        # Sparse scalar multiplication
        return d2_matrix.multiply(self.prefactor)
