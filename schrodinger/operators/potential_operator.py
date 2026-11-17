import numpy as np
from scipy import sparse
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential


class PotentialEnergyOperator:
    """
    Constructs the diagonal sparse matrix representation of the potential energy V(x).
    """

    def __init__(self, grid: SpatialGrid, potential: Potential):
        self.grid = grid
        self.potential = potential
        self.unit_system = potential.unit_system

    def build_matrix(self, interior_only: bool = False) -> sparse.csr_matrix:
        """
        Constructs the potential energy matrix.
        
        Args:
            interior_only (bool): If True, computes only over the interior grid points.
            
        Returns:
            sparse.csr_matrix: Diagonal matrix containing the evaluated potential.
        """
        x_points = self.grid.interior_points if interior_only else self.grid.grid
        v_array = self.potential(x_points)
        
        # Guard against complex potentials (Hamiltonian must be Hermitian/Real)
        if np.iscomplexobj(v_array) and np.any(np.imag(v_array) != 0):
            raise ValueError("Potential evaluation returned imaginary components. Potential must be strictly real.")
            
        # Build sparse diagonal matrix
        return sparse.diags(v_array, offsets=0, format="csr", dtype=self.grid.grid.dtype)
