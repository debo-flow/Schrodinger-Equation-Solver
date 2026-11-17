import numpy as np
from scipy import sparse
from schrodinger.numerical.grid import SpatialGrid

class PositionOperator:
    """Constructs position-space matrices."""

    @staticmethod
    def build_x(grid: SpatialGrid) -> sparse.csr_matrix:
        """Constructs the position operator x as a sparse diagonal matrix."""
        return sparse.diags(grid.grid, offsets=0, format='csr')

    @staticmethod
    def build_x_squared(grid: SpatialGrid) -> sparse.csr_matrix:
        """Constructs the x^2 operator as a sparse diagonal matrix."""
        return sparse.diags(grid.grid**2, offsets=0, format='csr')
