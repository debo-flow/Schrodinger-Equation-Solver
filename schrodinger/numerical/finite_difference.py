import numpy as np
from scipy import sparse
from enum import Enum
from typing import Tuple
from .grid import SpatialGrid, BoundaryCondition


class Scheme(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


class FiniteDifferenceEngine:
    """
    Constructs numerical differentiation approximations and sparse operators.
    Supports adjustable accuracy orders and integrates natively with grid boundaries.
    """

    def __init__(self, grid: SpatialGrid):
        self.grid = grid
        self.dx = grid.dx
        self.n = grid.points

    def _get_coefficients(self, derivative_order: int, accuracy_order: int) -> Tuple[list, list, float]:
        """Retrieves the diagonals, offsets, and multiplier for the specified operator."""
        if derivative_order == 1:
            if accuracy_order == 2:
                return ([-1, 0, 1], [-1, 0, 1], 1.0 / (2.0 * self.dx))
            elif accuracy_order == 4:
                return ([1, -8, 0, 8, -1], [-2, -1, 0, 1, 2], 1.0 / (12.0 * self.dx))
        elif derivative_order == 2:
            if accuracy_order == 2:
                return ([1, -2, 1], [-1, 0, 1], 1.0 / self.dx**2)
            elif accuracy_order == 4:
                return ([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], 1.0 / (12.0 * self.dx**2))
        
        raise ValueError(f"Unsupported configuration: derivative={derivative_order}, accuracy={accuracy_order}")

    def build_differentiation_matrix(
        self, 
        derivative_order: int = 2, 
        accuracy_order: int = 2, 
        interior_only: bool = False
    ) -> sparse.csr_matrix:
        """
        Constructs a sparse matrix representation of the differential operator.
        
        Args:
            derivative_order (int): 1 for first derivative, 2 for second derivative.
            accuracy_order (int): 2 for O(dx^2), 4 for O(dx^4).
            interior_only (bool): If True, returns an (N-2 x N-2) matrix. Ideal for Dirichlet TISE.
            
        Returns:
            sparse.csr_matrix: The constructed sparse differentiation matrix.
        """
        diagonals, offsets, coeff = self._get_coefficients(derivative_order, accuracy_order)
        matrix_size = self.n - 2 if interior_only else self.n
        
        # Build base diagonals
        diag_arrays = [
            coeff * np.ones(matrix_size - abs(off)) * val 
            for val, off in zip(diagonals, offsets)
        ]
        mat = sparse.diags(diag_arrays, offsets, shape=(matrix_size, matrix_size), format='lil')
        
        # Apply boundary condition modifiers if operating on the full grid
        if not interior_only:
            if derivative_order == 1:
                # Use one-sided differences at boundaries to maintain O(dx^2) accuracy
                if accuracy_order == 2:
                    mat[0, 0:3] = np.array([-3, 4, -1]) * coeff
                    mat[-1, -3:] = np.array([1, -4, 3]) * coeff
                    
            elif derivative_order == 2:
                if self.grid.boundary_condition == BoundaryCondition.NEUMANN:
                    # Enforce Neumann condition via ghost point reflection (psi_{-1} = psi_1)
                    if accuracy_order == 2:
                        mat[0, 1] = 2.0 * coeff
                        mat[-1, -2] = 2.0 * coeff
                    elif accuracy_order == 4:
                        mat[0, 1] = 32.0 * coeff
                        mat[0, 2] = -2.0 * coeff
                        mat[-1, -2] = 32.0 * coeff
                        mat[-1, -3] = -2.0 * coeff
                        
                elif self.grid.boundary_condition == BoundaryCondition.DIRICHLET:
                    # For a full-grid Dirichlet matrix, we zero out the physical boundary rows
                    # and insert 1 on the diagonal to decouple them from eigenvalue calculations.
                    mat[0, :] = 0.0
                    mat[-1, :] = 0.0
                    mat[0, 0] = 1.0
                    mat[-1, -1] = 1.0

        return mat.tocsr()

    def forward_difference(self, array: np.ndarray) -> np.ndarray:
        """First derivative using O(dx) forward difference scheme."""
        if array.shape[0] != self.n:
            raise ValueError("Array size does not match grid points.")
        df = np.zeros_like(array)
        df[:-1] = (array[1:] - array[:-1]) / self.dx
        df[-1] = (array[-1] - array[-2]) / self.dx  # Fallback to backward at right edge
        return df

    def backward_difference(self, array: np.ndarray) -> np.ndarray:
        """First derivative using O(dx) backward difference scheme."""
        if array.shape[0] != self.n:
            raise ValueError("Array size does not match grid points.")
        df = np.zeros_like(array)
        df[1:] = (array[1:] - array[:-1]) / self.dx
        df[0] = (array[1] - array[0]) / self.dx  # Fallback to forward at left edge
        return df
