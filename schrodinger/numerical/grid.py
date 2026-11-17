import numpy as np
from enum import Enum, auto
from typing import Tuple
from schrodinger.potentials.base import Potential


class Precision(Enum):
    """Supported numerical precisions for grid arrays."""
    FLOAT32 = np.float32
    FLOAT64 = np.float64


class BoundaryCondition(Enum):
    """Supported mathematical boundary conditions."""
    DIRICHLET = auto()
    NEUMANN = auto()


class SpatialGrid:
    """
    Represents a 1D uniform numerical grid for spatial discretization.
    Handles domain boundaries, precision, and spatial steps.
    """

    def __init__(
        self, 
        x_min: float, 
        x_max: float, 
        points: int, 
        precision: Precision = Precision.FLOAT64,
        boundary_condition: BoundaryCondition = BoundaryCondition.DIRICHLET
    ):
        """
        Initializes the spatial grid and precomputes the coordinate array.
        
        Args:
            x_min (float): The left boundary of the spatial domain.
            x_max (float): The right boundary of the spatial domain.
            points (int): Total number of grid points (N). Must be >= 3.
            precision (Precision): Float32 or Float64.
            boundary_condition (BoundaryCondition): Type of boundary condition.
        """
        self._validate_parameters(x_min, x_max, points)
        
        self._x_min = float(x_min)
        self._x_max = float(x_max)
        self._points = int(points)
        self._precision = precision
        self._boundary_condition = boundary_condition
        
        # Calculate grid spacing (dx)
        self._dx = (self._x_max - self._x_min) / (self._points - 1)
        
        # Generate the uniform grid
        self._grid = np.linspace(
            self._x_min, 
            self._x_max, 
            self._points, 
            dtype=self._precision.value
        )

    @staticmethod
    def _validate_parameters(x_min: float, x_max: float, points: int) -> None:
        """Validates grid boundaries and resolution."""
        if x_max <= x_min:
            raise ValueError(f"Invalid domain boundaries: x_max ({x_max}) must be strictly greater than x_min ({x_min}).")
        if points < 3:
            raise ValueError(f"Insufficient grid resolution: points ({points}) must be at least 3 to have interior nodes.")

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def x_max(self) -> float:
        return self._x_max

    @property
    def points(self) -> int:
        return self._points

    @property
    def dx(self) -> float:
        """The spatial step size between adjacent points."""
        return self._dx

    @property
    def grid(self) -> np.ndarray:
        """The full discrete position array."""
        return self._grid

    @property
    def interior_points(self) -> np.ndarray:
        """Returns only the interior grid points (excluding endpoints)."""
        return self._grid[1:-1]

    @property
    def boundary_points(self) -> Tuple[float, float]:
        """Returns the left and right boundary coordinates."""
        return float(self._grid[0]), float(self._grid[-1])
        
    @property
    def boundary_condition(self) -> BoundaryCondition:
        return self._boundary_condition

    def evaluate_potential(self, potential: Potential) -> np.ndarray:
        """
        Evaluates a given quantum potential across the entire grid.
        
        Args:
            potential (Potential): An instantiated potential object.
            
        Returns:
            np.ndarray: The potential energy evaluated at every grid point.
        """
        v_array = potential(self._grid)
        # Ensure the output matches the grid's numerical precision
        return v_array.astype(self._precision.value)
