import pytest
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, Precision, BoundaryCondition
from schrodinger.potentials.models import HarmonicOscillator


def test_grid_initialization_and_spacing():
    """Test basic grid generation and dx calculation."""
    grid = SpatialGrid(x_min=0.0, x_max=10.0, points=11)
    
    assert grid.x_min == 0.0
    assert grid.x_max == 10.0
    assert grid.points == 11
    assert grid.dx == 1.0  # (10 - 0) / 10
    
    np.testing.assert_array_almost_equal(
        grid.grid, 
        np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    )

def test_grid_validation_errors():
    """Ensure invalid domains and insufficient resolutions are rejected."""
    # Invalid boundaries
    with pytest.raises(ValueError, match="strictly greater than"):
        SpatialGrid(x_min=5.0, x_max=0.0, points=100)
    with pytest.raises(ValueError, match="strictly greater than"):
        SpatialGrid(x_min=5.0, x_max=5.0, points=100)
        
    # Insufficient points (needs at least 3 for interior logic)
    with pytest.raises(ValueError, match="at least 3"):
        SpatialGrid(x_min=0.0, x_max=1.0, points=2)

def test_interior_and_boundary_extraction():
    """Test the separation of interior points and boundaries."""
    grid = SpatialGrid(x_min=-1.0, x_max=1.0, points=5)
    
    boundaries = grid.boundary_points
    assert boundaries == (-1.0, 1.0)
    
    interior = grid.interior_points
    assert len(interior) == 3
    np.testing.assert_array_almost_equal(interior, np.array([-0.5, 0.0, 0.5]))

def test_numerical_precision():
    """Test configuration and enforcement of float32/float64 datatypes."""
    grid_64 = SpatialGrid(x_min=0, x_max=1, points=10, precision=Precision.FLOAT64)
    assert grid_64.grid.dtype == np.float64
    
    grid_32 = SpatialGrid(x_min=0, x_max=1, points=10, precision=Precision.FLOAT32)
    assert grid_32.grid.dtype == np.float32

def test_potential_evaluation_integration():
    """Test that the grid correctly evaluates a Potential object over its points."""
    grid = SpatialGrid(x_min=-2.0, x_max=2.0, points=5, precision=Precision.FLOAT32)
    ho = HarmonicOscillator(mass=2.0, omega=1.0) # k = 2.0
    
    # Eval should be 0.5 * k * x^2 = 1.0 * x^2
    v_eval = grid.evaluate_potential(ho)
    
    assert isinstance(v_eval, np.ndarray)
    assert len(v_eval) == 5
    assert v_eval.dtype == np.float32
    
    # Expected: x = [-2, -1, 0, 1, 2] -> v = [4, 1, 0, 1, 4]
    np.testing.assert_array_almost_equal(v_eval, np.array([4.0, 1.0, 0.0, 1.0, 4.0]))

def test_boundary_condition_flag():
    """Test that boundary condition enums are stored properly."""
    grid = SpatialGrid(x_min=0, x_max=1, points=10, boundary_condition=BoundaryCondition.NEUMANN)
    assert grid.boundary_condition == BoundaryCondition.NEUMANN
