import pytest
import numpy as np
from scipy.sparse import issparse
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine

@pytest.fixture
def grid():
    return SpatialGrid(x_min=0.0, x_max=2*np.pi, points=1000)

@pytest.fixture
def engine(grid):
    return FiniteDifferenceEngine(grid)

def test_forward_backward_difference(engine, grid):
    """Test 1st derivative array approximations against analytical sin(x) -> cos(x)."""
    x = grid.grid
    f = np.sin(x)
    analytical_df = np.cos(x)
    
    forward_df = engine.forward_difference(f)
    backward_df = engine.backward_difference(f)
    
    # O(dx) schemes have higher errors, but should generally match the curve
    np.testing.assert_allclose(forward_df[:-1], analytical_df[:-1], atol=0.01)
    np.testing.assert_allclose(backward_df[1:], analytical_df[1:], atol=0.01)

def test_central_difference_matrix_accuracy(engine, grid):
    """Test 2nd order central difference matrix against analytical sin(x) -> -sin(x)."""
    x = grid.grid
    f = np.sin(x)
    analytical_d2f = -np.sin(x)
    
    mat = engine.build_differentiation_matrix(derivative_order=2, accuracy_order=2)
    assert issparse(mat)
    
    d2f = mat.dot(f)
    
    # Exclude boundary points for Dirichlet comparison
    np.testing.assert_allclose(d2f[1:-1], analytical_d2f[1:-1], atol=1e-4)

def test_polynomial_exactness():
    """Test that 2nd order is exact for x^2, and 4th order is exact for x^4."""
    grid = SpatialGrid(x_min=-1.0, x_max=1.0, points=100)
    engine = FiniteDifferenceEngine(grid)
    x = grid.grid
    
    # Test x^2 -> 2
    f2 = x**2
    mat_2nd = engine.build_differentiation_matrix(derivative_order=2, accuracy_order=2)
    d2_f2 = mat_2nd.dot(f2)
    np.testing.assert_allclose(d2_f2[2:-2], 2.0, atol=1e-10)
    
    # Test x^4 -> 12x^2 (Requires 4th order accuracy)
    f4 = x**4
    mat_4th = engine.build_differentiation_matrix(derivative_order=2, accuracy_order=4)
    d2_f4 = mat_4th.dot(f4)
    np.testing.assert_allclose(d2_f4[3:-3], 12 * x[3:-3]**2, atol=1e-9)

def test_numerical_convergence():
    """Verify that truncation error drops by factor of ~4 for O(dx^2) when dx is halved."""
    def get_error(points):
        g = SpatialGrid(x_min=0.0, x_max=1.0, points=points)
        eng = FiniteDifferenceEngine(g)
        x = g.grid
        f = np.exp(x)
        d2f_analytical = np.exp(x)
        
        mat = eng.build_differentiation_matrix(derivative_order=2, accuracy_order=2, interior_only=True)
        d2f_num = mat.dot(f[1:-1])
        return np.max(np.abs(d2f_num - d2f_analytical[1:-1]))

    error_coarse = get_error(100)
    error_fine = get_error(200) # dx halved
    
    ratio = error_coarse / error_fine
    # For O(dx^2), halving dx should divide error by 4
    assert 3.8 < ratio < 4.2

def test_interior_only_flag(engine):
    """Test the reduction of the matrix size for Dirichlet interior calculations."""
    mat_full = engine.build_differentiation_matrix(derivative_order=2, interior_only=False)
    mat_interior = engine.build_differentiation_matrix(derivative_order=2, interior_only=True)
    
    assert mat_full.shape == (1000, 1000)
    assert mat_interior.shape == (998, 998)

def test_neumann_ghost_points():
    """Verify that Neumann boundary logic applies the correct ghost point reflection."""
    grid = SpatialGrid(x_min=0.0, x_max=1.0, points=10, boundary_condition=BoundaryCondition.NEUMANN)
    engine = FiniteDifferenceEngine(grid)
    
    mat = engine.build_differentiation_matrix(derivative_order=2, accuracy_order=2)
    coeff = 1.0 / grid.dx**2
    
    # Boundary row 0 should be [-2, 2, 0, ...]
    assert np.isclose(mat[0, 0], -2.0 * coeff)
    assert np.isclose(mat[0, 1], 2.0 * coeff)
