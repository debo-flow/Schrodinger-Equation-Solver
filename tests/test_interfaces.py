"""
Tests for the core abstract base classes to ensure architectural integrity.
Validates that interfaces cannot be instantiated directly and that proper 
subclassing correctly implements the required methods.
"""

import pytest
import numpy as np
from schrodinger.core.interfaces import (
    Wavefunction, Potential, Hamiltonian, 
    EigenvalueSolver, NumericalEngine, QuantumSystem
)

def test_abstract_instantiation_fails():
    """Ensure ABCs prevent direct instantiation without implementations."""
    with pytest.raises(TypeError):
        Wavefunction()
    with pytest.raises(TypeError):
        Potential()
    with pytest.raises(TypeError):
        Hamiltonian()
    with pytest.raises(TypeError):
        EigenvalueSolver()
    with pytest.raises(TypeError):
        NumericalEngine()
    with pytest.raises(TypeError):
        QuantumSystem()

def test_dummy_potential_implementation():
    """Test that a valid implementation of an interface functions properly."""
    
    class HarmonicOscillator(Potential):
        def __init__(self, omega: float = 1.0, mass: float = 1.0):
            self.k = mass * omega**2
            
        def __call__(self, x: np.ndarray) -> np.ndarray:
            return 0.5 * self.k * x**2
            
    x_grid = np.array([-1.0, 0.0, 1.0])
    v = HarmonicOscillator()
    
    # Check return types and values
    v_out = v(x_grid)
    assert isinstance(v_out, np.ndarray)
    np.testing.assert_array_almost_equal(v_out, [0.5, 0.0, 0.5])

def test_dummy_engine_implementation():
    """Test standard numerical engine contract."""
    
    class BasicGrid(NumericalEngine):
        def discretize_space(self, x_min: float, x_max: float, points: int) -> np.ndarray:
            return np.linspace(x_min, x_max, points)
            
    engine = BasicGrid()
    grid = engine.discretize_space(0, 10, 5)
    
    assert len(grid) == 5
    assert grid[0] == 0.0
    assert grid[-1] == 10.0
