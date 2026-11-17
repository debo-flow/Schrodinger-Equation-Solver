import pytest
import numpy as np
from schrodinger.potentials.models import (
    FreeParticle, InfiniteSquareWell, FiniteSquareWell,
    HarmonicOscillator, DoubleWell
)
from schrodinger.potentials.base import Potential

def test_base_unit_validation():
    """Ensure invalid unit systems are rejected at instantiation."""
    with pytest.raises(ValueError):
        HarmonicOscillator(unit_system="imperial")
        
def test_input_validation():
    """Ensure non-finite inputs are safely caught."""
    fp = FreeParticle()
    with pytest.raises(ValueError):
        fp(np.array([0.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        fp(np.inf)

def test_scalar_vs_array_evaluation():
    """Ensure calling with scalar returns scalar, array returns array."""
    ho = HarmonicOscillator(mass=1.0, omega=1.0, center=0.0)
    
    scalar_res = ho(2.0)
    assert isinstance(scalar_res, float)
    assert scalar_res == 2.0  # 0.5 * 1.0 * 2.0^2
    
    arr_res = ho(np.array([-2.0, 2.0]))
    assert isinstance(arr_res, np.ndarray)
    np.testing.assert_array_equal(arr_res, np.array([2.0, 2.0]))

def test_free_particle():
    """Test mathematical correctness of Free Particle."""
    fp = FreeParticle()
    x = np.linspace(-10, 10, 50)
    v = fp(x)
    assert np.all(v == 0.0)

def test_infinite_square_well():
    """Test boundaries and parameter validation of Infinite Square Well."""
    with pytest.raises(ValueError):
        InfiniteSquareWell(width=-1.0)
        
    isw = InfiniteSquareWell(width=2.0, center=0.0)
    assert isw(0.0) == 0.0
    assert isw(-0.99) == 0.0
    assert isw(1.01) == np.inf
    assert isw(-1.01) == np.inf

def test_finite_square_well():
    """Test boundaries and depth of Finite Square Well."""
    with pytest.raises(ValueError):
        FiniteSquareWell(depth=-5.0, width=2.0)
        
    fsw = FiniteSquareWell(depth=10.0, width=2.0, center=1.0)
    # Inside well
    assert fsw(1.0) == -10.0
    assert fsw(2.0) == -10.0
    assert fsw(0.0) == -10.0
    # Outside well
    assert fsw(-0.01) == 0.0
    assert fsw(2.01) == 0.0

def test_harmonic_oscillator():
    """Test mathematical correctness of Harmonic Oscillator."""
    with pytest.raises(ValueError):
        HarmonicOscillator(mass=-1.0)
        
    ho = HarmonicOscillator(mass=2.0, omega=3.0, center=1.0) # k = 2 * 9 = 18
    assert ho(1.0) == 0.0
    assert ho(2.0) == 9.0  # 0.5 * 18 * (1)^2
    assert ho(0.0) == 9.0  # 0.5 * 18 * (-1)^2

def test_double_well():
    """Test minima and barrier behavior of Double Well."""
    with pytest.raises(ValueError):
        DoubleWell(barrier_height=-1.0, minima_offset=1.0)
        
    dw = DoubleWell(barrier_height=5.0, minima_offset=2.0, center=0.0)
    
    # Center barrier
    assert dw(0.0) == 5.0
    # Minima at center +/- offset
    assert dw(2.0) == 0.0
    assert dw(-2.0) == 0.0
    # Far away (should grow positive)
    assert dw(4.0) == 45.0 # 5.0 * ((4/2)^2 - 1)^2 = 5 * (3)^2 = 45
