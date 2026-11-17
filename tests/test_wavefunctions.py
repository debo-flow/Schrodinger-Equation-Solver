import pytest
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.wavefunctions import (
    PhaseManager, NormalizationEngine, IntegrationMethod,
    ProbabilityDensity, QuantumObservables, OrthogonalityAnalyzer,
    Wavefunction, WavefunctionReconstructor
)

@pytest.fixture
def grid():
    return SpatialGrid(x_min=0.0, x_max=np.pi, points=1000)

@pytest.fixture
def unnormalized_sin(grid):
    # Sin(x) in [0, pi]
    return np.sin(grid.grid)

@pytest.fixture
def complex_state(grid):
    # A state with an arbitrary phase
    psi = np.sin(grid.grid) * np.exp(1j * 0.7)
    return psi

def test_normalization_engine(grid, unnormalized_sin):
    # The integral of sin^2(x) from 0 to pi is pi/2
    # So the norm should be sqrt(pi/2)
    norm = NormalizationEngine.calculate_norm(unnormalized_sin, grid.grid)
    expected_norm = np.sqrt(np.pi / 2.0)
    assert np.isclose(norm, expected_norm, rtol=1e-4)
    
    normalized_psi, _ = NormalizationEngine.normalize(unnormalized_sin, grid.grid)
    
    # Verify it integrates to 1.0
    assert NormalizationEngine.verify_normalization(normalized_psi, grid.grid)

def test_phase_manager(complex_state):
    # State has a 0.7 rad global phase
    standardized = PhaseManager.standardize_phase(complex_state)
    
    # The maximum amplitude point should now be strictly real and positive
    max_idx = np.argmax(np.abs(standardized))
    assert np.isclose(np.imag(standardized[max_idx]), 0.0)
    assert np.real(standardized[max_idx]) > 0.0

def test_probability_density():
    psi = np.array([0.5, 0.5j, -0.5, -0.5j])
    density = ProbabilityDensity.calculate(psi)
    np.testing.assert_array_almost_equal(density, [0.25, 0.25, 0.25, 0.25])

def test_quantum_observables(grid):
    # For a normalized particle in a box (0 to pi), ground state sin(x)
    # <x> = pi/2
    psi, _ = NormalizationEngine.normalize(np.sin(grid.grid), grid.grid)
    
    exp_x = QuantumObservables.expectation_x(psi, grid.grid)
    assert np.isclose(exp_x, np.pi / 2.0, rtol=1e-4)
    
    # Check node counting
    ground = np.sin(grid.grid)
    first_excited = np.sin(2 * grid.grid)
    
    assert QuantumObservables.count_nodes(ground) == 0
    assert QuantumObservables.count_nodes(first_excited) == 1

def test_orthogonality(grid):
    psi_1, _ = NormalizationEngine.normalize(np.sin(grid.grid), grid.grid)
    psi_2, _ = NormalizationEngine.normalize(np.sin(2 * grid.grid), grid.grid)
    
    # Different eigenstates of the box should be orthogonal
    assert OrthogonalityAnalyzer.check_orthogonality(psi_1, psi_2, grid.grid)
    
    # A state is not orthogonal to itself
    assert not OrthogonalityAnalyzer.check_orthogonality(psi_1, psi_1, grid.grid)

def test_wavefunction_object(grid):
    psi, _ = NormalizationEngine.normalize(np.sin(grid.grid), grid.grid)
    wf = Wavefunction(x=grid.grid, psi=psi, energy=1.0)
    
    assert wf.energy == 1.0
    assert wf.nodes == 0
    assert np.isclose(wf.expectation_x, np.pi / 2.0, rtol=1e-4)
    
    # Test rejection of unnormalized state
    with pytest.raises(ValueError, match="not continuously normalized"):
        Wavefunction(x=grid.grid, psi=np.sin(grid.grid), energy=1.0)

def test_reconstruction(grid):
    reconstructor = WavefunctionReconstructor(grid)
    
    # Mock solver output for 2 states
    # Note: solver outputs are columns of a 2D array
    raw_psi_1 = np.sin(grid.grid)
    raw_psi_2 = np.sin(2 * grid.grid)
    wavefunctions_2d = np.column_stack((raw_psi_1, raw_psi_2))
    
    solver_output = {
        "energies": [1.0, 4.0],
        "wavefunctions": wavefunctions_2d
    }
    
    physical_states = reconstructor.reconstruct(solver_output)
    
    assert len(physical_states) == 2
    assert isinstance(physical_states[0], Wavefunction)
    assert physical_states[0].energy == 1.0
    assert physical_states[1].energy == 4.0
    
    # Ensure they were normalized during reconstruction
    assert NormalizationEngine.verify_normalization(physical_states[0].psi, physical_states[0].x)
