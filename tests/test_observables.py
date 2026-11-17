import pytest
import numpy as np
from scipy import sparse, integrate
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine
from schrodinger.observables import (
    ProbabilityEngine, PositionOperator, MomentumOperator,
    ObservableEngine, CommutatorEngine, UncertaintyAnalyzer, QuantumStatistics
)

@pytest.fixture
def grid():
    return SpatialGrid(x_min=0.0, x_max=np.pi, points=500, boundary_condition=BoundaryCondition.DIRICHLET)

@pytest.fixture
def fd_engine(grid):
    return FiniteDifferenceEngine(grid)

@pytest.fixture
def ground_state_wf(grid):
    # Normalized ground state of Infinite Square Well [0, pi]
    psi = np.sqrt(2.0 / np.pi) * np.sin(grid.grid)
    return Wavefunction(x=grid.grid, psi=psi, energy=1.0)

def test_probability_engine(ground_state_wf):
    # Prob from 0 to pi/2 should be exactly 0.5 by symmetry
    prob_half = ProbabilityEngine.interval_probability(ground_state_wf, 0.0, np.pi / 2.0)
    assert np.isclose(prob_half, 0.5, rtol=1e-4)

    # CDF ending should be 1.0
    cdf = ProbabilityEngine.cumulative_probability(ground_state_wf)
    assert np.isclose(cdf[-1], 1.0)
    assert np.isclose(cdf[len(cdf)//2], 0.5, rtol=1e-2)

def test_position_momentum_expectation(ground_state_wf, grid, fd_engine):
    x_op = PositionOperator.build_x(grid)
    p_op = MomentumOperator.build_p(fd_engine, hbar=1.0)

    exp_x = ObservableEngine.expectation(x_op, ground_state_wf)
    exp_p = ObservableEngine.expectation(p_op, ground_state_wf)

    # <x> = pi/2
    assert np.isclose(exp_x, np.pi / 2.0, rtol=1e-4)
    # <p> = 0 for a stationary bound state
    assert np.isclose(exp_p, 0.0, atol=1e-4)

def test_uncertainty_and_heisenberg(ground_state_wf, grid, fd_engine):
    x_op = PositionOperator.build_x(grid)
    x2_op = PositionOperator.build_x_squared(grid)
    p_op = MomentumOperator.build_p(fd_engine)
    p2_op = MomentumOperator.build_p_squared(fd_engine)

    delta_x = UncertaintyAnalyzer.calculate_uncertainty(x_op, x2_op, ground_state_wf)
    delta_p = UncertaintyAnalyzer.calculate_uncertainty(p_op, p2_op, ground_state_wf)

    # Verify Heisenberg: dx * dp >= hbar / 2 (0.5)
    assert UncertaintyAnalyzer.verify_heisenberg(delta_x, delta_p, hbar=1.0)
    
    product = delta_x * delta_p
    assert product >= 0.5

def test_commutator_xp(grid, fd_engine):
    hbar = 1.0
    x_op = PositionOperator.build_x(grid)
    p_op = MomentumOperator.build_p(fd_engine, hbar=hbar)

    # Calculate [x, p] = xp - px
    xp_comm = CommutatorEngine.evaluate_matrix(x_op, p_op)
    
    # In theory, [x, p] = i * hbar * I
    # We check interior points to avoid boundary finite-difference artifacts
    interior_diag = xp_comm.diagonal()[5:-5]
    expected = 1j * hbar
    
    np.testing.assert_allclose(interior_diag, expected, atol=1e-3)

def test_quantum_statistics_report(ground_state_wf, grid):
    stats_engine = QuantumStatistics(grid, hbar=1.0)
    report = stats_engine.analyze_state(ground_state_wf)
    
    assert report.heisenberg_satisfied is True
    assert np.isclose(report.expectation_x, np.pi / 2.0, rtol=1e-4)
    assert np.isclose(report.expectation_p, 0.0, atol=1e-4)
    assert report.uncertainty_product >= 0.5
