import json
import math
import pytest
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.models import InfiniteSquareWell, HarmonicOscillator
from schrodinger.solvers.solver_configuration import SolverConfig, SolverMethod
from schrodinger.optimization import OptimizationEngine, OptimizationConfig

from schrodinger.coupled import (
    CouplingMatrix,
    CoupledPotential,
    TimeDependentCoupledPotential,
    MultiComponentWavefunction,
    CoupledHamiltonian,
    CoupledEigenvalueSolver,
    ChannelPopulationAnalyzer,
    CoupledEvolutionConfig,
    CoupledTimeEvolution,
    CoupledQuantumSystem
)


def test_two_channel_zero_coupling_and_uncoupled_limit():
    """Test Case 1 & 8: Two-channel system with zero coupling reproduces sorted single-channel union."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    ho_1 = HarmonicOscillator(mass=1.0, omega=1.0)  # E_n = 0.5, 1.5, 2.5, ...
    ho_2 = HarmonicOscillator(mass=1.0, omega=1.6)  # E_m = 0.8, 2.4, 4.0, ...

    # Single-channel reference energies
    opt = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))
    e1 = opt.solve_system(grid, ho_1, num_eigenvalues=3).energies
    e2 = opt.solve_system(grid, ho_2, num_eigenvalues=3).energies
    expected_union = np.sort(np.concatenate([e1, e2]))[:4]

    coupled_pot = CoupledPotential.from_channels([ho_1, ho_2], couplings=None)
    ham = CoupledHamiltonian(grid, coupled_pot, mass=1.0, hbar=1.0)
    solver = CoupledEigenvalueSolver(SolverConfig(num_eigenvalues=4))
    res = solver.solve(ham, include_uncoupled_comparison=True)

    np.testing.assert_allclose(res.energies, expected_union, atol=1e-10)
    assert res.uncoupled_comparison is not None
    assert res.uncoupled_comparison.max_energy_shift < 1e-12

    # Each uncoupled eigenstate must reside 100% in one channel and 0% in the other
    for st in res.eigenstates:
        pops = st.channel_populations
        assert np.isclose(np.sum(pops), 1.0, atol=1e-6)
        assert np.isclose(np.max(pops), 1.0, atol=1e-6)
        assert np.isclose(np.min(pops), 0.0, atol=1e-6)


def test_two_channel_constant_coupling_analytical_splitting():
    """Test Case 2 & 4: Identical wells with constant real/complex coupling split by E_n +/- |Delta|."""
    L = math.pi
    grid = SpatialGrid(x_min=0.0, x_max=L, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    well = InfiniteSquareWell(width=L, center=0.5 * L)

    # Complex Hermitian coupling Delta = 0.3 * exp(i * 0.8), |Delta| = 0.3
    delta = 0.3 * np.exp(1j * 0.8)
    coupled_pot = CoupledPotential.from_channels([well, well], couplings={(0, 1): delta})

    # Base ISW energies for L=pi, m=1, hbar=1: E_1 = 0.5, E_2 = 2.0
    # Coupled lowest 4 levels: 0.5 - 0.3 = 0.2, 0.5 + 0.3 = 0.8, 2.0 - 0.3 = 1.7, 2.0 + 0.3 = 2.3
    expected_energies = np.array([0.2, 0.8, 1.7, 2.3])

    system = CoupledQuantumSystem(grid, coupled_pot, mass=1.0, hbar=1.0)
    res = system.solve_eigenstates(num_eigenvalues=4)

    np.testing.assert_allclose(res.energies, expected_energies, atol=1e-3)
    assert res.max_residual < 1e-8

    # Symmetric/antisymmetric dressed states have 50% / 50% channel populations
    for st in res.eigenstates:
        np.testing.assert_allclose(st.channel_populations, [0.5, 0.5], atol=1e-5)


def test_position_dependent_coupling_and_three_channel_system():
    """Test Case 3 & multi-channel support: Gaussian position-dependent coupling across 3 channels."""
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=200, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)

    # Localized Gaussian coupling V_{0,1}(x) = 0.5 * exp(-x^2) and V_{1,2}(x) = 0.4 * exp(-x^2)
    gauss_01 = lambda x: 0.5 * np.exp(-(x ** 2))
    gauss_12 = lambda x: 0.4 * np.exp(-(x ** 2))

    coupled_pot = CoupledPotential.from_channels(
        [ho, ho, ho],
        couplings={(0, 1): gauss_01, (1, 2): gauss_12}
    )

    # Test sparse eigensolver on 3-channel block matrix (3 * 198 = 594 > sparse_threshold=500)
    solver = CoupledEigenvalueSolver(SolverConfig(num_eigenvalues=4, method=SolverMethod.SPARSE))
    ham = CoupledHamiltonian(grid, coupled_pot)
    res = solver.solve(ham)

    assert res.num_channels == 3
    assert res.method_used == "sparse"
    assert res.max_residual < 1e-6
    # Ground state energy must be depressed below uncoupled E_0 = 0.5 by attractive hybridization
    assert res.energies[0] < 0.5
    assert res.eigenstates[0].is_orthogonal_to(res.eigenstates[1], tolerance=1e-5)


def test_non_hermitian_potential_rejection_and_invalid_dimensions():
    """Test Case 5: Reject non-Hermitian coupling matrices, complex diagonals, and mismatched shapes."""
    cm = CouplingMatrix(num_channels=2)
    cm.set_diagonal(0, 0.0)
    cm.set_diagonal(1, 0.0)
    # Explicitly break Hermiticity: V_01 = 1.0, V_10 = 0.2
    cm.set_coupling(0, 1, 1.0, auto_conjugate=False)
    cm.set_coupling(1, 0, 0.2, auto_conjugate=False)

    with pytest.raises(ValueError, match="not Hermitian"):
        cm.evaluate(np.linspace(-1.0, 1.0, 10))

    # Complex diagonal potential rejected
    with pytest.raises(ValueError, match="strictly real"):
        cm.set_diagonal(0, 1.0 + 0.5j)

    # Incompatible channel count in composition rejected
    p2 = CoupledPotential.from_channels([HarmonicOscillator(), HarmonicOscillator()])
    p3 = CoupledPotential.from_channels([HarmonicOscillator(), HarmonicOscillator(), HarmonicOscillator()])
    with pytest.raises(ValueError, match="Channel count mismatch"):
        _ = p2 + p3


def test_complete_state_normalization_and_superposition():
    """Test Case 6 & 7: Complete state normalization applies across all channels jointly."""
    grid = SpatialGrid(x_min=0.0, x_max=math.pi, points=200)
    x = grid.grid

    # Construct raw unnormalized components: psi_0 = sin(x), psi_1 = sin(x)
    raw = np.vstack([np.sin(x), np.sin(x)])
    mc_wf = MultiComponentWavefunction(x=x, components=raw, energy=0.5, normalize=True)

    # Total norm must be 1.0, while each channel has population 0.5 (not normalized independently to 1!)
    assert np.isclose(mc_wf.total_norm, 1.0, atol=1e-6)
    np.testing.assert_allclose(mc_wf.channel_populations, [0.5, 0.5], atol=1e-6)

    pop_report = ChannelPopulationAnalyzer.analyze_state(mc_wf)
    assert pop_report.bounds_valid is True
    assert pop_report.conservation_error < 1e-6

    # Orthogonal state [sin(x), -sin(x)]
    orth_wf = MultiComponentWavefunction(x=x, components=np.vstack([np.sin(x), -np.sin(x)]), normalize=True)
    assert mc_wf.is_orthogonal_to(orth_wf, tolerance=1e-6)
    assert ChannelPopulationAnalyzer.transition_probability(mc_wf, orth_wf) < 1e-10

    # Superposition preserves complete-state normalization
    sup = MultiComponentWavefunction.superposition([mc_wf, orth_wf], [1.0, 1.0j])
    assert np.isclose(sup.total_norm, 1.0, atol=1e-6)


def test_time_dependent_rabi_oscillations_and_probability_conservation():
    """Test Case 9: Coupled time evolution reproduces exact Rabi oscillations P_1(t)=cos^2(Delta*t), P_2(t)=sin^2(Delta*t)."""
    L = math.pi
    grid = SpatialGrid(x_min=0.0, x_max=L, points=150, boundary_condition=BoundaryCondition.DIRICHLET)
    well = InfiniteSquareWell(width=L, center=0.5 * L)

    delta = 0.8
    coupled_pot = CoupledPotential.from_channels([well, well], couplings={(0, 1): delta})

    # Initialize 100% in channel 0 ground state: psi_0(x) = sqrt(2/pi) * sin(x), psi_1(x) = 0
    psi_g = np.sqrt(2.0 / L) * np.sin(grid.grid)
    init_state = MultiComponentWavefunction(
        x=grid.grid,
        components=np.vstack([psi_g, np.zeros_like(psi_g)]),
        energy=0.5
    )

    t_end = 1.2
    evo = CoupledTimeEvolution(
        grid=grid,
        coupled_potential=coupled_pot,
        mass=1.0,
        hbar=1.0,
        config=CoupledEvolutionConfig(t_end=t_end, dt=2e-3, method="crank_nicolson", save_every=25)
    )
    evo_res = evo.propagate(init_state)

    # Total probability across both channels must remain 1.0 at every step
    assert evo_res.max_norm_deviation < 1e-10
    assert evo_res.channel_trajectory.bounds_valid is True

    # Verify exact Rabi oscillation populations P_0(t) = cos^2(delta * t), P_1(t) = sin^2(delta * t)
    expected_p0 = np.cos(delta * evo_res.times) ** 2
    expected_p1 = np.sin(delta * evo_res.times) ** 2
    np.testing.assert_allclose(evo_res.channel_trajectory.populations_over_time[:, 0], expected_p0, atol=1e-3)
    np.testing.assert_allclose(evo_res.channel_trajectory.populations_over_time[:, 1], expected_p1, atol=1e-3)


def test_time_dependent_coupling_potential_and_unified_workflow(tmp_path):
    """Test time-dependent coupling V_01(x, t) and end-to-end CoupledQuantumSystem JSON report."""
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=150, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)

    # Time-dependent coupling pulse: V_01(x, t) = 0.5 * sin(2 * t)
    td_pot = TimeDependentCoupledPotential(num_channels=2)
    td_pot.set_term(0, 0, ho)
    td_pot.set_term(1, 1, ho)
    td_pot.set_term(0, 1, lambda x, t: 0.5 * math.sin(2.0 * t) * np.ones_like(x), auto_conjugate=True)

    system = CoupledQuantumSystem.from_channels(
        grid=grid,
        channel_potentials=[ho, ho],
        couplings={(0, 1): 0.15},
        channel_labels=["ground_band", "excited_band"]
    )
    static_res = system.solve_eigenstates(num_eigenvalues=2)

    # Start purely in channel 0 using the spatial profile of channel 0
    ch0_profile = static_res.eigenstates[0].channel(0)
    init_state = MultiComponentWavefunction(
        x=grid.grid,
        components=np.vstack([ch0_profile, np.zeros_like(ch0_profile)]),
        energy=0.5,
        channel_labels=["ground_band", "excited_band"]
    )

    workflow_res = system.run_workflow(
        num_eigenvalues=4,
        initial_state=init_state,
        t_end=1.0,
        dt=2e-3,
        evolution_method="crank_nicolson",
        time_dependent_potential=td_pot
    )

    assert workflow_res.validation_report.validation_passed is True
    assert workflow_res.evolution_result is not None
    assert workflow_res.evolution_result.max_norm_deviation < 1e-10
    # Time-dependent coupling must transfer population into channel 1
    assert workflow_res.evolution_result.final_state.channel_populations[1] > 0.05

    # Verify JSON export
    report_path = tmp_path / "coupled_validation.json"
    saved = workflow_res.validation_report.export_json(report_path)
    assert saved.exists()
    parsed = json.loads(saved.read_text(encoding="utf-8"))
    assert parsed["validation_passed"] is True
    assert parsed["num_channels"] == 2
    assert len(parsed["energy_eigenvalues"]) == 4
