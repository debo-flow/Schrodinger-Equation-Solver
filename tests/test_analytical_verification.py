import json
import math
import pytest
import numpy as np
from scipy import integrate

from schrodinger.potentials.models import InfiniteSquareWell, HarmonicOscillator
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.operators import KineticEnergyOperator, PotentialEnergyOperator, Hamiltonian
from schrodinger.solvers import EigenvalueSolver, SolverConfig
from schrodinger.wavefunctions import Wavefunction, WavefunctionReconstructor

from schrodinger.verification import (
    AnalyticalSolver,
    InfiniteSquareWellAnalytical,
    HarmonicOscillatorAnalytical,
    FreeParticleAnalytical,
    ErrorAnalyzer,
    NumericalComparisonEngine,
    ConvergenceAnalyzer,
    ValidationTolerances,
    ScientificValidator
)


def _solve_numerical_system(potential, x_min, x_max, points, num_states=3, mass=1.0, hbar=1.0):
    grid = SpatialGrid(x_min=x_min, x_max=x_max, points=points, boundary_condition=BoundaryCondition.DIRICHLET)
    fd_engine = FiniteDifferenceEngine(grid)
    kin_op = KineticEnergyOperator(fd_engine, mass=mass, hbar=hbar)
    pot_op = PotentialEnergyOperator(grid, potential)
    ham = Hamiltonian(kin_op, pot_op)
    h_mat = ham.build_sparse(accuracy_order=2, interior_only=True)
    res = EigenvalueSolver(SolverConfig(num_eigenvalues=num_states)).solve(h_mat)
    return WavefunctionReconstructor(grid).reconstruct(res)


def test_analytical_registry_factory():
    models = AnalyticalSolver.available_models()
    assert "infinite_square_well" in models
    assert "harmonic_oscillator" in models
    assert "free_particle" in models

    isw = AnalyticalSolver.create("infinite_square_well", width=2.0)
    assert isinstance(isw, InfiniteSquareWellAnalytical)

    with pytest.raises(ValueError, match="Unknown analytical model"):
        AnalyticalSolver.create("morse_potential")


def test_infinite_square_well_analytical_exactness():
    L = 2.0
    m = 1.5
    hbar = 1.0
    isw = InfiniteSquareWellAnalytical(width=L, x_min=0.0, mass=m, hbar=hbar)

    # Verify E_n = n^2 * pi^2 * hbar^2 / (2 * m * L^2)
    for n in [1, 2, 3]:
        expected_e = (n ** 2 * math.pi ** 2 * hbar ** 2) / (2.0 * m * L ** 2)
        assert np.isclose(isw.energy(n), expected_e, rtol=1e-14)

    x = np.linspace(0.0, L, 1001)
    psi_1 = isw.wavefunction(1, x)
    psi_2 = isw.wavefunction(2, x)

    # Normalization & Orthogonality
    assert np.isclose(integrate.simpson(y=psi_1 ** 2, x=x), 1.0, atol=1e-6)
    assert np.isclose(integrate.simpson(y=psi_1 * psi_2, x=x), 0.0, atol=1e-6)
    assert isw.verify_boundary_conditions(1, x)

    with pytest.raises(ValueError):
        isw.energy(0)


def test_harmonic_oscillator_analytical_exactness():
    omega = 2.5
    m = 1.2
    hbar = 1.0
    qho = HarmonicOscillatorAnalytical(omega=omega, center=0.5, mass=m, hbar=hbar)

    # Verify E_n = hbar * omega * (n + 1/2)
    spectrum = qho.energy_spectrum(4)
    expected = np.array([hbar * omega * (n + 0.5) for n in range(4)])
    np.testing.assert_allclose(spectrum, expected, rtol=1e-14)

    x = np.linspace(-8.0, 9.0, 2001)
    psi_0 = qho.wavefunction(0, x)
    psi_1 = qho.wavefunction(1, x)
    psi_2 = qho.wavefunction(2, x)

    # Normalization & Orthogonality
    assert np.isclose(integrate.simpson(y=psi_0 ** 2, x=x), 1.0, atol=1e-6)
    assert np.isclose(integrate.simpson(y=psi_2 ** 2, x=x), 1.0, atol=1e-6)
    assert np.isclose(integrate.simpson(y=psi_0 * psi_1, x=x), 0.0, atol=1e-6)
    assert qho.verify_boundary_conditions(0, x, tolerance=1e-6)

    # High-n normalization should not overflow
    assert np.isfinite(qho.normalization_constant(50))

    with pytest.raises(ValueError):
        qho.energy(-1)


def test_free_particle_plane_wave_reference():
    L = 10.0
    fp = FreeParticleAnalytical(box_length=L, mass=2.0, hbar=1.0)

    k = 1.5
    assert np.isclose(fp.momentum_from_k(k), 1.5)
    assert np.isclose(fp.energy_from_k(k), (1.5 ** 2) / 4.0)

    x = np.linspace(0.0, L, 1001)
    psi_mode_2 = fp.wavefunction(2, x)

    # Box normalization: integral |psi|^2 dx == 1
    assert np.isclose(integrate.simpson(y=np.abs(psi_mode_2) ** 2, x=x), 1.0, atol=1e-6)
    assert fp.verify_boundary_conditions(2, x)

    # Dirac normalization amplitude
    psi_dirac = fp.plane_wave(k, x, normalization="dirac")
    np.testing.assert_allclose(np.abs(psi_dirac) ** 2, 1.0 / (2.0 * math.pi), rtol=1e-12)


def test_error_analyzer_and_near_zero_handling():
    # Near-zero analytical reference handling
    assert ErrorAnalyzer.relative_error(0.0, 0.0) == 0.0
    assert ErrorAnalyzer.relative_error(1e-3, 0.0) is None
    assert np.isclose(ErrorAnalyzer.relative_error(1.05, 1.0), 0.05)

    num_arr = np.array([1.01, 0.0, 2.02, 1e-4])
    ana_arr = np.array([1.00, 0.0, 2.00, 0.0])
    rel_arr = ErrorAnalyzer.relative_error(num_arr, ana_arr)

    assert np.isclose(rel_arr[0], 0.01)
    assert rel_arr[1] == 0.0
    assert np.isclose(rel_arr[2], 0.01)
    assert np.isnan(rel_arr[3])

    with pytest.raises(ValueError, match="Shape mismatch"):
        ErrorAnalyzer.rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_phase_independent_wavefunction_comparison():
    x = np.linspace(0.0, 1.0, 501)
    isw = InfiniteSquareWellAnalytical(width=1.0, x_min=0.0)
    psi_exact = isw.wavefunction(1, x)

    # Apply an arbitrary complex global phase factor e^(i * 2.1)
    rotated_psi = psi_exact * np.exp(1j * 2.1)
    wf_rotated = Wavefunction(x=x, psi=rotated_psi, energy=isw.energy(1))

    result = NumericalComparisonEngine.compare_state(wf_rotated, isw, quantum_number=1)

    assert result.wavefunction_l2_error < 1e-12
    assert result.probability_density_rmse < 1e-12
    assert np.isclose(result.state_fidelity, 1.0, atol=1e-12)


def test_numerical_vs_analytical_infinite_well():
    L = 1.0
    potential = InfiniteSquareWell(width=L, center=0.5)
    analytical = InfiniteSquareWellAnalytical(width=L, x_min=0.0)

    wfs = _solve_numerical_system(potential, x_min=0.0, x_max=L, points=300, num_states=4)
    comparison = NumericalComparisonEngine.compare_spectrum(wfs, analytical)

    assert comparison.max_energy_relative_error < 1e-3
    assert comparison.max_wavefunction_l2_error < 1e-2
    assert comparison.max_orthogonality_error < 1e-5


def test_numerical_vs_analytical_harmonic_oscillator():
    potential = HarmonicOscillator(mass=1.0, omega=1.0, center=0.0)
    analytical = HarmonicOscillatorAnalytical(mass=1.0, omega=1.0, center=0.0)

    wfs = _solve_numerical_system(potential, x_min=-8.0, x_max=8.0, points=400, num_states=4)
    comparison = NumericalComparisonEngine.compare_spectrum(wfs, analytical)

    assert comparison.max_energy_relative_error < 2e-3
    assert comparison.max_wavefunction_l2_error < 2e-2
    assert comparison.max_orthogonality_error < 1e-5


def test_grid_convergence_order_second_order_fd():
    L = 1.0
    potential = InfiniteSquareWell(width=L, center=0.5)
    analytical = InfiniteSquareWellAnalytical(width=L, x_min=0.0)

    conv = ConvergenceAnalyzer.run_study(
        potential=potential,
        analytical_solution=analytical,
        x_min=0.0,
        x_max=L,
        grid_resolutions=[50, 100, 200],
        num_states=2,
        accuracy_order=2
    )

    assert conv.is_monotonically_converging is True
    assert conv.is_numerically_stable is True
    # 2nd-order central difference must exhibit convergence order ~ 2.0
    assert conv.estimated_energy_order is not None
    assert 1.9 < conv.estimated_energy_order < 2.1


def test_scientific_validation_report_and_json_export(tmp_path):
    potential = HarmonicOscillator(mass=1.0, omega=1.0, center=0.0)
    analytical = HarmonicOscillatorAnalytical(mass=1.0, omega=1.0, center=0.0)

    wfs = _solve_numerical_system(potential, x_min=-8.0, x_max=8.0, points=300, num_states=3)
    conv = ConvergenceAnalyzer.run_study(
        potential=potential,
        analytical_solution=analytical,
        x_min=-8.0,
        x_max=8.0,
        grid_resolutions=[100, 200, 300],
        num_states=2,
        accuracy_order=2
    )

    validator = ScientificValidator(ValidationTolerances())
    report = validator.validate(wfs, analytical, convergence_result=conv)

    assert report.validation_passed is True
    assert report.potential_model == "HarmonicOscillator"
    assert len(report.warnings) == 0

    # Verify JSON serialization and file export
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["validation_passed"] is True
    assert parsed["grid_resolution"] == 300
    assert parsed["convergence_info"]["is_numerically_stable"] is True

    output_file = tmp_path / "validation_report.json"
    saved_path = report.export_json(output_file)
    assert saved_path.exists()
    assert json.loads(saved_path.read_text(encoding="utf-8"))["potential_model"] == "HarmonicOscillator"
