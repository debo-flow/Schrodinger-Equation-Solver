import json
import math
import pytest
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.models import HarmonicOscillator, InfiniteSquareWell
from schrodinger.potentials.step import StepPotential
from schrodinger.coupled import CoupledPotential, CoupledQuantumSystem
from schrodinger.optimization import OptimizationEngine, OptimizationConfig

from schrodinger.perturbation import (
    PerturbationOperator,
    PerturbationMatrix,
    NonDegeneratePerturbationSolver,
    WavefunctionCorrectionEngine,
    DegeneratePerturbationSolver,
    VariationalSolver,
    RayleighRitzSolver,
    MonomialPotential,
    HarmonicOscillatorPerturbationBenchmarks,
    InfiniteWellPerturbationBenchmark,
    PerturbationConvergenceAnalyzer,
    PerturbationSystem,
    ApproximationValidationEngine
)


def test_zero_perturbation_limit():
    """Test 1: At lambda = 0, perturbed energies and wavefunctions equal unperturbed states exactly."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    h_prime = MonomialPotential(power=4, coeff=1.0)

    sys = PerturbationSystem.from_single_channel(
        grid, ho, h_prime, lambda_strength=0.0, num_unperturbed_states=6
    )
    nd_res = sys.solve_non_degenerate(target_states=[0, 1, 2], order=2)
    wf_res = sys.correct_wavefunctions(target_states=[0, 1])

    for r in nd_res:
        assert np.isclose(r.total_approximate_energy, r.unperturbed_energy, atol=1e-14)
        assert r.is_perturbatively_valid is True
    for wr in wf_res:
        assert wr.correction_norm == 0.0
        assert np.isclose(wr.overlap_probability, 1.0, atol=1e-12)


def test_harmonic_oscillator_linear_and_quadratic_benchmarks():
    """Test 2, 3 & 9: Verify 1st- and 2nd-order corrections for linear (exact) and quadratic QHO perturbations."""
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=350, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    bench = HarmonicOscillatorPerturbationBenchmarks(mass=1.0, omega=1.0, hbar=1.0)

    # 1. Linear perturbation H' = alpha * x (alpha = 0.2): E_n^(1) = 0, E_n^(2) = -alpha^2 / 2 is exact
    alpha = 0.2
    lin_sys = PerturbationSystem.from_single_channel(
        grid, ho, MonomialPotential(power=1, coeff=1.0), lambda_strength=alpha, num_unperturbed_states=10
    )
    lin_res = lin_sys.solve_non_degenerate(target_states=[0, 1, 2], order=2)
    exact_lin = lin_sys.solve_exact_numerical(num_states=3)

    for n, r in enumerate(lin_res):
        ref = bench.linear_perturbation(n, alpha)
        assert np.isclose(r.first_order_correction, 0.0, atol=1e-5)
        assert np.isclose((alpha ** 2) * r.second_order_correction, ref["E2"], atol=1e-3)
        assert np.isclose(r.total_approximate_energy, exact_lin[n], atol=1e-3)

    # 2. Quadratic perturbation H' = gamma * x^2 (gamma = 0.05)
    gamma = 0.05
    quad_sys = PerturbationSystem.from_single_channel(
        grid, ho, MonomialPotential(power=2, coeff=1.0), lambda_strength=gamma, num_unperturbed_states=10
    )
    quad_res = quad_sys.solve_non_degenerate(target_states=[0, 1, 2], order=2)
    exact_quad = quad_sys.solve_exact_numerical(num_states=3)

    for n, r in enumerate(quad_res):
        ref = bench.quadratic_perturbation(n, gamma)
        assert np.isclose(gamma * r.first_order_correction, ref["E1"], atol=2e-3)
        assert np.isclose((gamma ** 2) * r.second_order_correction, ref["E2"], atol=2e-3)
        assert np.isclose(r.total_approximate_energy, exact_quad[n], atol=1e-3)


def test_quartic_anharmonic_and_cubic_benchmarks_and_convergence():
    """Test 9 & 10: Quartic anharmonic oscillator, cubic stability metadata, and lambda scan."""
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=350, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    bench = HarmonicOscillatorPerturbationBenchmarks(mass=1.0, omega=1.0, hbar=1.0)

    beta = 0.02
    q_sys = PerturbationSystem.from_single_channel(
        grid, ho, MonomialPotential(power=4, coeff=1.0), lambda_strength=beta, num_unperturbed_states=14
    )
    q_res = q_sys.solve_non_degenerate(target_states=[0, 1], order=2)
    exact_q = q_sys.solve_exact_numerical(num_states=2)

    for n, r in enumerate(q_res):
        ref = bench.quartic_perturbation(n, beta)
        assert np.isclose(beta * r.first_order_correction, ref["E1"], atol=2e-3)
        assert np.isclose((beta ** 2) * r.second_order_correction, ref["E2"], atol=3e-3)
        # Second-order must be closer to exact numerical than first-order for small beta
        err1 = abs((r.unperturbed_energy + beta * r.first_order_correction) - exact_q[n])
        err2 = abs(r.total_approximate_energy - exact_q[n])
        assert err2 < err1
        assert err2 < 1e-3

    # Cubic perturbation benchmark exposes stability metadata and accurate local E_0^(2)
    cubic_info = bench.cubic_perturbation(0, kappa=0.05)
    assert cubic_info["is_globally_bounded"] is False
    assert np.isclose(cubic_info["E1"], 0.0)
    assert cubic_info["E2"] < 0.0

    # Lambda convergence scan verifies 2nd-order error scales ~ O(lambda^3)
    analyzer = PerturbationConvergenceAnalyzer()
    conv = analyzer.scan_lambda(
        state_index=0,
        unperturbed_energies=q_sys.unperturbed_energies,
        perturbation_matrix=q_sys.perturbation_matrix,
        lambda_values=[0.005, 0.01, 0.02, 0.05],
        exact_energy_fn=lambda lam: float(q_sys.solve_exact_numerical(num_states=1, lambda_strength=lam)[0])
    )
    assert conv.points[0].second_order_improves_over_first is True
    assert conv.estimated_second_order_scaling is not None
    assert 2.5 < conv.estimated_second_order_scaling < 3.5


def test_wavefunction_correction_and_infinite_square_well():
    """Test 4 & ISW benchmark: First-order wavefunction correction and weak tilt inside ISW."""
    L = math.pi
    grid = SpatialGrid(x_min=0.0, x_max=L, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    isw = InfiniteSquareWell(width=L, center=0.5 * L)
    # Weak half-well step potential inside [0, pi]
    step = StepPotential(step_position=0.5 * L, left_height=1.0, right_height=0.0)
    isw_bench = InfiniteWellPerturbationBenchmark(width=L, mass=1.0, hbar=1.0)

    lam = 0.2
    sys = PerturbationSystem.from_single_channel(
        grid, isw, step, lambda_strength=lam, num_unperturbed_states=10
    )
    nd_res = sys.solve_non_degenerate(target_states=[0, 1], order=2)
    assert np.isclose(nd_res[0].first_order_correction, isw_bench.half_well_step_first_order(1, 1.0), atol=1e-2)

    wf_corr_list = sys.correct_wavefunctions(target_states=[0, 1])
    wf0_corr = wf_corr_list[0]
    assert wf0_corr.orthogonality_to_unperturbed_error < 1e-5
    assert 0.9 < wf0_corr.overlap_probability < 1.0
    # Step is higher on left half [0, pi/2], pushing ground-state <x> to the right (> pi/2)
    assert wf0_corr.corrected_state.expectation_x > 0.5 * L

    # Orthogonality across corrected states holds to O(lambda^2)
    cross_err = WavefunctionCorrectionEngine.max_cross_orthogonality_error(
        [wr.corrected_state for wr in wf_corr_list]
    )
    assert cross_err < 1e-2


def test_small_denominator_detection():
    """Test 11: NonDegeneratePerturbationSolver detects near-degenerate denominators and warns."""
    energies = [1.0, 1.0 + 1e-8, 3.0]
    w_mat = PerturbationMatrix(np.array([[0.0, 0.5, 0.1], [0.5, 0.0, 0.1], [0.1, 0.1, 0.0]]))

    solver = NonDegeneratePerturbationSolver(denominator_tolerance=1e-5)
    res = solver.solve_state(0, energies, w_mat, lambda_strength=0.1, order=2)

    assert res.small_denominator_detected is True
    assert res.is_perturbatively_valid is False
    assert any("Small energy denominator" in msg for msg in res.warnings)


def test_degenerate_perturbation_and_complex_hermitian_coupled_splitting():
    """Test 5, 6 & 12: Degenerate perturbation theory on 2-channel coupled system with complex coupling."""
    L = math.pi
    grid = SpatialGrid(x_min=0.0, x_max=L, points=200, boundary_condition=BoundaryCondition.DIRICHLET)
    well = InfiniteSquareWell(width=L, center=0.5 * L)

    # Unperturbed 2-channel identical wells -> every level is 2-fold degenerate!
    unpert_sys = CoupledQuantumSystem.from_channels(grid, [well, well], couplings=None)

    # Complex Hermitian off-diagonal coupling perturbation: V'_01 = 0.5 * exp(i * 1.1)
    delta = 0.5 * np.exp(1j * 1.1)
    pert_pot = CoupledPotential.from_channels([0.0, 0.0], couplings={(0, 1): delta})
    lam = 0.4

    pert_sys = PerturbationSystem.from_coupled_system(
        unpert_sys, pert_pot, lambda_strength=lam, num_unperturbed_states=4
    )
    assert pert_sys.perturbation_matrix.is_complex is True

    deg_results = pert_sys.solve_degenerate(degeneracy_tolerance=1e-5)
    assert len(deg_results) == 2  # Ground doublet (0, 1) and first excited doublet (2, 3)

    # Exact splitting for complex coupling: +/- lambda * |delta| = +/- 0.2
    ground_doublet = deg_results[0]
    assert ground_doublet.degeneracy_lifted is True
    np.testing.assert_allclose(ground_doublet.first_order_shifts, [-0.2, 0.2], atol=1e-6)

    # Verify against exact numerical coupled eigenvalues
    exact_coupled = pert_sys.solve_exact_numerical(num_states=4)
    np.testing.assert_allclose(ground_doublet.split_energies, exact_coupled[:2], atol=1e-5)

    # Lifted eigenstates exhibit 50% / 50% perturbative channel mixing
    for lifted_state in ground_doublet.corrected_eigenstates:
        np.testing.assert_allclose(lifted_state.channel_populations, [0.5, 0.5], atol=1e-5)


def test_variational_solver_upper_bound_and_accuracy():
    """Test 7: Variational solver recovers exact QHO ground state and respects E[psi] >= E_0."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=300, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    solver = VariationalSolver(grid, ho, mass=1.0, hbar=1.0)

    # Gaussian trial state psi(x; a) = exp(-a * x^2) -> exact optimum at a = 0.5, E_0 = 0.5
    def gaussian_trial(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        a = params[0]
        if a <= 0.01:
            return np.zeros_like(x)
        return np.exp(-a * (x ** 2))

    res = solver.optimize(
        gaussian_trial,
        initial_params=[1.5],
        method="Nelder-Mead",
        compare_with_numerical=True
    )

    assert res.converged is True
    assert res.upper_bound_verified is True
    assert np.isclose(res.optimal_parameters[0], 0.5, atol=1e-2)
    assert np.isclose(res.optimal_energy, 0.5, atol=1e-3)
    assert res.reference_ground_energy is not None
    assert res.optimal_energy >= res.reference_ground_energy - 1e-7


def test_rayleigh_ritz_non_orthogonal_basis_and_convergence():
    """Test 8: Rayleigh-Ritz solves H c = E S c for non-orthogonal Gaussians and converges monotonically."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    # Quartic anharmonic oscillator V(x) = 0.5 * x^2 + 0.25 * x^4
    anharm = HarmonicOscillator(mass=1.0, omega=1.0) + MonomialPotential(power=4, coeff=0.25)

    # Non-orthogonal shifted/scaled Gaussian basis functions
    basis = [
        lambda x: np.exp(-0.5 * (x ** 2)),
        lambda x: np.exp(-1.0 * (x ** 2)),
        lambda x: np.exp(-0.25 * (x ** 2)),
        lambda x: (x ** 2) * np.exp(-0.7 * (x ** 2)),
        lambda x: x * np.exp(-0.6 * (x ** 2)),
        lambda x: (x ** 3) * np.exp(-0.8 * (x ** 2))
    ]

    rr_solver = RayleighRitzSolver(grid, anharm, mass=1.0, hbar=1.0)
    conv = rr_solver.convergence_study(basis, basis_sizes=[1, 2, 4, 6], track_states=2)

    assert conv.is_monotonically_nonincreasing is True
    assert conv.final_result.is_orthogonal_basis is False
    assert conv.final_result.max_generalized_residual < 1e-10

    # Compare converged Rayleigh-Ritz ground state against exact numerical eigenvalue
    opt = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False))
    num_e = opt.solve_system(grid, anharm, num_eigenvalues=2).energies
    assert conv.final_result.energies[0] >= num_e[0] - 1e-7
    assert abs(conv.final_result.energies[0] - num_e[0]) < 2e-3

    # Linearly dependent basis must be rejected
    with pytest.raises(ValueError, match="linearly dependent"):
        rr_solver.solve([basis[0], basis[0]])


def test_approximation_validation_engine_json_export(tmp_path):
    """Test 12: Structured JSON report generation for perturbation, variational, and Rayleigh-Ritz."""
    grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=250, boundary_condition=BoundaryCondition.DIRICHLET)
    ho = HarmonicOscillator(mass=1.0, omega=1.0)
    pert_sys = PerturbationSystem.from_single_channel(
        grid, ho, MonomialPotential(power=4, coeff=1.0), lambda_strength=0.01, num_unperturbed_states=10
    )

    validator = ApproximationValidationEngine(energy_error_tolerance=1e-2)
    report = validator.validate_perturbation_system(pert_sys, target_states=[0, 1], order=2)

    assert report.validation_passed is True
    assert report.max_absolute_error is not None and report.max_absolute_error < 1e-3

    out_file = tmp_path / "perturbation_report.json"
    saved = report.export_json(out_file)
    assert saved.exists()
    parsed = json.loads(saved.read_text(encoding="utf-8"))
    assert parsed["validation_passed"] is True
    assert parsed["approximation_order"] == 2
    assert len(parsed["total_approximate_energies"]) == 2
