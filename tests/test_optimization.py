import json
import pytest
import numpy as np
from scipy import sparse

from schrodinger.numerical.grid import SpatialGrid, Precision, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.models import HarmonicOscillator, InfiniteSquareWell, FiniteSquareWell
from schrodinger.operators import KineticEnergyOperator, PotentialEnergyOperator, Hamiltonian
from schrodinger.solvers import EigenvalueSolver, SolverConfig, SolverMethod
from schrodinger.wavefunctions import WavefunctionReconstructor
from schrodinger.verification import (
    HarmonicOscillatorAnalytical,
    NumericalComparisonEngine,
    ErrorAnalyzer
)
from schrodinger.optimization import (
    ComputationCache,
    PerformanceMonitor,
    OptimizedSolverMethod,
    SolverSelectionCriteria,
    SolverSelector,
    NumericalStabilityAnalyzer,
    AdaptiveGridConfig,
    AdaptiveGridFramework,
    ParallelExecutionEngine,
    OptimizationConfig,
    OptimizationEngine,
    BenchmarkEngine
)


def test_numerical_equivalence_all_optimized_backends():
    """Verify TRIDIAGONAL, DENSE, SPARSE_SHIFT_INVERT, and SPARSE_DIRECT match baseline."""
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=200, boundary_condition=BoundaryCondition.DIRICHLET)
    potential = HarmonicOscillator(mass=1.0, omega=1.0)

    # Baseline solution
    fd = FiniteDifferenceEngine(grid)
    ham = Hamiltonian(KineticEnergyOperator(fd), PotentialEnergyOperator(grid, potential))
    h_base = ham.build_sparse(accuracy_order=2, interior_only=True)
    base_res = EigenvalueSolver(SolverConfig(num_eigenvalues=4)).solve(h_base)
    base_wfs = WavefunctionReconstructor(grid).reconstruct(base_res)

    for method in (
        OptimizedSolverMethod.TRIDIAGONAL,
        OptimizedSolverMethod.DENSE,
        OptimizedSolverMethod.SPARSE_SHIFT_INVERT,
        OptimizedSolverMethod.SPARSE_DIRECT
    ):
        engine = OptimizationEngine(OptimizationConfig(forced_method=method, enable_cache=False))
        opt_res = engine.solve_system(grid, potential, num_eigenvalues=4)

        assert opt_res.method_used == method
        np.testing.assert_allclose(opt_res.energies, base_res["energies"], atol=1e-7)

        for wf_b, wf_o in zip(base_wfs, opt_res.wavefunctions):
            aligned, _ = NumericalComparisonEngine.align_global_phase(wf_o.psi, wf_b.psi, grid.grid)
            l2_err = ErrorAnalyzer.l2_wavefunction_error(aligned, wf_b.psi, grid.grid)
            assert l2_err < 1e-6


def test_fourth_order_banded_equivalence():
    """Verify 4th-order BANDED solver matches 4th-order baseline Hamiltonian."""
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=150, boundary_condition=BoundaryCondition.DIRICHLET)
    potential = HarmonicOscillator(mass=1.0, omega=1.0)

    fd = FiniteDifferenceEngine(grid)
    ham = Hamiltonian(KineticEnergyOperator(fd), PotentialEnergyOperator(grid, potential))
    h_base = ham.build_sparse(accuracy_order=4, interior_only=True)
    base_res = EigenvalueSolver(SolverConfig(num_eigenvalues=3)).solve(h_base)

    engine = OptimizationEngine(OptimizationConfig(accuracy_order=4, enable_cache=False))
    opt_res = engine.solve_system(grid, potential, num_eigenvalues=3)

    assert opt_res.method_used == OptimizedSolverMethod.BANDED
    np.testing.assert_allclose(opt_res.energies, base_res["energies"], atol=1e-7)


def test_computation_cache_hit_miss_and_invalidation():
    """Verify caching reuses Hamiltonians and invalidates automatically when parameters change."""
    cache = ComputationCache(max_entries=16)
    engine = OptimizationEngine(OptimizationConfig(enable_cache=True), cache=cache)

    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=120)
    pot_1 = HarmonicOscillator(mass=1.0, omega=1.0)
    pot_2 = HarmonicOscillator(mass=1.0, omega=2.0)

    # First run -> miss
    res_1 = engine.solve_system(grid, pot_1, num_eigenvalues=3)
    assert res_1.cache_hit is False

    # Identical parameters -> hit
    res_2 = engine.solve_system(grid, pot_1, num_eigenvalues=3)
    assert res_2.cache_hit is True
    np.testing.assert_array_equal(res_1.energies, res_2.energies)

    # Changed potential parameter -> automatic cache miss (new fingerprint)
    res_3 = engine.solve_system(grid, pot_2, num_eigenvalues=3)
    assert res_3.cache_hit is False
    assert not np.allclose(res_1.energies, res_3.energies)

    # Explicit namespace invalidation
    removed = cache.invalidate("hamiltonian")
    assert removed >= 2
    res_4 = engine.solve_system(grid, pot_1, num_eigenvalues=3)
    assert res_4.cache_hit is False


def test_solver_selector_routing():
    """Verify intelligent solver routing across structures, sizes, and memory budgets."""
    # Real symmetric tridiagonal -> TRIDIAGONAL
    rec_tri = SolverSelector.select(
        SolverSelectionCriteria(dimension=500, num_eigenvalues=5, is_symmetric_tridiagonal=True, sparsity_ratio=0.99)
    )
    assert rec_tri.method == OptimizedSolverMethod.TRIDIAGONAL

    # 4th-order banded -> BANDED
    rec_band = SolverSelector.select(
        SolverSelectionCriteria(dimension=800, num_eigenvalues=5, bandwidth=2, sparsity_ratio=0.99)
    )
    assert rec_band.method == OptimizedSolverMethod.BANDED

    # Complex Hermitian small matrix -> DENSE
    rec_complex = SolverSelector.select(
        SolverSelectionCriteria(dimension=100, num_eigenvalues=5, is_complex=True, sparsity_ratio=0.2)
    )
    assert rec_complex.method == OptimizedSolverMethod.DENSE

    # Large sparse matrix exceeding memory for dense -> SPARSE_SHIFT_INVERT
    rec_large = SolverSelector.select(
        SolverSelectionCriteria(dimension=5000, num_eigenvalues=5, sparsity_ratio=0.999, available_memory_mb=64.0)
    )
    assert rec_large.method == OptimizedSolverMethod.SPARSE_SHIFT_INVERT

    # Requesting all eigenvalues -> DENSE
    rec_full = SolverSelector.select(
        SolverSelectionCriteria(dimension=50, num_eigenvalues=50, sparsity_ratio=0.9)
    )
    assert rec_full.method == OptimizedSolverMethod.DENSE


def test_performance_monitor_reporting():
    """Verify stage timing, memory tracking, and JSON serialization."""
    monitor = PerformanceMonitor(track_memory=True)
    monitor.start()
    with monitor.stage("matrix_construction"):
        arr = np.ones((200, 200), dtype=np.float64)
        _ = arr.sum()
    with monitor.stage("eigenvalue_computation"):
        _ = np.linalg.norm(arr)
    monitor.stop()

    report = monitor.build_report()
    assert report.total_time_seconds > 0.0
    assert report.matrix_construction_time > 0.0
    assert report.eigenvalue_computation_time > 0.0
    assert report.peak_memory_bytes >= 0

    parsed = json.loads(report.to_json())
    assert "total_time_seconds" in parsed
    assert "peak_memory_mb" in parsed


def test_numerical_stability_and_precision_sensitivity():
    """Verify conditioning checks, invalid matrix detection, and float32 vs float64 analysis."""
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, points=150)
    potential = HarmonicOscillator(mass=1.0, omega=1.0)
    engine = OptimizationEngine(OptimizationConfig(enable_cache=False, run_stability_check=True))

    res = engine.solve_system(grid, potential, num_eigenvalues=3)
    assert res.stability_report is not None
    assert res.stability_report.is_stable is True
    assert res.stability_report.diagnostics is not None
    assert res.stability_report.diagnostics.converged is True

    # Invalid NaN matrix detection
    bad_mat = sparse.csr_matrix(np.array([[1.0, np.nan], [np.nan, 2.0]]))
    bad_report = NumericalStabilityAnalyzer.analyze(bad_mat)
    assert bad_report.is_stable is False
    assert bad_report.is_finite is False
    assert len(bad_report.recovery_recommendations) > 0

    # Precision sensitivity check
    sens = NumericalStabilityAnalyzer.analyze_precision_sensitivity(
        potential, x_min=-6.0, x_max=6.0, points=150, num_states=2
    )
    assert np.isfinite(sens.max_energy_difference)
    assert sens.recommended_precision in ("float32", "float64")


def test_adaptive_grid_bounded_refinement():
    """Verify adaptive refinement improves accuracy and strictly respects caps."""
    potential = HarmonicOscillator(mass=1.0, omega=1.0)
    analytical = HarmonicOscillatorAnalytical(mass=1.0, omega=1.0)
    engine = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False, run_stability_check=False))

    config = AdaptiveGridConfig(
        min_points=40,
        max_points=300,
        max_refinement_steps=4,
        refinement_factor=1.6,
        energy_convergence_tolerance=1e-3
    )
    adaptive = AdaptiveGridFramework(config)
    coarse_grid = SpatialGrid(x_min=-7.0, x_max=7.0, points=45)

    def _solve(g: SpatialGrid):
        return engine.solve_system(g, potential, num_eigenvalues=3).wavefunctions

    result = adaptive.refine_until_converged(coarse_grid, potential, _solve)

    assert result.final_grid.points <= config.max_points
    assert result.iterations <= config.max_refinement_steps
    assert result.final_grid.points > coarse_grid.points
    assert abs(result.final_wavefunctions[0].energy - analytical.energy(0)) < 1e-2


def test_adaptive_grid_handles_infinite_square_well_without_nan():
    """Ensure InfiniteSquareWell infinities at boundaries do not break potential analysis."""
    grid = SpatialGrid(x_min=0.0, x_max=1.0, points=100)
    isw = InfiniteSquareWell(width=1.0, center=0.5)
    adaptive = AdaptiveGridFramework()

    assessment = adaptive.assess_grid(grid, isw)
    assert np.isfinite(assessment.potential_variation_score)
    assert assessment.needs_domain_expansion is False


def test_parallel_execution_determinism():
    """Verify parallel batch solves preserve exact ordering and match sequential execution."""
    omegas = [0.5, 1.0, 1.5, 2.0]
    grid = SpatialGrid(x_min=-8.0, x_max=8.0, points=150)

    def _solve_for_omega(omega: float) -> float:
        eng = OptimizationEngine(OptimizationConfig(enable_cache=False, monitor_performance=False, run_stability_check=False))
        res = eng.solve_system(grid, HarmonicOscillator(omega=omega), num_eigenvalues=1)
        return float(res.energies[0])

    seq_engine = ParallelExecutionEngine(max_workers=1)
    par_engine = ParallelExecutionEngine(max_workers=2, min_tasks_for_parallel=2)

    seq_results = seq_engine.map_ordered(_solve_for_omega, omegas)
    par_results = par_engine.map_ordered(_solve_for_omega, omegas)

    np.testing.assert_allclose(par_results, seq_results, rtol=1e-12)
    # Energies must strictly increase with omega
    assert par_results == sorted(par_results)


def test_benchmark_engine_and_json_export(tmp_path):
    """Verify reproducible benchmarking and numerical equivalence reporting."""
    potential = HarmonicOscillator(mass=1.0, omega=1.0)
    suite = BenchmarkEngine.run_standard_suite(
        potential, x_min=-8.0, x_max=8.0, small_points=80, medium_points=200, large_points=400
    )

    assert len(suite.cases) == 4
    assert suite.all_equivalent is True
    assert suite.average_speedup > 0.0

    out_path = tmp_path / "benchmark_report.json"
    saved = suite.export_json(out_path)
    assert saved.exists()
    parsed = json.loads(saved.read_text(encoding="utf-8"))
    assert parsed["all_equivalent"] is True
    assert len(parsed["cases"]) == 4
