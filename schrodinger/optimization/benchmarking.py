import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.potential_operator import PotentialEnergyOperator
from schrodinger.operators.hamiltonian import Hamiltonian
from schrodinger.solvers.solver_configuration import SolverConfig, SolverMethod
from schrodinger.solvers.eigenvalue_solver import EigenvalueSolver
from schrodinger.wavefunctions.reconstruction import WavefunctionReconstructor
from schrodinger.verification.numerical_comparison import NumericalComparisonEngine
from schrodinger.verification.error_analysis import ErrorAnalyzer

from .performance_monitor import PerformanceMonitor
from .solver_selector import OptimizedSolverMethod
from .optimization_engine import OptimizationEngine, OptimizationConfig


@dataclass
class BenchmarkCaseResult:
    """Comparison metrics for a single baseline vs optimized benchmark case."""
    case_name: str
    grid_points: int
    num_eigenvalues: int
    baseline_method: str
    optimized_method: str
    baseline_time_seconds: float
    optimized_time_seconds: float
    speedup_factor: float
    baseline_peak_memory_mb: float
    optimized_peak_memory_mb: float
    max_energy_difference: float
    max_wavefunction_l2_difference: float
    numerically_equivalent: bool


@dataclass
class BenchmarkSuiteReport:
    """Aggregated benchmark results supporting JSON serialization."""
    cases: List[BenchmarkCaseResult]
    average_speedup: float
    all_equivalent: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, filepath: Union[str, Path], indent: int = 4) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path


class BenchmarkEngine:
    """
    Runs reproducible benchmarks comparing the Milestone 5-8 baseline pipeline
    against the Milestone 11 OptimizationEngine across grid sizes, matrix types,
    and eigenvalue counts while verifying strict numerical equivalence.
    """

    @staticmethod
    def _run_baseline(
        grid: SpatialGrid,
        potential: Potential,
        num_eigenvalues: int,
        baseline_method: SolverMethod,
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ):
        monitor = PerformanceMonitor(track_memory=True)
        monitor.start()

        with monitor.stage("matrix_construction"):
            fd_engine = FiniteDifferenceEngine(grid)
            kin_op = KineticEnergyOperator(fd_engine, mass=mass, hbar=hbar)
            pot_op = PotentialEnergyOperator(grid, potential)
            ham = Hamiltonian(kin_op, pot_op)
            h_mat = (
                ham.build_dense(accuracy_order=accuracy_order, interior_only=True)
                if baseline_method == SolverMethod.DENSE
                else ham.build_sparse(accuracy_order=accuracy_order, interior_only=True)
            )

        with monitor.stage("eigenvalue_computation"):
            solver = EigenvalueSolver(
                SolverConfig(num_eigenvalues=num_eigenvalues, method=baseline_method)
            )
            solver_res = solver.solve(h_mat)

        with monitor.stage("wavefunction_reconstruction"):
            wfs = WavefunctionReconstructor(grid).reconstruct(solver_res)

        monitor.stop()
        return solver_res["energies"], wfs, solver_res["method_used"], monitor.build_report()

    @classmethod
    def benchmark_case(
        cls,
        case_name: str,
        potential: Potential,
        x_min: float,
        x_max: float,
        grid_points: int,
        num_eigenvalues: int = 5,
        baseline_method: SolverMethod = SolverMethod.AUTO,
        optimized_method: Optional[OptimizedSolverMethod] = None,
        accuracy_order: int = 2,
        equivalence_tolerance: float = 1e-6
    ) -> BenchmarkCaseResult:
        """Benchmarks a single configuration and verifies numerical equivalence."""
        grid = SpatialGrid(
            x_min=x_min,
            x_max=x_max,
            points=grid_points,
            boundary_condition=BoundaryCondition.DIRICHLET
        )

        base_e, base_wfs, base_method_used, base_perf = cls._run_baseline(
            grid, potential, num_eigenvalues, baseline_method, accuracy_order=accuracy_order
        )

        opt_config = OptimizationConfig(
            accuracy_order=accuracy_order,
            forced_method=optimized_method,
            enable_cache=False,
            monitor_performance=True,
            run_stability_check=False
        )
        opt_engine = OptimizationEngine(opt_config)
        opt_res = opt_engine.solve_system(grid, potential, num_eigenvalues=num_eigenvalues)

        # Numerical equivalence check (phase-aligned)
        max_e_diff = float(np.max(np.abs(base_e - opt_res.energies)))
        max_wf_diff = 0.0
        for wf_b, wf_o in zip(base_wfs, opt_res.wavefunctions):
            aligned_o, _ = NumericalComparisonEngine.align_global_phase(wf_o.psi, wf_b.psi, grid.grid)
            l2_err = ErrorAnalyzer.l2_wavefunction_error(aligned_o, wf_b.psi, grid.grid)
            max_wf_diff = max(max_wf_diff, l2_err)

        opt_time = opt_res.performance_report.total_time_seconds if opt_res.performance_report else 1e-12
        opt_mem = opt_res.performance_report.peak_memory_mb if opt_res.performance_report else 0.0
        speedup = base_perf.total_time_seconds / max(opt_time, 1e-12)

        equivalent = (max_e_diff <= equivalence_tolerance) and (max_wf_diff <= max(equivalence_tolerance * 100, 1e-4))

        return BenchmarkCaseResult(
            case_name=case_name,
            grid_points=grid_points,
            num_eigenvalues=num_eigenvalues,
            baseline_method=base_method_used,
            optimized_method=opt_res.method_used.value,
            baseline_time_seconds=base_perf.total_time_seconds,
            optimized_time_seconds=opt_time,
            speedup_factor=float(speedup),
            baseline_peak_memory_mb=base_perf.peak_memory_mb,
            optimized_peak_memory_mb=opt_mem,
            max_energy_difference=max_e_diff,
            max_wavefunction_l2_difference=max_wf_diff,
            numerically_equivalent=equivalent
        )

    @classmethod
    def run_standard_suite(
        cls,
        potential: Potential,
        x_min: float = -8.0,
        x_max: float = 8.0,
        small_points: int = 100,
        medium_points: int = 400,
        large_points: int = 1000
    ) -> BenchmarkSuiteReport:
        """Runs benchmarks across small, medium, and large grids, dense/sparse, and k counts."""
        cases = [
            cls.benchmark_case(
                "small_grid_dense_k3", potential, x_min, x_max, small_points,
                num_eigenvalues=3, baseline_method=SolverMethod.DENSE
            ),
            cls.benchmark_case(
                "medium_grid_dense_k5", potential, x_min, x_max, medium_points,
                num_eigenvalues=5, baseline_method=SolverMethod.DENSE
            ),
            cls.benchmark_case(
                "large_grid_sparse_k5", potential, x_min, x_max, large_points,
                num_eigenvalues=5, baseline_method=SolverMethod.SPARSE
            ),
            cls.benchmark_case(
                "medium_grid_k10", potential, x_min, x_max, medium_points,
                num_eigenvalues=10, baseline_method=SolverMethod.AUTO
            )
        ]

        avg_speedup = float(np.mean([c.speedup_factor for c in cases]))
        all_equiv = all(c.numerically_equivalent for c in cases)
        return BenchmarkSuiteReport(
            cases=cases,
            average_speedup=avg_speedup,
            all_equivalent=all_equiv
        )
