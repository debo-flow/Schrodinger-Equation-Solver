from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from schrodinger.potentials.base import Potential
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.potential_operator import PotentialEnergyOperator
from schrodinger.operators.hamiltonian import Hamiltonian
from schrodinger.solvers.solver_configuration import SolverConfig
from schrodinger.solvers.eigenvalue_solver import EigenvalueSolver
from schrodinger.wavefunctions.reconstruction import WavefunctionReconstructor
from .analytical_solver import AnalyticalSolution
from .numerical_comparison import NumericalComparisonEngine


@dataclass
class ConvergencePoint:
    """Error metrics recorded at a single spatial grid resolution."""
    grid_points: int
    dx: float
    ground_energy_error: float
    max_energy_error: float
    ground_wavefunction_l2_error: float


@dataclass
class ConvergenceResult:
    """Full grid-refinement convergence study summary."""
    model_name: str
    accuracy_order: int
    points: List[ConvergencePoint]
    estimated_energy_order: Optional[float]
    estimated_wavefunction_order: Optional[float]
    pairwise_energy_orders: List[float]
    is_monotonically_converging: bool
    is_numerically_stable: bool


class ConvergenceAnalyzer:
    """
    Evaluates numerical solver accuracy across multiple spatial grid resolutions
    and estimates the empirical rate of convergence with respect to grid spacing dx.
    """

    @staticmethod
    def estimate_convergence_order(dx_values: List[float], errors: List[float]) -> Optional[float]:
        """
        Estimates convergence order p in Error ~ C * dx^p via log-log linear regression.
        Returns None if insufficient positive error samples exist.
        """
        dx_arr = np.asarray(dx_values, dtype=np.float64)
        err_arr = np.asarray(errors, dtype=np.float64)

        valid = (dx_arr > 0) & (err_arr > 1e-15) & np.isfinite(err_arr)
        if np.sum(valid) < 2:
            return None

        slope, _ = np.polyfit(np.log(dx_arr[valid]), np.log(err_arr[valid]), 1)
        return float(slope)

    @staticmethod
    def pairwise_orders(dx_values: List[float], errors: List[float]) -> List[float]:
        """Computes local convergence rates between consecutive grid resolutions."""
        orders: List[float] = []
        for i in range(1, len(dx_values)):
            e_prev, e_curr = errors[i - 1], errors[i]
            dx_prev, dx_curr = dx_values[i - 1], dx_values[i]
            if e_prev > 1e-15 and e_curr > 1e-15 and dx_prev != dx_curr:
                rate = np.log(e_prev / e_curr) / np.log(dx_prev / dx_curr)
                orders.append(float(rate))
        return orders

    @classmethod
    def run_study(
        cls,
        potential: Potential,
        analytical_solution: AnalyticalSolution,
        x_min: float,
        x_max: float,
        grid_resolutions: List[int],
        num_states: int = 3,
        accuracy_order: int = 2,
        boundary_condition: BoundaryCondition = BoundaryCondition.DIRICHLET
    ) -> ConvergenceResult:
        """
        Executes the full numerical pipeline across sorted grid resolutions
        and evaluates convergence against the analytical reference.
        """
        if len(grid_resolutions) < 2:
            raise ValueError("Convergence analysis requires at least 2 distinct grid resolutions.")

        sorted_resolutions = sorted(set(grid_resolutions))
        if len(sorted_resolutions) < 2:
            raise ValueError("Grid resolutions must contain at least 2 distinct values.")

        points: List[ConvergencePoint] = []
        interior_only = (boundary_condition == BoundaryCondition.DIRICHLET)

        for n_points in sorted_resolutions:
            grid = SpatialGrid(
                x_min=x_min,
                x_max=x_max,
                points=n_points,
                boundary_condition=boundary_condition
            )
            fd_engine = FiniteDifferenceEngine(grid)
            kinetic_op = KineticEnergyOperator(
                fd_engine,
                mass=analytical_solution.mass,
                hbar=analytical_solution.hbar,
                unit_system=analytical_solution.unit_system
            )
            potential_op = PotentialEnergyOperator(grid, potential)
            hamiltonian = Hamiltonian(kinetic_op, potential_op)

            h_sparse = hamiltonian.build_sparse(
                accuracy_order=accuracy_order,
                interior_only=interior_only
            )

            solver = EigenvalueSolver(SolverConfig(num_eigenvalues=num_states))
            solver_result = solver.solve(h_sparse)

            reconstructor = WavefunctionReconstructor(grid)
            numerical_wfs = reconstructor.reconstruct(solver_result)

            comparison = NumericalComparisonEngine.compare_spectrum(
                numerical_wfs,
                analytical_solution
            )

            points.append(
                ConvergencePoint(
                    grid_points=n_points,
                    dx=grid.dx,
                    ground_energy_error=comparison.state_comparisons[0].energy_absolute_error,
                    max_energy_error=comparison.max_energy_absolute_error,
                    ground_wavefunction_l2_error=comparison.state_comparisons[0].wavefunction_l2_error
                )
            )

        dx_vals = [p.dx for p in points]
        e_errs = [p.ground_energy_error for p in points]
        wf_errs = [p.ground_wavefunction_l2_error for p in points]

        est_e_order = cls.estimate_convergence_order(dx_vals, e_errs)
        est_wf_order = cls.estimate_convergence_order(dx_vals, wf_errs)
        pw_orders = cls.pairwise_orders(dx_vals, e_errs)

        is_monotonic = all(e_errs[i] <= e_errs[i - 1] * (1.0 + 1e-9) for i in range(1, len(e_errs)))
        is_stable = (
            is_monotonic
            and (est_e_order is not None and est_e_order > 0.5)
            and all(np.isfinite(e_errs))
        )

        return ConvergenceResult(
            model_name=analytical_solution.name,
            accuracy_order=accuracy_order,
            points=points,
            estimated_energy_order=est_e_order,
            estimated_wavefunction_order=est_wf_order,
            pairwise_energy_orders=pw_orders,
            is_monotonically_converging=is_monotonic,
            is_numerically_stable=is_stable
        )
