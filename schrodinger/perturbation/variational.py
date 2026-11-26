from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.potential_operator import PotentialEnergyOperator
from schrodinger.operators.hamiltonian import Hamiltonian
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.optimization.optimization_engine import OptimizationEngine, OptimizationConfig


TrialWavefunctionFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class VariationalResult:
    """Result of Rayleigh-Ritz variational energy minimization."""
    optimal_parameters: np.ndarray
    initial_energy: float
    optimal_energy: float
    optimal_wavefunction: Wavefunction
    optimizer_method: str
    converged: bool
    iterations: int
    optimizer_message: str
    reference_ground_energy: Optional[float]
    energy_error_vs_reference: Optional[float]
    upper_bound_verified: Optional[bool]


class VariationalSolver:
    """
    Rayleigh-Ritz variational quantum solver minimizing:
        E[psi(theta)] = <psi(theta) | H | psi(theta)> / <psi(theta) | psi(theta)>
    using the consistent Hermitian discretized Hamiltonian to preserve the
    variational upper-bound property E[psi] >= E_0.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Potential,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        accuracy_order: int = 2
    ):
        self.grid = grid
        self.potential = potential
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system
        self.accuracy_order = int(accuracy_order)
        self._interior_only = (grid.boundary_condition == BoundaryCondition.DIRICHLET)

        fd = FiniteDifferenceEngine(grid)
        kin_op = KineticEnergyOperator(fd, mass=self.mass, hbar=self.hbar, unit_system=self.unit_system)
        pot_op = PotentialEnergyOperator(grid, potential)
        self._h_sparse = Hamiltonian(kin_op, pot_op).build_sparse(
            accuracy_order=self.accuracy_order,
            interior_only=self._interior_only
        )

    def _active_vector(self, psi_full: np.ndarray) -> np.ndarray:
        arr = np.asarray(psi_full, dtype=np.complex128)
        if arr.shape != (self.grid.points,):
            raise ValueError(f"Trial wavefunction shape {arr.shape} must match grid {(self.grid.points,)}.")
        return arr[1:-1] if self._interior_only else arr

    def evaluate_energy(
        self,
        trial_fn: TrialWavefunctionFn,
        params: Sequence[float]
    ) -> float:
        """Evaluates the Rayleigh quotient <psi | H | psi> / <psi | psi> for `params`."""
        p_arr = np.asarray(params, dtype=np.float64)
        psi_full = trial_fn(self.grid.grid, p_arr)
        vec = self._active_vector(psi_full)

        if not np.all(np.isfinite(vec)):
            raise ValueError("Trial wavefunction produced NaN or Inf entries.")
        norm_sq = float(np.real(np.vdot(vec, vec)))
        if norm_sq <= 1e-30:
            raise ValueError("Trial wavefunction has zero norm on active grid points.")

        h_vec = self._h_sparse.dot(vec)
        rayleigh = np.vdot(vec, h_vec) / norm_sq
        if abs(np.imag(rayleigh)) > 1e-8:
            raise ValueError(f"Non-real Rayleigh quotient encountered: {rayleigh}")
        return float(np.real(rayleigh))

    def optimize(
        self,
        trial_fn: TrialWavefunctionFn,
        initial_params: Sequence[float],
        method: str = "Nelder-Mead",
        bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
        tolerance: float = 1e-8,
        options: Optional[Dict[str, Any]] = None,
        compare_with_numerical: bool = True,
        upper_bound_tolerance: float = 1e-7
    ) -> VariationalResult:
        """
        Optimizes variational parameters using `scipy.optimize.minimize` and
        optionally verifies the variational upper bound E[psi] >= E_0^num.
        """
        p0 = np.asarray(initial_params, dtype=np.float64)
        if p0.ndim != 1 or len(p0) == 0:
            raise ValueError("initial_params must be a non-empty 1D sequence.")

        initial_energy = self.evaluate_energy(trial_fn, p0)

        def _objective(p: np.ndarray) -> float:
            try:
                val = self.evaluate_energy(trial_fn, p)
                return val if np.isfinite(val) else 1e30
            except (ValueError, ArithmeticError, OverflowError):
                return 1e30

        opt_res = minimize(
            _objective,
            x0=p0,
            method=method,
            bounds=bounds,
            tol=tolerance,
            options=options
        )

        opt_params = np.asarray(opt_res.x, dtype=np.float64)
        opt_energy = self.evaluate_energy(trial_fn, opt_params)

        # Build normalized Wavefunction for optimal parameters
        psi_opt_raw = np.asarray(trial_fn(self.grid.grid, opt_params), dtype=np.complex128).copy()
        if self._interior_only:
            psi_opt_raw[0] = 0.0 + 0.0j
            psi_opt_raw[-1] = 0.0 + 0.0j
        psi_normed, _ = NormalizationEngine.normalize(
            psi_opt_raw, self.grid.grid, method=IntegrationMethod.SIMPSON
        )
        opt_wf = Wavefunction(x=self.grid.grid, psi=psi_normed, energy=opt_energy)

        ref_e: Optional[float] = None
        err_vs_ref: Optional[float] = None
        upper_ok: Optional[bool] = None

        if compare_with_numerical:
            opt_eng = OptimizationEngine(
                OptimizationConfig(
                    accuracy_order=self.accuracy_order,
                    enable_cache=False,
                    monitor_performance=False,
                    run_stability_check=False
                )
            )
            num_res = opt_eng.solve_system(
                self.grid,
                self.potential,
                num_eigenvalues=1,
                mass=self.mass,
                hbar=self.hbar,
                unit_system=self.unit_system,
                interior_only=self._interior_only
            )
            ref_e = float(num_res.energies[0])
            err_vs_ref = float(opt_energy - ref_e)
            upper_ok = bool(opt_energy >= ref_e - upper_bound_tolerance)

        return VariationalResult(
            optimal_parameters=opt_params,
            initial_energy=initial_energy,
            optimal_energy=opt_energy,
            optimal_wavefunction=opt_wf,
            optimizer_method=method,
            converged=bool(opt_res.success),
            iterations=int(getattr(opt_res, "nit", 0)),
            optimizer_message=str(opt_res.message),
            reference_ground_energy=ref_e,
            energy_error_vs_reference=err_vs_ref,
            upper_bound_verified=upper_ok
        )
