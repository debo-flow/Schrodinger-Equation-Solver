import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import numpy as np

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.potentials.base import Potential
from schrodinger.solvers.solver_configuration import SolverConfig
from schrodinger.optimization.optimization_engine import OptimizationEngine, OptimizationConfig
from schrodinger.coupled.coupled_potential import CoupledPotential
from schrodinger.coupled.coupled_system import CoupledQuantumSystem
from schrodinger.coupled.coupled_hamiltonian import CoupledHamiltonian
from schrodinger.coupled.coupled_solver import CoupledEigenvalueSolver

from .perturbation_operator import PerturbationOperator, PerturbationMatrix, QuantumState
from .non_degenerate import NonDegenerateStateResult, NonDegeneratePerturbationSolver
from .wavefunction_correction import WavefunctionCorrectionResult, WavefunctionCorrectionEngine
from .degenerate import DegenerateSubspaceResult, DegeneratePerturbationSolver
from .variational import VariationalResult
from .rayleigh_ritz import RayleighRitzResult


class PerturbationSystem:
    """
    Unified orchestrator for H = H_0 + lambda * H' supporting both single-channel
    and coupled-channel quantum systems.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        unperturbed_eigenstates: Sequence[QuantumState],
        perturbation: PerturbationOperator,
        lambda_strength: float = 1.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        unperturbed_potential: Optional[Union[Potential, CoupledPotential]] = None,
        accuracy_order: int = 2
    ):
        if not unperturbed_eigenstates:
            raise ValueError("unperturbed_eigenstates must not be empty.")
        self.grid = grid
        self.eigenstates: List[QuantumState] = list(unperturbed_eigenstates)
        self.unperturbed_energies: np.ndarray = np.array(
            [st.energy for st in self.eigenstates], dtype=np.float64
        )
        self.perturbation = perturbation
        self.lambda_strength = float(lambda_strength)
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system
        self.unperturbed_potential = unperturbed_potential
        self.accuracy_order = int(accuracy_order)
        self.is_coupled = isinstance(self.eigenstates[0], object) and hasattr(self.eigenstates[0], "num_channels")

        self._perturbation_matrix: Optional[PerturbationMatrix] = None

    @classmethod
    def from_single_channel(
        cls,
        grid: SpatialGrid,
        unperturbed_potential: Potential,
        perturbation: Union[PerturbationOperator, Potential],
        lambda_strength: float = 1.0,
        num_unperturbed_states: int = 10,
        mass: float = 1.0,
        hbar: float = 1.0,
        accuracy_order: int = 2
    ) -> "PerturbationSystem":
        """Solves H_0 on `grid` and initializes a single-channel PerturbationSystem."""
        pert_op = (
            perturbation
            if isinstance(perturbation, PerturbationOperator)
            else PerturbationOperator(perturbation)
        )
        if pert_op.unit_system != unperturbed_potential.unit_system:
            raise ValueError(
                f"Unit mismatch: H_0 has '{unperturbed_potential.unit_system}', "
                f"H' has '{pert_op.unit_system}'."
            )

        opt = OptimizationEngine(
            OptimizationConfig(
                accuracy_order=accuracy_order,
                enable_cache=False,
                monitor_performance=False,
                run_stability_check=True
            )
        )
        res = opt.solve_system(
            grid=grid,
            potential=unperturbed_potential,
            num_eigenvalues=num_unperturbed_states,
            mass=mass,
            hbar=hbar,
            unit_system=unperturbed_potential.unit_system,
            interior_only=(grid.boundary_condition == BoundaryCondition.DIRICHLET)
        )
        return cls(
            grid=grid,
            unperturbed_eigenstates=res.wavefunctions,
            perturbation=pert_op,
            lambda_strength=lambda_strength,
            mass=mass,
            hbar=hbar,
            unit_system=unperturbed_potential.unit_system,
            unperturbed_potential=unperturbed_potential,
            accuracy_order=accuracy_order
        )

    @classmethod
    def from_coupled_system(
        cls,
        coupled_system: CoupledQuantumSystem,
        perturbation: Union[PerturbationOperator, CoupledPotential],
        lambda_strength: float = 1.0,
        num_unperturbed_states: int = 8,
        accuracy_order: int = 2
    ) -> "PerturbationSystem":
        """Solves coupled H_0 and initializes a coupled-channel PerturbationSystem."""
        pert_op = (
            perturbation
            if isinstance(perturbation, PerturbationOperator)
            else PerturbationOperator(perturbation)
        )
        if pert_op.unit_system != coupled_system.unit_system:
            raise ValueError(
                f"Unit mismatch: coupled H_0 has '{coupled_system.unit_system}', "
                f"H' has '{pert_op.unit_system}'."
            )

        res = coupled_system.solve_eigenstates(
            num_eigenvalues=num_unperturbed_states,
            accuracy_order=accuracy_order,
            interior_only=(coupled_system.grid.boundary_condition == BoundaryCondition.DIRICHLET),
            include_uncoupled_comparison=False
        )
        return cls(
            grid=coupled_system.grid,
            unperturbed_eigenstates=res.eigenstates,
            perturbation=pert_op,
            lambda_strength=lambda_strength,
            mass=coupled_system.mass,
            hbar=coupled_system.hbar,
            unit_system=coupled_system.unit_system,
            unperturbed_potential=coupled_system.coupled_potential,
            accuracy_order=accuracy_order
        )

    @property
    def perturbation_matrix(self) -> PerturbationMatrix:
        """Lazily computes and caches W_{mn} = <m^(0) | H' | n^(0)>."""
        if self._perturbation_matrix is None:
            self._perturbation_matrix = PerturbationMatrix.from_eigenstates(
                self.eigenstates, self.perturbation
            )
        return self._perturbation_matrix

    def solve_non_degenerate(
        self,
        target_states: Optional[Sequence[int]] = None,
        order: int = 2,
        lambda_strength: Optional[float] = None,
        denominator_tolerance: float = 1e-6
    ) -> List[NonDegenerateStateResult]:
        lam = self.lambda_strength if lambda_strength is None else float(lambda_strength)
        solver = NonDegeneratePerturbationSolver(denominator_tolerance=denominator_tolerance)
        return solver.solve(
            unperturbed_energies=self.unperturbed_energies,
            perturbation_matrix=self.perturbation_matrix,
            lambda_strength=lam,
            order=order,
            target_states=target_states
        )

    def correct_wavefunctions(
        self,
        target_states: Optional[Sequence[int]] = None,
        lambda_strength: Optional[float] = None,
        denominator_tolerance: float = 1e-6
    ) -> List[WavefunctionCorrectionResult]:
        lam = self.lambda_strength if lambda_strength is None else float(lambda_strength)
        engine = WavefunctionCorrectionEngine(denominator_tolerance=denominator_tolerance)
        return engine.correct_states(
            eigenstates=self.eigenstates,
            perturbation_matrix=self.perturbation_matrix,
            lambda_strength=lam,
            target_states=target_states
        )

    def solve_degenerate(
        self,
        lambda_strength: Optional[float] = None,
        degeneracy_tolerance: float = 1e-5
    ) -> List[DegenerateSubspaceResult]:
        lam = self.lambda_strength if lambda_strength is None else float(lambda_strength)
        solver = DegeneratePerturbationSolver(degeneracy_tolerance=degeneracy_tolerance)
        return solver.solve_all_subspaces(
            eigenstates=self.eigenstates,
            perturbation_matrix=self.perturbation_matrix,
            lambda_strength=lam
        )

    def solve_exact_numerical(
        self,
        num_states: int,
        lambda_strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes independent exact numerical eigenvalues of H_0 + lambda * H'
        for direct benchmarking against perturbative approximations.
        """
        lam = self.lambda_strength if lambda_strength is None else float(lambda_strength)
        if self.unperturbed_potential is None:
            raise ValueError("unperturbed_potential is required to compute exact numerical H_0 + lambda*H'.")

        interior_only = (self.grid.boundary_condition == BoundaryCondition.DIRICHLET)

        if isinstance(self.unperturbed_potential, CoupledPotential):
            if not isinstance(self.perturbation.operator, CoupledPotential):
                raise TypeError("Exact coupled numerical benchmark requires a CoupledPotential perturbation.")
            total_pot = self.unperturbed_potential + (lam * self.perturbation.operator)
            ham = CoupledHamiltonian(
                grid=self.grid,
                coupled_potential=total_pot,
                mass=self.mass,
                hbar=self.hbar,
                unit_system=self.unit_system
            )
            res = CoupledEigenvalueSolver(SolverConfig(num_eigenvalues=num_states)).solve(
                ham,
                accuracy_order=self.accuracy_order,
                interior_only=interior_only,
                include_uncoupled_comparison=False
            )
            return res.energies

        if not isinstance(self.perturbation.operator, Potential):
            raise TypeError("Exact single-channel numerical benchmark requires a Potential perturbation.")

        total_single_pot = self.unperturbed_potential + (lam * self.perturbation.operator)
        opt = OptimizationEngine(
            OptimizationConfig(
                accuracy_order=self.accuracy_order,
                enable_cache=False,
                monitor_performance=False,
                run_stability_check=False
            )
        )
        res_single = opt.solve_system(
            grid=self.grid,
            potential=total_single_pot,
            num_eigenvalues=num_states,
            mass=self.mass,
            hbar=self.hbar,
            unit_system=self.unit_system,
            interior_only=interior_only
        )
        return res_single.energies


@dataclass
class ApproximationValidationReport:
    """Structured JSON-serializable report for perturbation and variational calculations."""
    method: str
    is_coupled_system: bool
    lambda_strength: float
    approximation_order: Optional[int]
    unperturbed_energies: List[float]
    first_order_corrections: List[float]
    second_order_corrections: List[float]
    total_approximate_energies: List[float]
    wavefunction_correction_info: List[Dict[str, Any]]
    degeneracy_info: List[Dict[str, Any]]
    variational_info: Optional[Dict[str, Any]]
    numerical_reference_energies: Optional[List[float]]
    absolute_errors: Optional[List[float]]
    relative_errors: Optional[List[Optional[float]]]
    max_absolute_error: Optional[float]
    warnings: List[str]
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, filepath: Union[str, Path], indent: int = 4) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path


class ApproximationValidationEngine:
    """
    Validates perturbative, variational, and Rayleigh-Ritz approximations against
    independent exact numerical solutions and compiles structured JSON reports.
    """

    def __init__(self, energy_error_tolerance: float = 1e-2):
        if energy_error_tolerance <= 0:
            raise ValueError("energy_error_tolerance must be strictly positive.")
        self.energy_error_tolerance = float(energy_error_tolerance)

    @staticmethod
    def _safe_rel(approx: float, ref: float) -> Optional[float]:
        err = abs(approx - ref)
        if abs(ref) < 1e-12:
            return 0.0 if err < 1e-12 else None
        return float(err / abs(ref))

    def validate_perturbation_system(
        self,
        system: PerturbationSystem,
        target_states: Optional[Sequence[int]] = None,
        order: int = 2,
        compare_exact_numerical: bool = True,
        degeneracy_tolerance: float = 1e-5
    ) -> ApproximationValidationReport:
        """Runs non-degenerate/degenerate analysis, wavefunction corrections, and numerical comparison."""
        indices = list(range(min(3, len(system.eigenstates)))) if target_states is None else list(target_states)
        warnings: List[str] = []

        deg_results = system.solve_degenerate(degeneracy_tolerance=degeneracy_tolerance)
        deg_info = [
            {
                "subspace_indices": r.subspace_indices,
                "unperturbed_energy": r.unperturbed_energy,
                "first_order_shifts": [float(s) for s in r.first_order_shifts],
                "split_energies": [float(e) for e in r.split_energies],
                "degeneracy_lifted": r.degeneracy_lifted,
                "splitting_spread": r.splitting_spread
            }
            for r in deg_results
        ]

        nd_results = system.solve_non_degenerate(target_states=indices, order=order)
        wf_results = system.correct_wavefunctions(target_states=indices)

        unpert_e: List[float] = []
        e1_list: List[float] = []
        e2_list: List[float] = []
        approx_e: List[float] = []

        # Map degenerate state indices to their lifted energies when applicable
        deg_energy_map: Dict[int, tuple] = {}
        for r in deg_results:
            for pos, idx in enumerate(r.subspace_indices):
                deg_energy_map[idx] = (
                    float(r.first_order_shifts[pos] / system.lambda_strength) if system.lambda_strength != 0 else 0.0,
                    float(r.split_energies[pos])
                )

        for r in nd_results:
            unpert_e.append(r.unperturbed_energy)
            if r.state_index in deg_energy_map:
                shift_unit, split_e = deg_energy_map[r.state_index]
                e1_list.append(shift_unit)
                e2_list.append(0.0)
                approx_e.append(split_e)
            else:
                e1_list.append(r.first_order_correction)
                e2_list.append(r.second_order_correction)
                approx_e.append(r.total_approximate_energy)
                warnings.extend(r.warnings)

        wf_info: List[Dict[str, Any]] = []
        for wr in wf_results:
            wf_info.append({
                "state_index": wr.state_index,
                "correction_norm": wr.correction_norm,
                "overlap_probability_with_unperturbed": wr.overlap_probability,
                "orthogonality_to_unperturbed_error": wr.orthogonality_to_unperturbed_error
            })
            if wr.state_index not in deg_energy_map:
                warnings.extend(wr.warnings)

        ref_energies: Optional[List[float]] = None
        abs_errs: Optional[List[float]] = None
        rel_errs: Optional[List[Optional[float]]] = None
        max_abs: Optional[float] = None

        if compare_exact_numerical and system.unperturbed_potential is not None:
            max_idx = max(indices) + 1
            exact_all = system.solve_exact_numerical(num_states=max_idx)
            ref_energies = [float(exact_all[i]) for i in indices]
            abs_errs = [float(abs(a - b)) for a, b in zip(approx_e, ref_energies)]
            rel_errs = [self._safe_rel(a, b) for a, b in zip(approx_e, ref_energies)]
            max_abs = float(max(abs_errs)) if abs_errs else 0.0
            if max_abs > self.energy_error_tolerance:
                warnings.append(
                    f"Max energy error vs exact numerical ({max_abs:.2e}) exceeds tolerance ({self.energy_error_tolerance:.2e})."
                )

        method_label = "degenerate_and_non_degenerate_perturbation" if deg_results else "non_degenerate_perturbation"
        return ApproximationValidationReport(
            method=method_label,
            is_coupled_system=system.is_coupled,
            lambda_strength=system.lambda_strength,
            approximation_order=order,
            unperturbed_energies=unpert_e,
            first_order_corrections=e1_list,
            second_order_corrections=e2_list,
            total_approximate_energies=approx_e,
            wavefunction_correction_info=wf_info,
            degeneracy_info=deg_info,
            variational_info=None,
            numerical_reference_energies=ref_energies,
            absolute_errors=abs_errs,
            relative_errors=rel_errs,
            max_absolute_error=max_abs,
            warnings=warnings,
            validation_passed=(len(warnings) == 0)
        )

    def validate_variational(self, result: VariationalResult) -> ApproximationValidationReport:
        """Builds a structured validation report for a VariationalResult."""
        warnings: List[str] = []
        if not result.converged:
            warnings.append(f"Variational optimizer did not converge: {result.optimizer_message}")
        if result.upper_bound_verified is False:
            warnings.append("Variational upper-bound property E[psi] >= E_0 was violated.")
        if result.energy_error_vs_reference is not None and abs(result.energy_error_vs_reference) > self.energy_error_tolerance:
            warnings.append(
                f"Variational energy error ({result.energy_error_vs_reference:.2e}) exceeds tolerance "
                f"({self.energy_error_tolerance:.2e})."
            )

        ref_list = [result.reference_ground_energy] if result.reference_ground_energy is not None else None
        abs_list = [abs(result.energy_error_vs_reference)] if result.energy_error_vs_reference is not None else None
        rel_list = (
            [self._safe_rel(result.optimal_energy, result.reference_ground_energy)]
            if result.reference_ground_energy is not None
            else None
        )

        return ApproximationValidationReport(
            method=f"variational_{result.optimizer_method}",
            is_coupled_system=False,
            lambda_strength=1.0,
            approximation_order=None,
            unperturbed_energies=[result.initial_energy],
            first_order_corrections=[],
            second_order_corrections=[],
            total_approximate_energies=[result.optimal_energy],
            wavefunction_correction_info=[],
            degeneracy_info=[],
            variational_info={
                "optimal_parameters": [float(p) for p in result.optimal_parameters],
                "optimizer_method": result.optimizer_method,
                "converged": result.converged,
                "iterations": result.iterations,
                "upper_bound_verified": result.upper_bound_verified
            },
            numerical_reference_energies=ref_list,
            absolute_errors=abs_list,
            relative_errors=rel_list,
            max_absolute_error=abs_list[0] if abs_list else None,
            warnings=warnings,
            validation_passed=(len(warnings) == 0)
        )

    def validate_rayleigh_ritz(
        self,
        result: RayleighRitzResult,
        numerical_reference_energies: Optional[Sequence[float]] = None
    ) -> ApproximationValidationReport:
        """Builds a structured validation report for a RayleighRitzResult."""
        warnings: List[str] = []
        if result.max_generalized_residual > 1e-6:
            warnings.append(f"Large generalized eigenpair residual: {result.max_generalized_residual:.2e}.")

        ref_list: Optional[List[float]] = None
        abs_errs: Optional[List[float]] = None
        rel_errs: Optional[List[Optional[float]]] = None
        max_abs: Optional[float] = None

        if numerical_reference_energies is not None:
            k = min(len(result.energies), len(numerical_reference_energies))
            ref_list = [float(numerical_reference_energies[i]) for i in range(k)]
            abs_errs = [float(abs(result.energies[i] - ref_list[i])) for i in range(k)]
            rel_errs = [self._safe_rel(float(result.energies[i]), ref_list[i]) for i in range(k)]
            max_abs = float(max(abs_errs))
            if max_abs > self.energy_error_tolerance:
                warnings.append(
                    f"Rayleigh-Ritz max energy error ({max_abs:.2e}) exceeds tolerance ({self.energy_error_tolerance:.2e})."
                )

        return ApproximationValidationReport(
            method="rayleigh_ritz_basis_expansion",
            is_coupled_system=False,
            lambda_strength=1.0,
            approximation_order=None,
            unperturbed_energies=[],
            first_order_corrections=[],
            second_order_corrections=[],
            total_approximate_energies=[float(e) for e in result.energies],
            wavefunction_correction_info=[],
            degeneracy_info=[],
            variational_info={
                "basis_size": result.basis_size,
                "is_orthogonal_basis": result.is_orthogonal_basis,
                "min_overlap_eigenvalue": result.min_overlap_eigenvalue,
                "overlap_condition_number": result.overlap_condition_number,
                "max_generalized_residual": result.max_generalized_residual
            },
            numerical_reference_energies=ref_list,
            absolute_errors=abs_errs,
            relative_errors=rel_errs,
            max_absolute_error=max_abs,
            warnings=warnings,
            validation_passed=(len(warnings) == 0)
        )
