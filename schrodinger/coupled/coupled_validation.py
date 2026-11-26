import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from schrodinger.analysis.hermiticity import HermiticityAnalyzer
from schrodinger.solvers.solver_configuration import SolverConfig
from .coupled_hamiltonian import CoupledHamiltonian
from .coupled_solver import CoupledSolverResult
from .coupled_evolution import CoupledEvolutionResult
from .channel_analysis import ChannelPopulationAnalyzer


@dataclass
class CoupledValidationReport:
    """JSON-compatible validation report for a coupled quantum system."""
    num_channels: int
    hamiltonian_dimensions: List[int]
    is_hermitian: bool
    hermiticity_deviation: float
    coupling_info: Dict[str, Any]
    energy_eigenvalues: List[float]
    eigenpair_residuals: List[float]
    max_eigenpair_residual: float
    channel_probabilities_per_state: List[List[float]]
    normalization_errors: List[float]
    max_normalization_error: float
    conservation_error: float
    uncoupled_energy_shifts: Optional[List[float]]
    solver_configuration: Dict[str, Any]
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


class CoupledValidationEngine:
    """
    Validates coupled Hamiltonians, multi-component eigenpairs, channel population
    bounds, uncoupled limits, and time-dependent probability conservation.
    """

    def __init__(
        self,
        hermitian_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-5,
        norm_tolerance: float = 1e-4,
        conservation_tolerance: float = 1e-3
    ):
        self.hermitian_tolerance = float(hermitian_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.norm_tolerance = float(norm_tolerance)
        self.conservation_tolerance = float(conservation_tolerance)

    def generate_report(
        self,
        coupled_hamiltonian: CoupledHamiltonian,
        solver_result: CoupledSolverResult,
        solver_config: SolverConfig,
        evolution_result: Optional[CoupledEvolutionResult] = None,
        accuracy_order: int = 2,
        interior_only: bool = True
    ) -> CoupledValidationReport:
        """Runs comprehensive validation checks and builds a CoupledValidationReport."""
        warnings: List[str] = []

        h_sparse = coupled_hamiltonian.build_sparse(
            accuracy_order=accuracy_order,
            interior_only=interior_only
        )
        herm_analyzer = HermiticityAnalyzer(tolerance=self.hermitian_tolerance)
        is_herm, herm_dev = herm_analyzer.is_hermitian(h_sparse)
        if not is_herm:
            warnings.append(f"Coupled Hamiltonian is not Hermitian (dev={herm_dev:.2e}).")

        # Validate eigenpair residuals
        if solver_result.max_residual > self.residual_tolerance:
            warnings.append(
                f"Max eigenpair residual {solver_result.max_residual:.2e} > {self.residual_tolerance:.2e}."
            )

        # Validate multi-component state normalization and channel population bounds
        channel_probs: List[List[float]] = []
        norm_errors: List[float] = []
        for idx, state in enumerate(solver_result.eigenstates):
            norm_err = abs(state.total_norm - 1.0)
            norm_errors.append(float(norm_err))
            if norm_err > self.norm_tolerance:
                warnings.append(
                    f"Eigenstate {idx} normalization error {norm_err:.2e} > {self.norm_tolerance:.2e}."
                )

            pop_report = ChannelPopulationAnalyzer.analyze_state(state, tolerance=self.norm_tolerance)
            channel_probs.append(pop_report.channel_populations)
            if not pop_report.bounds_valid:
                warnings.append(
                    f"Eigenstate {idx} channel populations violated bounds/conservation "
                    f"(sum={pop_report.total_population:.6f})."
                )

        # Conservation error from static eigenstate populations and/or time evolution
        static_cons_err = max((abs(sum(p) - 1.0) for p in channel_probs), default=0.0)
        evo_cons_err = evolution_result.max_norm_deviation if evolution_result is not None else 0.0
        total_cons_err = float(max(static_cons_err, evo_cons_err))

        if evolution_result is not None:
            warnings.extend(evolution_result.stability_warnings)
            if evolution_result.max_norm_deviation > self.conservation_tolerance:
                warnings.append(
                    f"Time-dependent probability deviation {evolution_result.max_norm_deviation:.2e} "
                    f"> {self.conservation_tolerance:.2e}."
                )
            if not evolution_result.channel_trajectory.bounds_valid:
                warnings.append("Time-dependent channel populations violated [0, 1] bounds.")

        x_probe = coupled_hamiltonian.grid.interior_points if interior_only else coupled_hamiltonian.grid.grid
        coupling_info = coupled_hamiltonian.coupled_potential.coupling_summary(x_probe)

        shifts: Optional[List[float]] = None
        if solver_result.uncoupled_comparison is not None:
            shifts = [float(s) for s in solver_result.uncoupled_comparison.energy_shifts]

        cfg_dict = {
            "num_eigenvalues": solver_config.num_eigenvalues,
            "tolerance": solver_config.tolerance,
            "method": solver_config.method.value,
            "method_used": solver_result.method_used,
            "accuracy_order": accuracy_order,
            "interior_only": interior_only,
            "mass": coupled_hamiltonian.mass,
            "hbar": coupled_hamiltonian.hbar,
            "unit_system": coupled_hamiltonian.unit_system
        }

        return CoupledValidationReport(
            num_channels=coupled_hamiltonian.num_channels,
            hamiltonian_dimensions=list(h_sparse.shape),
            is_hermitian=is_herm,
            hermiticity_deviation=float(herm_dev),
            coupling_info=coupling_info,
            energy_eigenvalues=[float(e) for e in solver_result.energies],
            eigenpair_residuals=[float(r) for r in solver_result.residuals],
            max_eigenpair_residual=float(solver_result.max_residual),
            channel_probabilities_per_state=channel_probs,
            normalization_errors=norm_errors,
            max_normalization_error=float(max(norm_errors)) if norm_errors else 0.0,
            conservation_error=total_cons_err,
            uncoupled_energy_shifts=shifts,
            solver_configuration=cfg_dict,
            warnings=warnings,
            validation_passed=(len(warnings) == 0)
        )
