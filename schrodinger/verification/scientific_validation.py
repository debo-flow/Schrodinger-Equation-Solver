import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from schrodinger.wavefunctions.wavefunction import Wavefunction
from .analytical_solver import AnalyticalSolution
from .numerical_comparison import NumericalComparisonEngine, SpectrumComparisonResult
from .convergence_analysis import ConvergenceResult


@dataclass
class ValidationTolerances:
    """Acceptance criteria thresholds for scientific validation."""
    max_energy_relative_error: float = 1e-2
    max_energy_absolute_error: float = 1e-2
    max_wavefunction_l2_error: float = 5e-2
    min_state_fidelity: float = 0.99
    max_normalization_error: float = 1e-4
    max_orthogonality_error: float = 1e-4
    min_convergence_order: float = 1.5


@dataclass
class ScientificValidationReport:
    """Structured scientific benchmark report supporting JSON export."""
    potential_model: str
    physical_parameters: Dict[str, Any]
    grid_resolution: int
    grid_spacing: float
    domain: List[float]
    quantum_numbers: List[int]
    numerical_energies: List[float]
    analytical_energies: List[float]
    energy_absolute_errors: List[float]
    energy_relative_errors: List[Optional[float]]
    wavefunction_metrics: Dict[str, Any]
    convergence_info: Optional[Dict[str, Any]]
    tolerances: Dict[str, float]
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


class ScientificValidator:
    """
    Compiles numerical comparisons and optional grid convergence studies into a
    standardized ScientificValidationReport with pass/fail gating.
    """

    def __init__(self, tolerances: ValidationTolerances = ValidationTolerances()):
        self.tolerances = tolerances

    def validate(
        self,
        numerical_wavefunctions: List[Wavefunction],
        analytical_solution: AnalyticalSolution,
        convergence_result: Optional[ConvergenceResult] = None
    ) -> ScientificValidationReport:
        """Evaluates numerical outputs against analytical truth and tolerances."""
        if not numerical_wavefunctions:
            raise ValueError("Cannot validate an empty list of wavefunctions.")

        comparison: SpectrumComparisonResult = NumericalComparisonEngine.compare_spectrum(
            numerical_wavefunctions,
            analytical_solution
        )

        x = numerical_wavefunctions[0].x
        grid_resolution = int(len(x))
        grid_spacing = float((x[-1] - x[0]) / (grid_resolution - 1))
        domain = [float(x[0]), float(x[-1])]

        warnings: List[str] = []

        # Energy checks
        if comparison.max_energy_absolute_error > self.tolerances.max_energy_absolute_error:
            warnings.append(
                f"Max energy absolute error {comparison.max_energy_absolute_error:.2e} "
                f"exceeds tolerance {self.tolerances.max_energy_absolute_error:.2e}."
            )
        if (
            comparison.max_energy_relative_error is not None
            and comparison.max_energy_relative_error > self.tolerances.max_energy_relative_error
        ):
            warnings.append(
                f"Max energy relative error {comparison.max_energy_relative_error:.2e} "
                f"exceeds tolerance {self.tolerances.max_energy_relative_error:.2e}."
            )

        # Wavefunction & state checks
        min_fidelity = min(s.state_fidelity for s in comparison.state_comparisons)
        max_norm_err = max(s.normalization_error for s in comparison.state_comparisons)

        if comparison.max_wavefunction_l2_error > self.tolerances.max_wavefunction_l2_error:
            warnings.append(
                f"Max wavefunction L2 error {comparison.max_wavefunction_l2_error:.2e} "
                f"exceeds tolerance {self.tolerances.max_wavefunction_l2_error:.2e}."
            )
        if min_fidelity < self.tolerances.min_state_fidelity:
            warnings.append(
                f"Minimum state fidelity {min_fidelity:.6f} is below "
                f"threshold {self.tolerances.min_state_fidelity:.6f}."
            )
        if max_norm_err > self.tolerances.max_normalization_error:
            warnings.append(
                f"Max normalization error {max_norm_err:.2e} exceeds "
                f"tolerance {self.tolerances.max_normalization_error:.2e}."
            )
        if comparison.max_orthogonality_error > self.tolerances.max_orthogonality_error:
            warnings.append(
                f"Max orthogonality error {comparison.max_orthogonality_error:.2e} exceeds "
                f"tolerance {self.tolerances.max_orthogonality_error:.2e}."
            )

        # Convergence checks (if provided)
        conv_dict: Optional[Dict[str, Any]] = None
        if convergence_result is not None:
            conv_dict = {
                "accuracy_order": convergence_result.accuracy_order,
                "grid_resolutions": [p.grid_points for p in convergence_result.points],
                "dx_values": [p.dx for p in convergence_result.points],
                "ground_energy_errors": [p.ground_energy_error for p in convergence_result.points],
                "ground_wavefunction_l2_errors": [p.ground_wavefunction_l2_error for p in convergence_result.points],
                "estimated_energy_order": convergence_result.estimated_energy_order,
                "estimated_wavefunction_order": convergence_result.estimated_wavefunction_order,
                "is_monotonically_converging": convergence_result.is_monotonically_converging,
                "is_numerically_stable": convergence_result.is_numerically_stable
            }
            if not convergence_result.is_numerically_stable:
                warnings.append("Convergence study did not demonstrate numerical stability.")
            if (
                convergence_result.estimated_energy_order is not None
                and convergence_result.estimated_energy_order < self.tolerances.min_convergence_order
            ):
                warnings.append(
                    f"Estimated energy convergence order {convergence_result.estimated_energy_order:.2f} "
                    f"is below minimum {self.tolerances.min_convergence_order:.2f}."
                )

        wf_metrics = {
            "max_l2_error": comparison.max_wavefunction_l2_error,
            "min_state_fidelity": float(min_fidelity),
            "max_normalization_error": float(max_norm_err),
            "max_orthogonality_error": comparison.max_orthogonality_error,
            "per_state": [asdict(s) for s in comparison.state_comparisons]
        }

        return ScientificValidationReport(
            potential_model=analytical_solution.name,
            physical_parameters=analytical_solution.parameters,
            grid_resolution=grid_resolution,
            grid_spacing=grid_spacing,
            domain=domain,
            quantum_numbers=[s.quantum_number for s in comparison.state_comparisons],
            numerical_energies=[s.numerical_energy for s in comparison.state_comparisons],
            analytical_energies=[s.analytical_energy for s in comparison.state_comparisons],
            energy_absolute_errors=[s.energy_absolute_error for s in comparison.state_comparisons],
            energy_relative_errors=[s.energy_relative_error for s in comparison.state_comparisons],
            wavefunction_metrics=wf_metrics,
            convergence_info=conv_dict,
            tolerances=asdict(self.tolerances),
            warnings=warnings,
            validation_passed=(len(warnings) == 0)
        )
