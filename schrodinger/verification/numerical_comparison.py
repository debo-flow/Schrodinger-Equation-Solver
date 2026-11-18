from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import integrate
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.quantum_observables import QuantumObservables
from schrodinger.wavefunctions.orthogonality import OrthogonalityAnalyzer
from .analytical_solver import AnalyticalSolution
from .error_analysis import ErrorAnalyzer


@dataclass
class StateComparisonResult:
    """Comparison metrics for a single eigenstate against its analytical reference."""
    quantum_number: int
    numerical_energy: float
    analytical_energy: float
    energy_absolute_error: float
    energy_relative_error: Optional[float]
    wavefunction_l2_error: float
    wavefunction_rmse: float
    wavefunction_max_error: float
    state_fidelity: float
    phase_shift_radians: float
    probability_density_rmse: float
    probability_density_max_error: float
    expectation_x_error: float
    variance_x_error: float
    normalization_error: float


@dataclass
class SpectrumComparisonResult:
    """Aggregated comparison metrics across multiple eigenstates."""
    model_name: str
    num_states: int
    state_comparisons: List[StateComparisonResult]
    max_energy_absolute_error: float
    max_energy_relative_error: Optional[float]
    energy_rmse: float
    max_wavefunction_l2_error: float
    max_orthogonality_error: float


class NumericalComparisonEngine:
    """
    Compares numerical eigenstates and spectra against exact analytical solutions,
    accounting for arbitrary global phase ambiguity.
    """

    @staticmethod
    def align_global_phase(
        psi_numerical: np.ndarray,
        psi_analytical: np.ndarray,
        x: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Rotates psi_numerical by an optimal global phase factor exp(i * theta)
        so that the overlap <psi_aligned | psi_analytical> is real and non-negative.

        Returns:
            Tuple[np.ndarray, float]: (phase-aligned wavefunction, applied phase shift in radians).
        """
        overlap = complex(integrate.simpson(y=np.conj(psi_numerical) * psi_analytical, x=x))
        if abs(overlap) < 1e-15:
            return psi_numerical.copy(), 0.0

        phase_angle = float(np.angle(overlap))
        phase_factor = np.exp(1j * phase_angle)
        return psi_numerical * phase_factor, phase_angle

    @staticmethod
    def calculate_fidelity(
        psi_numerical: np.ndarray,
        psi_analytical: np.ndarray,
        x: np.ndarray
    ) -> float:
        """Computes state fidelity F = |<psi_num | psi_ana>|^2."""
        overlap = complex(integrate.simpson(y=np.conj(psi_numerical) * psi_analytical, x=x))
        return float(min(1.0, abs(overlap) ** 2))

    @classmethod
    def compare_state(
        cls,
        numerical_wf: Wavefunction,
        analytical_solution: AnalyticalSolution,
        quantum_number: int,
        zero_threshold: float = 1e-12
    ) -> StateComparisonResult:
        """Performs full state-by-state comparison against an analytical eigenstate."""
        x = numerical_wf.x
        e_num = numerical_wf.energy
        e_ana = analytical_solution.energy(quantum_number)

        psi_ana = analytical_solution.wavefunction(quantum_number, x)
        psi_aligned, phase_shift = cls.align_global_phase(numerical_wf.psi, psi_ana, x)

        # Energy errors
        e_abs = float(ErrorAnalyzer.absolute_error(e_num, e_ana))
        e_rel = ErrorAnalyzer.relative_error(e_num, e_ana, zero_threshold=zero_threshold)

        # Wavefunction errors (phase-aligned)
        wf_l2 = ErrorAnalyzer.l2_wavefunction_error(psi_aligned, psi_ana, x)
        wf_rmse = ErrorAnalyzer.rmse(psi_aligned, psi_ana)
        wf_max = ErrorAnalyzer.max_absolute_error(psi_aligned, psi_ana)
        fidelity = cls.calculate_fidelity(numerical_wf.psi, psi_ana, x)

        # Probability density errors (intrinsically phase-invariant)
        rho_num = numerical_wf.probability_density
        rho_ana = np.abs(psi_ana) ** 2
        rho_rmse = ErrorAnalyzer.rmse(rho_num, rho_ana)
        rho_max = ErrorAnalyzer.max_absolute_error(rho_num, rho_ana)

        # Observable errors
        exp_x_ana = QuantumObservables.expectation_x(psi_ana, x)
        var_x_ana = QuantumObservables.variance_x(psi_ana, x)
        exp_x_err = abs(numerical_wf.expectation_x - exp_x_ana)
        var_x_err = abs(numerical_wf.variance_x - var_x_ana)

        # Normalization error
        norm_integral = float(integrate.simpson(y=rho_num, x=x))
        norm_err = abs(norm_integral - 1.0)

        return StateComparisonResult(
            quantum_number=quantum_number,
            numerical_energy=e_num,
            analytical_energy=e_ana,
            energy_absolute_error=e_abs,
            energy_relative_error=e_rel,
            wavefunction_l2_error=wf_l2,
            wavefunction_rmse=wf_rmse,
            wavefunction_max_error=wf_max,
            state_fidelity=fidelity,
            phase_shift_radians=phase_shift,
            probability_density_rmse=rho_rmse,
            probability_density_max_error=rho_max,
            expectation_x_error=float(exp_x_err),
            variance_x_error=float(var_x_err),
            normalization_error=float(norm_err)
        )

    @classmethod
    def compare_spectrum(
        cls,
        numerical_wavefunctions: List[Wavefunction],
        analytical_solution: AnalyticalSolution,
        zero_threshold: float = 1e-12
    ) -> SpectrumComparisonResult:
        """Compares a sequence of lowest numerical eigenstates against analytical references."""
        if not numerical_wavefunctions:
            raise ValueError("numerical_wavefunctions list cannot be empty.")

        num_states = len(numerical_wavefunctions)
        q_nums = analytical_solution.quantum_numbers(num_states)

        state_results = [
            cls.compare_state(wf, analytical_solution, n, zero_threshold=zero_threshold)
            for wf, n in zip(numerical_wavefunctions, q_nums)
        ]

        # Orthogonality check among numerical eigenstates
        max_ortho_err = 0.0
        for i in range(num_states):
            for j in range(i + 1, num_states):
                overlap = abs(
                    OrthogonalityAnalyzer.calculate_overlap(
                        numerical_wavefunctions[i].psi,
                        numerical_wavefunctions[j].psi,
                        numerical_wavefunctions[i].x
                    )
                )
                max_ortho_err = max(max_ortho_err, float(overlap))

        e_num = np.array([r.numerical_energy for r in state_results], dtype=np.float64)
        e_ana = np.array([r.analytical_energy for r in state_results], dtype=np.float64)
        spec_dev = ErrorAnalyzer.energy_spectrum_deviation(e_num, e_ana, zero_threshold=zero_threshold)

        return SpectrumComparisonResult(
            model_name=analytical_solution.name,
            num_states=num_states,
            state_comparisons=state_results,
            max_energy_absolute_error=spec_dev["max_absolute_error"],
            max_energy_relative_error=spec_dev["max_relative_error"],
            energy_rmse=spec_dev["rmse"],
            max_wavefunction_l2_error=max(r.wavefunction_l2_error for r in state_results),
            max_orthogonality_error=max_ortho_err
        )
