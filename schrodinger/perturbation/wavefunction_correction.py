import math
from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from scipy import integrate
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.wavefunctions.orthogonality import OrthogonalityAnalyzer
from schrodinger.coupled.multi_wavefunction import MultiComponentWavefunction
from .perturbation_operator import PerturbationMatrix, QuantumState


@dataclass
class WavefunctionCorrectionResult:
    """First-order perturbative wavefunction correction summary for state n."""
    state_index: int
    lambda_strength: float
    unperturbed_state: QuantumState
    first_order_correction_array: np.ndarray
    corrected_state: QuantumState
    correction_norm: float
    overlap_with_unperturbed: complex
    overlap_probability: float
    orthogonality_to_unperturbed_error: float
    small_denominator_detected: bool
    warnings: List[str]


class WavefunctionCorrectionEngine:
    """
    Computes first-order Rayleigh-Schrödinger wavefunction corrections:
        |n^(1)> = sum_{m != n} (<m^(0) | H' | n^(0)> / (E_n^(0) - E_m^(0))) |m^(0)>
        |psi_n^corr> = N * (|n^(0)> + lambda * |n^(1)>)
    Supports both single-channel Wavefunction and MultiComponentWavefunction states.
    """

    def __init__(
        self,
        denominator_tolerance: float = 1e-6,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ):
        if denominator_tolerance <= 0:
            raise ValueError("denominator_tolerance must be strictly positive.")
        self.denominator_tolerance = float(denominator_tolerance)
        self.integration_method = integration_method

    def _align_phase_to_reference(
        self,
        candidate: np.ndarray,
        reference: np.ndarray,
        x: np.ndarray,
        is_coupled: bool
    ) -> np.ndarray:
        """Rotates global complex phase of `candidate` so <reference | candidate> is real positive."""
        if is_coupled:
            integrand = np.sum(np.conj(reference) * candidate, axis=0)
        else:
            integrand = np.conj(reference) * candidate

        overlap = complex(
            integrate.simpson(y=integrand, x=x)
            if self.integration_method == IntegrationMethod.SIMPSON
            else integrate.trapezoid(y=integrand, x=x)
        )
        if abs(overlap) > 1e-14:
            phase = float(np.angle(overlap))
            return candidate * np.exp(-1j * phase)
        return candidate

    def correct_state(
        self,
        state_index: int,
        eigenstates: Sequence[QuantumState],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0,
        corrected_energy: Optional[float] = None
    ) -> WavefunctionCorrectionResult:
        """Computes normalized 1st-order corrected wavefunction for `state_index`."""
        if not math.isfinite(lambda_strength):
            raise ValueError("lambda_strength must be finite.")
        n_states = len(eigenstates)
        if n_states != perturbation_matrix.dimension:
            raise ValueError("eigenstates length must match perturbation_matrix dimension.")
        if state_index < 0 or state_index >= n_states:
            raise IndexError(f"state_index {state_index} out of bounds.")

        base_state = eigenstates[state_index]
        is_coupled = isinstance(base_state, MultiComponentWavefunction)
        x = base_state.x
        e0 = base_state.energy
        w_mat = perturbation_matrix.matrix

        base_arr = base_state.components if is_coupled else base_state.psi
        psi_1 = np.zeros_like(base_arr, dtype=np.complex128)
        small_denom = False
        warnings: List[str] = []

        for m in range(n_states):
            if m == state_index:
                continue
            denom = float(e0 - eigenstates[m].energy)
            w_mn = w_mat[m, state_index]
            if abs(denom) <= self.denominator_tolerance:
                if abs(w_mn) > 1e-12:
                    small_denom = True
                    warnings.append(
                        f"Skipped degenerate/near-degenerate state m={m} (|dE|={abs(denom):.2e})."
                    )
                continue
            coeff = w_mn / denom
            m_arr = eigenstates[m].components if is_coupled else eigenstates[m].psi
            psi_1 += coeff * m_arr

        # Calculate ||lambda * |n^(1)>|| and orthogonality <n^(0) | n^(1)>
        scaled_corr = lambda_strength * psi_1
        if is_coupled:
            corr_dens = np.sum(np.abs(scaled_corr) ** 2, axis=0)
            ortho_integrand = np.sum(np.conj(base_arr) * psi_1, axis=0)
        else:
            corr_dens = np.abs(scaled_corr) ** 2
            ortho_integrand = np.conj(base_arr) * psi_1

        corr_norm_sq = float(integrate.simpson(y=corr_dens, x=x))
        corr_norm = float(np.sqrt(max(0.0, corr_norm_sq)))
        ortho_err = float(abs(integrate.simpson(y=ortho_integrand, x=x)))

        if corr_norm > 0.5:
            warnings.append(
                f"Large first-order wavefunction correction norm ||lambda |n^(1)>|| = {corr_norm:.3f}; "
                "first-order perturbative expansion may be unreliable."
            )

        raw_corrected = base_arr + scaled_corr
        aligned_corrected = self._align_phase_to_reference(raw_corrected, base_arr, x, is_coupled)

        e_out = (
            float(corrected_energy)
            if corrected_energy is not None
            else float(e0 + lambda_strength * np.real(w_mat[state_index, state_index]))
        )

        if is_coupled:
            corrected_wf: QuantumState = MultiComponentWavefunction(
                x=x,
                components=aligned_corrected,
                energy=e_out,
                channel_labels=base_state.channel_labels,
                normalize=True,
                integration_method=self.integration_method
            )
            overlap = base_state.inner_product(corrected_wf)
        else:
            normed_psi, _ = NormalizationEngine.normalize(
                aligned_corrected, x, method=self.integration_method
            )
            corrected_wf = Wavefunction(x=x, psi=normed_psi, energy=e_out)
            overlap = OrthogonalityAnalyzer.calculate_overlap(base_state.psi, corrected_wf.psi, x)

        return WavefunctionCorrectionResult(
            state_index=int(state_index),
            lambda_strength=float(lambda_strength),
            unperturbed_state=base_state,
            first_order_correction_array=psi_1,
            corrected_state=corrected_wf,
            correction_norm=corr_norm,
            overlap_with_unperturbed=complex(overlap),
            overlap_probability=float(min(1.0, abs(overlap) ** 2)),
            orthogonality_to_unperturbed_error=ortho_err,
            small_denominator_detected=small_denom,
            warnings=warnings
        )

    def correct_states(
        self,
        eigenstates: Sequence[QuantumState],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0,
        target_states: Optional[Sequence[int]] = None,
        corrected_energies: Optional[Sequence[float]] = None
    ) -> List[WavefunctionCorrectionResult]:
        """Computes 1st-order corrected wavefunctions for all `target_states`."""
        indices = list(range(len(eigenstates))) if target_states is None else list(target_states)
        results: List[WavefunctionCorrectionResult] = []
        for pos, idx in enumerate(indices):
            e_corr = corrected_energies[pos] if corrected_energies is not None else None
            results.append(
                self.correct_state(
                    state_index=idx,
                    eigenstates=eigenstates,
                    perturbation_matrix=perturbation_matrix,
                    lambda_strength=lambda_strength,
                    corrected_energy=e_corr
                )
            )
        return results

    @staticmethod
    def max_cross_orthogonality_error(corrected_states: Sequence[QuantumState]) -> float:
        """Evaluates max_{i != j} |<psi_i^corr | psi_j^corr>| among corrected eigenstates."""
        if len(corrected_states) < 2:
            return 0.0
        max_ov = 0.0
        for i in range(len(corrected_states)):
            for j in range(i + 1, len(corrected_states)):
                si, sj = corrected_states[i], corrected_states[j]
                if isinstance(si, MultiComponentWavefunction) and isinstance(sj, MultiComponentWavefunction):
                    ov = abs(si.inner_product(sj))
                elif isinstance(si, Wavefunction) and isinstance(sj, Wavefunction):
                    ov = abs(OrthogonalityAnalyzer.calculate_overlap(si.psi, sj.psi, si.x))
                else:
                    raise TypeError("All corrected_states must have identical quantum state types.")
                max_ov = max(max_ov, float(ov))
        return max_ov
