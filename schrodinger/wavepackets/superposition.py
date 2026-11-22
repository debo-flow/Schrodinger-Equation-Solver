from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.wavefunctions.orthogonality import OrthogonalityAnalyzer


@dataclass
class SuperpositionReport:
    """Diagnostic summary of a constructed quantum superposition state."""
    num_components: int
    normalized_coefficients: np.ndarray
    occupation_probabilities: np.ndarray
    expected_energy: float
    basis_is_orthogonal: bool
    max_basis_overlap: float
    continuous_norm: float


class SuperpositionEngine:
    """
    Constructs and validates normalized quantum state superpositions:
        psi(x) = sum_n c_n * psi_n(x)
    supporting complex amplitudes, basis orthogonality checks, and energy expectations.
    """

    @staticmethod
    def validate_components(
        eigenstates: Sequence[Wavefunction],
        coefficients: Sequence[complex]
    ) -> np.ndarray:
        """Validates basis states and complex coefficients, returning complex128 array."""
        if not eigenstates:
            raise ValueError("Superposition requires at least one basis eigenstate.")
        if len(eigenstates) != len(coefficients):
            raise ValueError(
                f"Number of eigenstates ({len(eigenstates)}) must match number of coefficients ({len(coefficients)})."
            )

        x_ref = eigenstates[0].x
        for idx, wf in enumerate(eigenstates):
            if not isinstance(wf, Wavefunction):
                raise TypeError(f"Element {idx} is not a Wavefunction instance.")
            if wf.x.shape != x_ref.shape or not np.allclose(wf.x, x_ref):
                raise ValueError(f"Eigenstate {idx} does not share the same spatial grid.")

        coeff_arr = np.asarray(coefficients, dtype=np.complex128)
        if not np.all(np.isfinite(coeff_arr)):
            raise ValueError("Superposition coefficients must be finite complex numbers.")
        if np.linalg.norm(coeff_arr) <= 1e-15:
            raise ValueError("At least one superposition coefficient must be non-zero.")

        return coeff_arr

    @staticmethod
    def check_basis_orthogonality(
        eigenstates: Sequence[Wavefunction],
        tolerance: float = 1e-4
    ) -> tuple:
        """
        Computes pairwise overlaps |<psi_i | psi_j>| for i != j.
        Returns (is_orthogonal, max_off_diagonal_overlap).
        """
        n = len(eigenstates)
        if n < 2:
            return True, 0.0

        x = eigenstates[0].x
        max_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                overlap = abs(OrthogonalityAnalyzer.calculate_overlap(eigenstates[i].psi, eigenstates[j].psi, x))
                max_overlap = max(max_overlap, float(overlap))

        return (max_overlap <= tolerance), max_overlap

    @classmethod
    def create_with_report(
        cls,
        eigenstates: Sequence[Wavefunction],
        coefficients: Sequence[complex],
        orthogonality_tolerance: float = 1e-4,
        enforce_orthogonality: bool = False,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> tuple:
        """
        Builds a normalized superposition Wavefunction and returns (Wavefunction, SuperpositionReport).
        """
        coeff_arr = cls.validate_components(eigenstates, coefficients)
        is_ortho, max_overlap = cls.check_basis_orthogonality(eigenstates, tolerance=orthogonality_tolerance)

        if enforce_orthogonality and not is_ortho:
            raise ValueError(
                f"Basis states failed orthogonality check: max overlap {max_overlap:.2e} > {orthogonality_tolerance:.2e}."
            )

        x_ref = eigenstates[0].x
        psi_sum = np.zeros_like(x_ref, dtype=np.complex128)
        for c, wf in zip(coeff_arr, eigenstates):
            psi_sum += c * wf.psi

        psi_norm, raw_norm = NormalizationEngine.normalize(psi_sum, x_ref, method=integration_method)
        norm_coeffs = coeff_arr / raw_norm
        probs = np.abs(norm_coeffs) ** 2
        probs /= np.sum(probs)

        energies = np.array([wf.energy for wf in eigenstates], dtype=np.float64)
        exp_energy = float(np.sum(probs * energies))

        wf_out = Wavefunction(x=x_ref, psi=psi_norm, energy=exp_energy)
        final_norm = NormalizationEngine.calculate_norm(wf_out.psi, x_ref, method=integration_method)

        report = SuperpositionReport(
            num_components=len(eigenstates),
            normalized_coefficients=norm_coeffs,
            occupation_probabilities=probs,
            expected_energy=exp_energy,
            basis_is_orthogonal=is_ortho,
            max_basis_overlap=max_overlap,
            continuous_norm=final_norm
        )
        return wf_out, report

    @classmethod
    def create(
        cls,
        eigenstates: Sequence[Wavefunction],
        coefficients: Sequence[complex],
        orthogonality_tolerance: float = 1e-4,
        enforce_orthogonality: bool = False,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Wavefunction:
        """Constructs and returns the normalized superposition Wavefunction."""
        wf, _ = cls.create_with_report(
            eigenstates=eigenstates,
            coefficients=coefficients,
            orthogonality_tolerance=orthogonality_tolerance,
            enforce_orthogonality=enforce_orthogonality,
            integration_method=integration_method
        )
        return wf
