import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np
from .perturbation_operator import PerturbationOperator, PerturbationMatrix, QuantumState


@dataclass
class NonDegenerateStateResult:
    """Perturbative energy correction summary for a single non-degenerate level n."""
    state_index: int
    lambda_strength: float
    perturbation_order: int
    unperturbed_energy: float
    first_order_correction: float
    second_order_correction: float
    total_approximate_energy: float
    min_energy_denominator: float
    max_mixing_ratio: float
    small_denominator_detected: bool
    is_perturbatively_valid: bool
    warnings: List[str]


class NonDegeneratePerturbationSolver:
    """
    Computes 1st-order and 2nd-order non-degenerate Rayleigh-Schrödinger
    perturbation energy corrections:
        E_n^(1) = <n^(0) | H' | n^(0)>
        E_n^(2) = sum_{m != n} |<m^(0) | H' | n^(0)>|^2 / (E_n^(0) - E_m^(0))
        E_n ≈ E_n^(0) + lambda * E_n^(1) + lambda^2 * E_n^(2)
    """

    def __init__(
        self,
        denominator_tolerance: float = 1e-6,
        mixing_warning_threshold: float = 0.25
    ):
        if denominator_tolerance <= 0:
            raise ValueError("denominator_tolerance must be strictly positive.")
        if mixing_warning_threshold <= 0:
            raise ValueError("mixing_warning_threshold must be strictly positive.")

        self.denominator_tolerance = float(denominator_tolerance)
        self.mixing_warning_threshold = float(mixing_warning_threshold)

    def solve_state(
        self,
        state_index: int,
        unperturbed_energies: Sequence[float],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0,
        order: int = 2
    ) -> NonDegenerateStateResult:
        """Calculates 1st- or 2nd-order energy correction for a single state index."""
        if order not in (1, 2):
            raise ValueError(f"Supported non-degenerate perturbation orders are 1 or 2. Received: {order}")
        if not math.isfinite(lambda_strength):
            raise ValueError(f"lambda_strength must be finite. Received: {lambda_strength}")

        energies = np.asarray(unperturbed_energies, dtype=np.float64)
        n_states = len(energies)
        if n_states != perturbation_matrix.dimension:
            raise ValueError(
                f"unperturbed_energies length ({n_states}) != perturbation_matrix dimension ({perturbation_matrix.dimension})."
            )
        if state_index < 0 or state_index >= n_states:
            raise IndexError(f"state_index {state_index} out of bounds for {n_states} states.")

        e0 = float(energies[state_index])
        w_mat = perturbation_matrix.matrix
        e1 = float(np.real(w_mat[state_index, state_index]))

        e2 = 0.0
        min_denom = float("inf")
        max_mixing = 0.0
        small_denom = False
        warnings: List[str] = []

        for m in range(n_states):
            if m == state_index:
                continue
            denom = float(e0 - energies[m])
            abs_denom = abs(denom)
            w_mn = w_mat[m, state_index]
            abs_w = float(abs(w_mn))

            if abs_w > 1e-14:
                min_denom = min(min_denom, abs_denom)

            if abs_denom <= self.denominator_tolerance:
                if abs_w > 1e-12:
                    small_denom = True
                    warnings.append(
                        f"Small energy denominator |E_{state_index}^(0) - E_{m}^(0)| = {abs_denom:.2e} "
                        f"with non-zero coupling |W_{m},{state_index}| = {abs_w:.2e}. "
                        "Use DegeneratePerturbationSolver."
                    )
                continue

            mixing = abs(lambda_strength * abs_w / denom)
            max_mixing = max(max_mixing, mixing)

            if order >= 2:
                e2 += (abs_w ** 2) / denom

        if order == 1:
            total_approx = e0 + lambda_strength * e1
        else:
            total_approx = e0 + lambda_strength * e1 + (lambda_strength ** 2) * e2

        if max_mixing >= self.mixing_warning_threshold:
            warnings.append(
                f"Large state-mixing ratio max |lambda * W_mn / dE| = {max_mixing:.3f} >= "
                f"{self.mixing_warning_threshold:.2f}; perturbation expansion may be inaccurate."
            )

        if order >= 2 and abs(lambda_strength * e1) > 1e-12:
            corr_ratio = abs((lambda_strength ** 2) * e2) / abs(lambda_strength * e1)
            if corr_ratio > 0.5:
                warnings.append(
                    f"Second-order correction magnitude ({abs((lambda_strength**2)*e2):.2e}) exceeds 50% "
                    f"of first-order correction ({abs(lambda_strength*e1):.2e}); series convergence uncertain."
                )

        if min_denom == float("inf"):
            min_denom = 0.0 if n_states <= 1 else float(np.min(np.abs(np.delete(energies, state_index) - e0)))

        is_valid = (not small_denom) and (max_mixing < self.mixing_warning_threshold)

        return NonDegenerateStateResult(
            state_index=int(state_index),
            lambda_strength=float(lambda_strength),
            perturbation_order=int(order),
            unperturbed_energy=e0,
            first_order_correction=e1,
            second_order_correction=float(e2) if order >= 2 else 0.0,
            total_approximate_energy=float(total_approx),
            min_energy_denominator=float(min_denom),
            max_mixing_ratio=float(max_mixing),
            small_denominator_detected=small_denom,
            is_perturbatively_valid=is_valid,
            warnings=warnings
        )

    def solve(
        self,
        unperturbed_energies: Sequence[float],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0,
        order: int = 2,
        target_states: Optional[Sequence[int]] = None
    ) -> List[NonDegenerateStateResult]:
        """Calculates non-degenerate energy corrections for all requested `target_states`."""
        indices = list(range(len(unperturbed_energies))) if target_states is None else list(target_states)
        return [
            self.solve_state(
                state_index=idx,
                unperturbed_energies=unperturbed_energies,
                perturbation_matrix=perturbation_matrix,
                lambda_strength=lambda_strength,
                order=order
            )
            for idx in indices
        ]

    def solve_from_eigenstates(
        self,
        eigenstates: Sequence[QuantumState],
        perturbation: PerturbationOperator,
        lambda_strength: float = 1.0,
        order: int = 2,
        target_states: Optional[Sequence[int]] = None
    ) -> List[NonDegenerateStateResult]:
        """Convenience method computing W_{mn} directly from eigenstates and PerturbationOperator."""
        energies = [st.energy for st in eigenstates]
        w_mat = PerturbationMatrix.from_eigenstates(eigenstates, perturbation)
        return self.solve(
            unperturbed_energies=energies,
            perturbation_matrix=w_mat,
            lambda_strength=lambda_strength,
            order=order,
            target_states=target_states
        )
