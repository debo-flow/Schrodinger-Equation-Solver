import math
from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from scipy import linalg
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.wavefunctions.phase_manager import PhaseManager
from schrodinger.coupled.multi_wavefunction import MultiComponentWavefunction
from schrodinger.analysis.hermiticity import HermiticityAnalyzer
from .perturbation_operator import PerturbationOperator, PerturbationMatrix, QuantumState


@dataclass
class DegenerateSubspaceResult:
    """First-order degenerate perturbation splitting and lifted eigenstates for one subspace."""
    subspace_indices: List[int]
    subspace_dimension: int
    unperturbed_energy: float
    lambda_strength: float
    perturbation_submatrix: np.ndarray
    first_order_shifts: np.ndarray
    split_energies: np.ndarray
    transformation_matrix: np.ndarray
    corrected_eigenstates: List[QuantumState]
    degeneracy_lifted: bool
    splitting_spread: float
    remaining_degeneracy_groups: List[List[int]]


class DegeneratePerturbationSolver:
    """
    Implements first-order degenerate perturbation theory:
      1. Identifies degenerate and nearly degenerate energy subspaces.
      2. Extracts the d x d Hermitian perturbation submatrix W^(D).
      3. Diagonalizes lambda * W^(D) (real symmetric or complex Hermitian) via eigh.
      4. Computes first-order energy splittings and reconstructs lifted eigenstates.
    """

    def __init__(
        self,
        degeneracy_tolerance: float = 1e-5,
        lifting_tolerance: float = 1e-7,
        hermitian_tolerance: float = 1e-8,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ):
        if degeneracy_tolerance <= 0 or lifting_tolerance <= 0 or hermitian_tolerance <= 0:
            raise ValueError("All tolerances must be strictly positive.")
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.lifting_tolerance = float(lifting_tolerance)
        self.hermitian_tolerance = float(hermitian_tolerance)
        self.integration_method = integration_method

    def detect_degenerate_subspaces(self, unperturbed_energies: Sequence[float]) -> List[List[int]]:
        """
        Groups sorted unperturbed energy indices into degenerate subspaces
        whenever consecutive energies satisfy |E_{i} - E_{i-1}| <= degeneracy_tolerance.
        Returns only subspaces with dimension >= 2.
        """
        energies = np.asarray(unperturbed_energies, dtype=np.float64)
        if len(energies) < 2:
            return []

        order = np.argsort(energies)
        subspaces: List[List[int]] = []
        current_group = [int(order[0])]

        for k in range(1, len(order)):
            prev_idx = int(order[k - 1])
            curr_idx = int(order[k])
            if abs(energies[curr_idx] - energies[prev_idx]) <= self.degeneracy_tolerance:
                current_group.append(curr_idx)
            else:
                if len(current_group) > 1:
                    subspaces.append(current_group)
                current_group = [curr_idx]

        if len(current_group) > 1:
            subspaces.append(current_group)

        return subspaces

    def _find_remaining_degeneracies(self, split_energies: np.ndarray) -> List[List[int]]:
        groups: List[List[int]] = []
        curr = [0]
        for i in range(1, len(split_energies)):
            if abs(split_energies[i] - split_energies[i - 1]) <= self.lifting_tolerance:
                curr.append(i)
            else:
                if len(curr) > 1:
                    groups.append(curr)
                curr = [i]
        if len(curr) > 1:
            groups.append(curr)
        return groups

    def solve_subspace(
        self,
        subspace_indices: Sequence[int],
        eigenstates: Sequence[QuantumState],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0
    ) -> DegenerateSubspaceResult:
        """Diagonalizes lambda * W^(D) within `subspace_indices` and builds lifted states."""
        if not math.isfinite(lambda_strength):
            raise ValueError("lambda_strength must be finite.")
        indices = [int(i) for i in subspace_indices]
        if len(indices) < 2:
            raise ValueError("Degenerate subspace must contain at least 2 states.")

        sub_w = perturbation_matrix.submatrix(indices)
        is_herm, dev = HermiticityAnalyzer(tolerance=self.hermitian_tolerance).is_hermitian(sub_w)
        if not is_herm:
            raise ValueError(
                f"Subspace perturbation matrix is not Hermitian (max dev = {dev:.2e})."
            )

        # Symmetrize machine-precision noise before eigh
        scaled_w = lambda_strength * (0.5 * (sub_w + np.conj(sub_w.T)))
        shifts, u_mat = linalg.eigh(scaled_w)
        shifts = np.asarray(shifts, dtype=np.float64)

        unperturbed_e = float(np.mean([eigenstates[i].energy for i in indices]))
        split_energies = unperturbed_e + shifts

        # Reconstruct corrected ("good") eigenstates: |psi_k> = sum_a U_{a, k} |subspace[a]>
        first_state = eigenstates[indices[0]]
        is_coupled = isinstance(first_state, MultiComponentWavefunction)
        x = first_state.x

        corrected_states: List[QuantumState] = []
        for k in range(len(indices)):
            coeffs = u_mat[:, k]
            if is_coupled:
                combined = np.zeros_like(first_state.components, dtype=np.complex128)
                for a, idx in enumerate(indices):
                    combined += coeffs[a] * eigenstates[idx].components
                # Standardize global phase
                flat_max = int(np.argmax(np.abs(combined)))
                peak_val = combined.ravel()[flat_max]
                if abs(peak_val) > 0:
                    combined *= np.exp(-1j * float(np.angle(peak_val)))
                state_k: QuantumState = MultiComponentWavefunction(
                    x=x,
                    components=combined,
                    energy=float(split_energies[k]),
                    channel_labels=first_state.channel_labels,
                    normalize=True,
                    integration_method=self.integration_method
                )
            else:
                combined = np.zeros_like(first_state.psi, dtype=np.complex128)
                for a, idx in enumerate(indices):
                    combined += coeffs[a] * eigenstates[idx].psi
                normed, _ = NormalizationEngine.normalize(combined, x, method=self.integration_method)
                std_psi = PhaseManager.standardize_phase(normed)
                state_k = Wavefunction(x=x, psi=std_psi, energy=float(split_energies[k]))

            corrected_states.append(state_k)

        spread = float(split_energies[-1] - split_energies[0])
        remaining = self._find_remaining_degeneracies(split_energies)
        lifted = spread > self.lifting_tolerance

        return DegenerateSubspaceResult(
            subspace_indices=indices,
            subspace_dimension=len(indices),
            unperturbed_energy=unperturbed_e,
            lambda_strength=float(lambda_strength),
            perturbation_submatrix=sub_w,
            first_order_shifts=shifts,
            split_energies=split_energies,
            transformation_matrix=u_mat,
            corrected_eigenstates=corrected_states,
            degeneracy_lifted=lifted,
            splitting_spread=spread,
            remaining_degeneracy_groups=remaining
        )

    def solve_all_subspaces(
        self,
        eigenstates: Sequence[QuantumState],
        perturbation_matrix: PerturbationMatrix,
        lambda_strength: float = 1.0
    ) -> List[DegenerateSubspaceResult]:
        """Automatically detects all degenerate subspaces and solves each one."""
        energies = [st.energy for st in eigenstates]
        subspaces = self.detect_degenerate_subspaces(energies)
        return [
            self.solve_subspace(sub, eigenstates, perturbation_matrix, lambda_strength=lambda_strength)
            for sub in subspaces
        ]

    def solve_from_operator(
        self,
        eigenstates: Sequence[QuantumState],
        perturbation: PerturbationOperator,
        lambda_strength: float = 1.0
    ) -> List[DegenerateSubspaceResult]:
        """Computes W_{mn} from `perturbation` and diagonalizes all degenerate subspaces."""
        w_mat = PerturbationMatrix.from_eigenstates(eigenstates, perturbation)
        return self.solve_all_subspaces(eigenstates, w_mat, lambda_strength=lambda_strength)
