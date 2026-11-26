from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from scipy import integrate
from schrodinger.numerical.grid import BoundaryCondition
from schrodinger.solvers.solver_configuration import SolverConfig, SolverMethod
from schrodinger.solvers.dense_solver import DenseSolver
from schrodinger.solvers.sparse_solver import SparseSolver
from schrodinger.solvers.eigenpair_validation import EigenpairValidator
from schrodinger.solvers.degeneracy import DegeneracyAnalyzer
from .coupled_hamiltonian import CoupledHamiltonian
from .multi_wavefunction import MultiComponentWavefunction


@dataclass
class SpectrumComparisonWithUncoupled:
    """Comparison between coupled-channel energies and the zero-coupling limit."""
    coupled_energies: np.ndarray
    uncoupled_energies: np.ndarray
    energy_shifts: np.ndarray
    max_energy_shift: float
    rms_energy_shift: float


@dataclass
class CoupledSolverResult:
    """Structured output of CoupledEigenvalueSolver."""
    energies: np.ndarray
    eigenstates: List[MultiComponentWavefunction]
    residuals: List[float]
    max_residual: float
    degeneracies: List[List[int]]
    method_used: str
    hamiltonian_dimension: int
    num_channels: int
    converged: bool
    uncoupled_comparison: Optional[SpectrumComparisonWithUncoupled]


class CoupledEigenvalueSolver:
    """
    Solves the coupled-channel time-independent Schrödinger eigenvalue problem:
        (T (x) I_N + V(x)) Psi_k(x) = E_k Psi_k(x)
    Reconstructs completely normalized MultiComponentWavefunction eigenstates,
    computes eigenpair residuals, detects degeneracies, and compares against the
    uncoupled limit.
    """

    def __init__(self, config: SolverConfig = SolverConfig()):
        self.config = config

    def _select_method(self, n_dim: int) -> SolverMethod:
        if self.config.method != SolverMethod.AUTO:
            return self.config.method
        if self.config.num_eigenvalues >= n_dim - 1:
            return SolverMethod.DENSE
        if n_dim > self.config.sparse_threshold:
            return SolverMethod.SPARSE
        return SolverMethod.DENSE

    @staticmethod
    def _reconstruct_multi_states(
        coupled_ham: CoupledHamiltonian,
        energies: np.ndarray,
        raw_vectors: np.ndarray,
        interior_only: bool,
        channel_labels: Optional[Sequence[str]] = None
    ) -> List[MultiComponentWavefunction]:
        """
        Reshapes stacked eigenvectors of length N * M_act into (N, M) multi-component
        wavefunctions, pads Dirichlet boundaries, standardizes global complex phase across
        all channels simultaneously, and normalizes the complete multi-component state.
        """
        grid = coupled_ham.grid
        n_channels = coupled_ham.num_channels
        m_total = grid.points
        m_act = m_total - 2 if interior_only else m_total
        num_states = len(energies)

        if raw_vectors.shape[0] != n_channels * m_act:
            raise ValueError(
                f"Eigenvector length {raw_vectors.shape[0]} does not match "
                f"num_channels ({n_channels}) * active_points ({m_act})."
            )

        states: List[MultiComponentWavefunction] = []
        for idx in range(num_states):
            vec = raw_vectors[:, idx].reshape(n_channels, m_act)
            if interior_only and grid.boundary_condition == BoundaryCondition.DIRICHLET:
                full_comp = np.zeros((n_channels, m_total), dtype=np.complex128)
                full_comp[:, 1:-1] = vec
            else:
                full_comp = vec.astype(np.complex128, copy=True)

            # Standardize global complex phase using peak amplitude across ALL channels
            flat_idx = int(np.argmax(np.abs(full_comp)))
            peak_val = full_comp.ravel()[flat_idx]
            if abs(peak_val) > 0.0:
                phase = float(np.angle(peak_val))
                full_comp *= np.exp(-1j * phase)

            if np.max(np.abs(np.imag(full_comp))) < 1e-12:
                full_comp = np.real(full_comp).astype(np.complex128)

            # Complete multi-component normalization: sum_c integral |psi_c|^2 dx = 1
            total_dens = np.sum(np.abs(full_comp) ** 2, axis=0)
            norm_sq = float(integrate.simpson(y=total_dens, x=grid.grid))
            if norm_sq <= 0.0:
                raise ValueError(f"Reconstructed coupled eigenstate {idx} has zero norm.")
            full_comp /= np.sqrt(norm_sq)

            states.append(
                MultiComponentWavefunction(
                    x=grid.grid,
                    components=full_comp,
                    energy=float(energies[idx]),
                    channel_labels=channel_labels,
                    normalize=False
                )
            )

        return states

    def solve(
        self,
        coupled_hamiltonian: CoupledHamiltonian,
        accuracy_order: int = 2,
        interior_only: bool = True,
        include_uncoupled_comparison: bool = True,
        channel_labels: Optional[Sequence[str]] = None
    ) -> CoupledSolverResult:
        """
        Solves the coupled eigenvalue problem and returns eigenvalues, multi-component
        eigenstates, residuals, degeneracies, and optional uncoupled comparison.
        """
        h_sparse = coupled_hamiltonian.build_sparse(
            accuracy_order=accuracy_order,
            interior_only=interior_only
        )
        n_dim = h_sparse.shape[0]
        method = self._select_method(n_dim)

        if method == SolverMethod.DENSE:
            energies, vectors = DenseSolver.solve(h_sparse.toarray(), self.config.num_eigenvalues)
        else:
            energies, vectors = SparseSolver.solve(h_sparse, self.config)

        val_tol = max(self.config.tolerance * 100.0, 1e-5)
        residuals = EigenpairValidator.validate_all_pairs(
            h_sparse, energies, vectors, tolerance=val_tol
        )
        max_res = float(np.max(residuals)) if residuals else 0.0

        degeneracies = DegeneracyAnalyzer.find_degeneracies(
            energies, tolerance=self.config.degeneracy_tolerance
        )

        eigenstates = self._reconstruct_multi_states(
            coupled_hamiltonian=coupled_hamiltonian,
            energies=energies,
            raw_vectors=vectors,
            interior_only=interior_only,
            channel_labels=channel_labels
        )

        uncoupled_comp: Optional[SpectrumComparisonWithUncoupled] = None
        if include_uncoupled_comparison:
            uncoupled_ham = coupled_hamiltonian.uncoupled_hamiltonian()
            h_unc = uncoupled_ham.build_sparse(
                accuracy_order=accuracy_order,
                interior_only=interior_only
            )
            if method == SolverMethod.DENSE:
                e_unc, _ = DenseSolver.solve(h_unc.toarray(), self.config.num_eigenvalues)
            else:
                e_unc, _ = SparseSolver.solve(h_unc, self.config)

            shifts = energies - e_unc
            uncoupled_comp = SpectrumComparisonWithUncoupled(
                coupled_energies=energies,
                uncoupled_energies=e_unc,
                energy_shifts=shifts,
                max_energy_shift=float(np.max(np.abs(shifts))),
                rms_energy_shift=float(np.sqrt(np.mean(shifts ** 2)))
            )

        return CoupledSolverResult(
            energies=energies,
            eigenstates=eigenstates,
            residuals=residuals,
            max_residual=max_res,
            degeneracies=degeneracies,
            method_used=method.value,
            hamiltonian_dimension=n_dim,
            num_channels=coupled_hamiltonian.num_channels,
            converged=(max_res <= val_tol),
            uncoupled_comparison=uncoupled_comp
        )
