from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence, Union
import numpy as np
from scipy import linalg

from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.potentials.base import Potential
from schrodinger.operators.kinetic_operator import KineticEnergyOperator
from schrodinger.operators.potential_operator import PotentialEnergyOperator
from schrodinger.operators.hamiltonian import Hamiltonian
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.wavefunctions.phase_manager import PhaseManager


BasisInput = Union[Callable[[np.ndarray], np.ndarray], np.ndarray, Wavefunction]


@dataclass
class RayleighRitzResult:
    """Output of a Rayleigh-Ritz basis-expansion generalized eigenvalue solve (H c = E S c)."""
    basis_size: int
    energies: np.ndarray
    wavefunctions: List[Wavefunction]
    coefficients: np.ndarray
    hamiltonian_matrix: np.ndarray
    overlap_matrix: np.ndarray
    is_orthogonal_basis: bool
    min_overlap_eigenvalue: float
    overlap_condition_number: float
    generalized_residuals: List[float]
    max_generalized_residual: float


@dataclass
class RayleighRitzConvergenceResult:
    """Convergence profile of Rayleigh-Ritz energies across increasing basis dimensions."""
    basis_sizes: List[int]
    ground_energies: List[float]
    lowest_k_energies: List[np.ndarray]
    is_monotonically_nonincreasing: bool
    final_result: RayleighRitzResult


class RayleighRitzSolver:
    """
    Implements the Rayleigh-Ritz linear basis-expansion method:
        |psi> = sum_i c_i |phi_i>
        H c = E S c
    where H_ij = <phi_i | H | phi_j> and S_ij = <phi_i | phi_j>.
    Supports both orthogonal and non-orthogonal basis sets with explicit
    linear-independence validation.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Potential,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        accuracy_order: int = 2,
        linear_independence_tolerance: float = 1e-10
    ):
        if linear_independence_tolerance <= 0:
            raise ValueError("linear_independence_tolerance must be strictly positive.")
        self.grid = grid
        self.potential = potential
        self.mass = float(mass)
        self.hbar = float(hbar)
        self.unit_system = unit_system
        self.accuracy_order = int(accuracy_order)
        self.linear_independence_tolerance = float(linear_independence_tolerance)
        self._interior_only = (grid.boundary_condition == BoundaryCondition.DIRICHLET)

        fd = FiniteDifferenceEngine(grid)
        kin_op = KineticEnergyOperator(fd, mass=self.mass, hbar=self.hbar, unit_system=self.unit_system)
        pot_op = PotentialEnergyOperator(grid, potential)
        self._h_sparse = Hamiltonian(kin_op, pot_op).build_sparse(
            accuracy_order=self.accuracy_order,
            interior_only=self._interior_only
        )

    def _evaluate_basis_matrix(self, basis_functions: Sequence[BasisInput]) -> tuple:
        """Evaluates all basis functions on full and active grid points."""
        if not basis_functions:
            raise ValueError("basis_functions must contain at least one basis function.")

        n_pts = self.grid.points
        k = len(basis_functions)
        full_cols = np.zeros((n_pts, k), dtype=np.complex128)

        for idx, bf in enumerate(basis_functions):
            if isinstance(bf, Wavefunction):
                if bf.x.shape != (n_pts,) or not np.allclose(bf.x, self.grid.grid):
                    raise ValueError(f"Basis Wavefunction {idx} grid does not match SpatialGrid.")
                vals = bf.psi
            elif callable(bf):
                vals = np.asarray(bf(self.grid.grid), dtype=np.complex128)
            else:
                vals = np.asarray(bf, dtype=np.complex128)

            if vals.shape != (n_pts,):
                raise ValueError(f"Basis function {idx} shape {vals.shape} != {(n_pts,)}.")
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"Basis function {idx} contains non-finite values.")
            full_cols[:, idx] = vals

        active_cols = full_cols[1:-1, :] if self._interior_only else full_cols
        return full_cols, active_cols

    def solve(
        self,
        basis_functions: Sequence[BasisInput],
        num_states: Optional[int] = None,
        orthogonality_tolerance: float = 1e-6
    ) -> RayleighRitzResult:
        """
        Builds H and S matrices, validates basis linear independence, solves H c = E S c,
        and reconstructs normalized approximate Wavefunctions.
        """
        full_cols, active_cols = self._evaluate_basis_matrix(basis_functions)
        k_basis = active_cols.shape[1]
        dx = self.grid.dx

        # Overlap matrix S_ij = dx * <phi_i | phi_j> (Hermitian positive-definite for independent basis)
        s_mat = dx * (np.conj(active_cols.T) @ active_cols)
        s_mat = 0.5 * (s_mat + np.conj(s_mat.T))

        # Validate basis linear independence via smallest eigenvalue of S
        s_eigs = linalg.eigvalsh(s_mat)
        min_s_eig = float(s_eigs[0])
        max_s_eig = float(s_eigs[-1])
        if min_s_eig <= self.linear_independence_tolerance:
            raise ValueError(
                f"Basis functions are linearly dependent or ill-conditioned: "
                f"min overlap eigenvalue = {min_s_eig:.2e} <= {self.linear_independence_tolerance:.2e}."
            )
        s_cond = float(max_s_eig / min_s_eig)

        # Check whether basis is orthogonal (off-diagonal normalized overlaps near zero)
        diag_norms = np.sqrt(np.real(np.diag(s_mat)))
        s_normed = s_mat / np.outer(diag_norms, diag_norms)
        off_diag_err = float(np.max(np.abs(s_normed - np.eye(k_basis)))) if k_basis > 1 else 0.0
        is_orthogonal = off_diag_err <= orthogonality_tolerance

        # Hamiltonian matrix H_ij = dx * <phi_i | H | phi_j>
        h_active = self._h_sparse.dot(active_cols)
        h_mat = dx * (np.conj(active_cols.T) @ h_active)
        h_mat = 0.5 * (h_mat + np.conj(h_mat.T))

        # Solve generalized Hermitian eigenvalue problem H c = E S c
        energies, coeffs = linalg.eigh(h_mat, s_mat)
        energies = np.asarray(energies, dtype=np.float64)

        limit = k_basis if num_states is None else min(int(num_states), k_basis)
        if limit <= 0:
            raise ValueError("num_states must be positive.")

        energies = energies[:limit]
        coeffs = coeffs[:, :limit]

        # Generalized eigenpair residuals ||H c_m - E_m S c_m||_2
        residuals: List[float] = []
        for m in range(limit):
            c_m = coeffs[:, m]
            res_vec = h_mat @ c_m - energies[m] * (s_mat @ c_m)
            residuals.append(float(np.linalg.norm(res_vec)))

        # Reconstruct normalized approximate Wavefunctions
        wavefunctions: List[Wavefunction] = []
        for m in range(limit):
            psi_raw = full_cols @ coeffs[:, m]
            if self._interior_only:
                psi_raw = psi_raw.copy()
                psi_raw[0] = 0.0 + 0.0j
                psi_raw[-1] = 0.0 + 0.0j
            psi_norm, _ = NormalizationEngine.normalize(
                psi_raw, self.grid.grid, method=IntegrationMethod.SIMPSON
            )
            psi_std = PhaseManager.standardize_phase(psi_norm)
            wavefunctions.append(Wavefunction(x=self.grid.grid, psi=psi_std, energy=float(energies[m])))

        return RayleighRitzResult(
            basis_size=k_basis,
            energies=energies,
            wavefunctions=wavefunctions,
            coefficients=coeffs,
            hamiltonian_matrix=h_mat,
            overlap_matrix=s_mat,
            is_orthogonal_basis=is_orthogonal,
            min_overlap_eigenvalue=min_s_eig,
            overlap_condition_number=s_cond,
            generalized_residuals=residuals,
            max_generalized_residual=float(max(residuals)) if residuals else 0.0
        )

    def convergence_study(
        self,
        basis_functions: Sequence[BasisInput],
        basis_sizes: Optional[Sequence[int]] = None,
        track_states: int = 1
    ) -> RayleighRitzConvergenceResult:
        """
        Evaluates Rayleigh-Ritz energy convergence as basis size increases,
        checking the Hylleraas-Undheim-MacDonald variational monotonicity property.
        """
        max_k = len(basis_functions)
        if basis_sizes is None:
            sizes = list(range(1, max_k + 1))
        else:
            sizes = sorted(set(int(s) for s in basis_sizes))
            if not sizes or sizes[0] < 1 or sizes[-1] > max_k:
                raise ValueError(f"basis_sizes must lie within [1, {max_k}].")

        ground_energies: List[float] = []
        lowest_k: List[np.ndarray] = []
        last_res: Optional[RayleighRitzResult] = None

        for sz in sizes:
            res = self.solve(basis_functions[:sz], num_states=min(track_states, sz))
            ground_energies.append(float(res.energies[0]))
            lowest_k.append(res.energies.copy())
            last_res = res

        monotonic = all(
            ground_energies[i] <= ground_energies[i - 1] + 1e-10
            for i in range(1, len(ground_energies))
        )

        assert last_res is not None
        return RayleighRitzConvergenceResult(
            basis_sizes=sizes,
            ground_energies=ground_energies,
            lowest_k_energies=lowest_k,
            is_monotonically_nonincreasing=monotonic,
            final_result=last_res
        )
