from typing import Callable, Literal, Optional, Sequence, Union
import numpy as np
from scipy import integrate, sparse
from schrodinger.potentials.base import Potential
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import IntegrationMethod
from schrodinger.coupled.coupled_potential import CoupledPotential
from schrodinger.coupled.multi_wavefunction import MultiComponentWavefunction
from schrodinger.analysis.hermiticity import HermiticityAnalyzer


QuantumState = Union[Wavefunction, MultiComponentWavefunction]
PerturbationSource = Union[
    Potential,
    CoupledPotential,
    Callable[[np.ndarray], np.ndarray],
    np.ndarray,
    sparse.spmatrix
]


class PerturbationOperator:
    """
    Represents a Hermitian perturbation Hamiltonian H' for either single-channel
    or coupled-channel quantum systems. Supports:
      - Potential subclasses (V'(x))
      - CoupledPotential matrix-valued perturbations (V'_{ij}(x))
      - Position-dependent callables f(x)
      - Dense or sparse Hermitian operator matrices
    """

    def __init__(
        self,
        operator: PerturbationSource,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        name: str = "PerturbationOperator",
        hermitian_tolerance: float = 1e-10
    ):
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")
        if hermitian_tolerance <= 0:
            raise ValueError("hermitian_tolerance must be strictly positive.")

        if isinstance(operator, (Potential, CoupledPotential)):
            self.unit_system = operator.unit_system
        elif isinstance(operator, (np.ndarray, sparse.spmatrix)):
            if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
                raise ValueError(f"Operator matrix must be square 2D. Received shape: {operator.shape}")
            is_herm, dev = HermiticityAnalyzer(tolerance=hermitian_tolerance).is_hermitian(operator)
            if not is_herm:
                raise ValueError(
                    f"Perturbation matrix is not Hermitian: max |H' - H'^dagger| = {dev:.2e}."
                )
            self.unit_system = unit_system
        elif callable(operator):
            self.unit_system = unit_system
        else:
            raise TypeError(f"Unsupported perturbation operator type: {type(operator)}")

        self.operator = operator
        self.name = name
        self.hermitian_tolerance = float(hermitian_tolerance)

    @property
    def is_coupled(self) -> bool:
        return isinstance(self.operator, CoupledPotential)

    def _apply_matrix_to_single(self, mat: Union[np.ndarray, sparse.spmatrix], wf: Wavefunction) -> np.ndarray:
        n_pts = len(wf.x)
        dim = mat.shape[0]
        if dim == n_pts:
            return np.asarray(mat.dot(wf.psi) if sparse.issparse(mat) else mat @ wf.psi, dtype=np.complex128)
        if dim == n_pts - 2:
            out = np.zeros(n_pts, dtype=np.complex128)
            inner = mat.dot(wf.psi[1:-1]) if sparse.issparse(mat) else mat @ wf.psi[1:-1]
            out[1:-1] = inner
            return out
        raise ValueError(
            f"Operator matrix dimension ({dim}) incompatible with wavefunction length ({n_pts})."
        )

    def _apply_matrix_to_coupled(
        self,
        mat: Union[np.ndarray, sparse.spmatrix],
        state: MultiComponentWavefunction
    ) -> np.ndarray:
        n_ch = state.num_channels
        n_pts = state.num_points
        dim = mat.shape[0]

        if dim == n_ch * n_pts:
            vec = state.components.reshape(-1)
            res = mat.dot(vec) if sparse.issparse(mat) else mat @ vec
            return np.asarray(res, dtype=np.complex128).reshape(n_ch, n_pts)
        if dim == n_ch * (n_pts - 2):
            vec = state.components[:, 1:-1].reshape(-1)
            res = mat.dot(vec) if sparse.issparse(mat) else mat @ vec
            out = np.zeros((n_ch, n_pts), dtype=np.complex128)
            out[:, 1:-1] = np.asarray(res, dtype=np.complex128).reshape(n_ch, n_pts - 2)
            return out
        raise ValueError(
            f"Operator matrix dimension ({dim}) incompatible with {n_ch} channels x {n_pts} points."
        )

    def apply(self, state: QuantumState) -> np.ndarray:
        """
        Applies H' to `state`:
          - Returns 1D complex128 array for Wavefunction
          - Returns 2D complex128 array (num_channels, num_points) for MultiComponentWavefunction
        """
        if isinstance(state, Wavefunction):
            if isinstance(self.operator, Potential):
                v_vals = np.asarray(self.operator(state.x), dtype=np.float64)
                if not np.all(np.isfinite(v_vals)):
                    # Handle infinite boundaries cleanly where psi == 0
                    finite_v = np.where(np.isfinite(v_vals), v_vals, 0.0)
                    if np.any((~np.isfinite(v_vals)) & (np.abs(state.psi) > 1e-12)):
                        raise ValueError("Infinite perturbation encountered where wavefunction is non-zero.")
                    v_vals = finite_v
                return (v_vals * state.psi).astype(np.complex128)

            if isinstance(self.operator, (np.ndarray, sparse.spmatrix)):
                return self._apply_matrix_to_single(self.operator, state)

            if callable(self.operator) and not isinstance(self.operator, CoupledPotential):
                v_vals = np.asarray(self.operator(state.x))
                if v_vals.ndim == 0:
                    v_vals = np.full(state.x.shape, v_vals.item())
                if v_vals.shape != state.x.shape:
                    raise ValueError(f"Perturbation callable returned shape {v_vals.shape}, expected {state.x.shape}.")
                if not np.all(np.isfinite(v_vals)):
                    raise ValueError("Perturbation callable returned non-finite values.")
                return (v_vals * state.psi).astype(np.complex128)

            raise TypeError(
                f"Cannot apply {type(self.operator).__name__} to single-channel Wavefunction."
            )

        if isinstance(state, MultiComponentWavefunction):
            if isinstance(self.operator, CoupledPotential):
                if self.operator.num_channels != state.num_channels:
                    raise ValueError(
                        f"Coupled perturbation channels ({self.operator.num_channels}) != "
                        f"state channels ({state.num_channels})."
                    )
                v_mat = self.operator(state.x)  # Shape (N, N, M)
                if not np.all(np.isfinite(v_mat)):
                    raise ValueError("Coupled perturbation contains non-finite values.")
                # (H' Psi)_a(x) = sum_b V_{ab}(x) * psi_b(x)
                return np.einsum("abm,bm->am", v_mat, state.components).astype(np.complex128)

            if isinstance(self.operator, (np.ndarray, sparse.spmatrix)):
                return self._apply_matrix_to_coupled(self.operator, state)

            if isinstance(self.operator, Potential):
                # Channel-independent scalar perturbation H' = V'(x) * I_N
                v_vals = np.asarray(self.operator(state.x), dtype=np.float64)
                return (v_vals[np.newaxis, :] * state.components).astype(np.complex128)

            raise TypeError(
                f"Cannot apply {type(self.operator).__name__} to MultiComponentWavefunction."
            )

        raise TypeError(f"Unsupported state type: {type(state)}")

    def matrix_element(
        self,
        bra: QuantumState,
        ket: QuantumState,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> complex:
        """
        Evaluates <bra | H' | ket> using continuous numerical integration.
        """
        if type(bra) is not type(ket):
            raise TypeError("bra and ket must have the same quantum state type.")
        if bra.x.shape != ket.x.shape or not np.allclose(bra.x, ket.x):
            raise ValueError("bra and ket must share the same spatial grid.")

        h_ket = self.apply(ket)
        if isinstance(bra, Wavefunction):
            integrand = np.conj(bra.psi) * h_ket
        else:
            if bra.num_channels != ket.num_channels:
                raise ValueError("bra and ket must have identical channel counts.")
            integrand = np.sum(np.conj(bra.components) * h_ket, axis=0)

        if integration_method == IntegrationMethod.SIMPSON:
            return complex(integrate.simpson(y=integrand, x=bra.x))
        return complex(integrate.trapezoid(y=integrand, x=bra.x))


class PerturbationMatrix:
    """
    Stores and validates the perturbation matrix W_{mn} = <m^(0) | H' | n^(0)>
    in an unperturbed eigenbasis. Supports construction from PerturbationOperator
    or direct wrapping of an explicit Hermitian matrix.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        hermitian_tolerance: float = 1e-8
    ):
        arr = np.asarray(matrix, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"PerturbationMatrix must be square 2D. Received shape: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("PerturbationMatrix contains non-finite values.")

        is_herm, dev = HermiticityAnalyzer(tolerance=hermitian_tolerance).is_hermitian(arr)
        if not is_herm:
            raise ValueError(
                f"PerturbationMatrix is not Hermitian: max |W - W^dagger| = {dev:.2e} > {hermitian_tolerance:.2e}."
            )

        # Symmetrize floating-point integration noise at machine level
        self._matrix = 0.5 * (arr + np.conj(arr.T))
        self._hermiticity_deviation = float(dev)

    @classmethod
    def from_eigenstates(
        cls,
        eigenstates: Sequence[QuantumState],
        perturbation: PerturbationOperator,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON,
        hermitian_tolerance: float = 1e-6
    ) -> "PerturbationMatrix":
        """Computes W_{mn} = <m^(0) | H' | n^(0)> across all pairs of unperturbed eigenstates."""
        if not eigenstates:
            raise ValueError("eigenstates must not be empty.")
        n = len(eigenstates)
        mat = np.zeros((n, n), dtype=np.complex128)

        for i in range(n):
            for j in range(i, n):
                val = perturbation.matrix_element(
                    eigenstates[i], eigenstates[j], integration_method=integration_method
                )
                mat[i, j] = val
                if i != j:
                    # Verify numerical Hermiticity explicitly against <j|H'|i>
                    val_ji = perturbation.matrix_element(
                        eigenstates[j], eigenstates[i], integration_method=integration_method
                    )
                    mat[j, i] = val_ji

        return cls(matrix=mat, hermitian_tolerance=hermitian_tolerance)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def hermiticity_deviation(self) -> float:
        return self._hermiticity_deviation

    @property
    def is_complex(self) -> bool:
        return bool(np.any(np.abs(np.imag(self._matrix)) > 1e-12))

    def element(self, m: int, n: int) -> complex:
        return complex(self._matrix[m, n])

    def diagonal(self) -> np.ndarray:
        """Returns real diagonal expectation values <n^(0) | H' | n^(0)>."""
        return np.real(np.diag(self._matrix)).astype(np.float64)

    def submatrix(self, indices: Sequence[int]) -> np.ndarray:
        """Extracts Hermitian submatrix for a degenerate subspace of state indices."""
        idx = list(indices)
        return self._matrix[np.ix_(idx, idx)].copy()
