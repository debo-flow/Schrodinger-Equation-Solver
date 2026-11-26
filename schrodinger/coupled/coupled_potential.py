import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential


CouplingTerm = Union[int, float, complex, Potential, Callable[[np.ndarray], np.ndarray]]
TimeDependentCouplingTerm = Union[
    CouplingTerm,
    TimeDependentPotential,
    Callable[[np.ndarray, float], np.ndarray]
]


class _ConjugatePotentialTerm:
    """Internal wrapper representing the complex conjugate of a coupling term."""

    def __init__(self, term: CouplingTerm):
        self.term = term

    def evaluate(self, x_arr: np.ndarray) -> np.ndarray:
        return np.conjugate(CouplingMatrix.evaluate_term(self.term, x_arr))


class CouplingMatrix:
    """
    Represents and validates an N x N Hermitian matrix-valued potential V(x).
    Supports constant, Potential-based, and position-dependent real or complex terms.
    """

    def __init__(
        self,
        num_channels: int,
        matrix: Optional[Sequence[Sequence[Any]]] = None,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        hermitian_tolerance: float = 1e-10
    ):
        if not isinstance(num_channels, (int, np.integer)) or num_channels < 1:
            raise ValueError(f"num_channels must be an integer >= 1. Received: {num_channels}")
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")
        if hermitian_tolerance <= 0:
            raise ValueError("hermitian_tolerance must be strictly positive.")

        self.num_channels = int(num_channels)
        self.unit_system = unit_system
        self.hermitian_tolerance = float(hermitian_tolerance)
        self._terms: List[List[Any]] = [
            [0.0 for _ in range(self.num_channels)]
            for _ in range(self.num_channels)
        ]

        if matrix is not None:
            self._load_matrix(matrix)

    def _validate_channel_index(self, idx: int) -> None:
        if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= self.num_channels:
            raise IndexError(
                f"Channel index {idx} out of bounds for {self.num_channels}-channel system."
            )

    def _validate_term_compatibility(self, term: Any, is_diagonal: bool) -> None:
        if isinstance(term, _ConjugatePotentialTerm):
            self._validate_term_compatibility(term.term, is_diagonal=is_diagonal)
            return

        if isinstance(term, Potential):
            if term.unit_system != self.unit_system:
                raise ValueError(
                    f"Unit mismatch: term has '{term.unit_system}', expected '{self.unit_system}'."
                )
            return

        if isinstance(term, (int, float, complex, np.number)) and not isinstance(term, bool):
            if not np.isfinite(term):
                raise ValueError(f"Scalar potential/coupling term must be finite. Received: {term}")
            if is_diagonal and abs(np.imag(term)) > self.hermitian_tolerance:
                raise ValueError(f"Diagonal channel potential must be strictly real. Received: {term}")
            return

        if callable(term):
            return

        raise TypeError(f"Unsupported potential/coupling term type: {type(term)}")

    def _load_matrix(self, matrix: Sequence[Sequence[Any]]) -> None:
        if len(matrix) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} rows in coupling matrix, received {len(matrix)}."
            )
        for i, row in enumerate(matrix):
            if len(row) != self.num_channels:
                raise ValueError(
                    f"Row {i} has {len(row)} columns, expected {self.num_channels}."
                )
            for j, term in enumerate(row):
                self._validate_term_compatibility(term, is_diagonal=(i == j))
                self._terms[i][j] = term

    def set_diagonal(self, channel: int, potential: CouplingTerm) -> None:
        """Sets the real diagonal potential V_{channel, channel}(x)."""
        self._validate_channel_index(channel)
        self._validate_term_compatibility(potential, is_diagonal=True)
        self._terms[channel][channel] = potential

    def set_coupling(
        self,
        i: int,
        j: int,
        coupling: CouplingTerm,
        auto_conjugate: bool = True
    ) -> None:
        """
        Sets off-diagonal coupling V_{i, j}(x). If `auto_conjugate=True` and i != j,
        automatically sets V_{j, i}(x) = V_{i, j}^*(x) to preserve Hermiticity.
        """
        self._validate_channel_index(i)
        self._validate_channel_index(j)
        if i == j:
            self.set_diagonal(i, coupling)
            return

        self._validate_term_compatibility(coupling, is_diagonal=False)
        self._terms[i][j] = coupling
        if auto_conjugate:
            if isinstance(coupling, (int, float, complex, np.number)):
                self._terms[j][i] = complex(np.conjugate(coupling))
            else:
                self._terms[j][i] = _ConjugatePotentialTerm(coupling)

    def get_term(self, i: int, j: int) -> Any:
        self._validate_channel_index(i)
        self._validate_channel_index(j)
        return self._terms[i][j]

    @staticmethod
    def evaluate_term(term: Any, x_arr: np.ndarray) -> np.ndarray:
        """Evaluates a single scalar, Potential, or callable term over 1D array `x_arr`."""
        if isinstance(term, _ConjugatePotentialTerm):
            return term.evaluate(x_arr)
        if isinstance(term, Potential):
            vals = np.asarray(term(x_arr))
        elif isinstance(term, (int, float, complex, np.number)) and not isinstance(term, bool):
            vals = np.full(x_arr.shape, term)
        elif callable(term):
            vals = np.asarray(term(x_arr))
            if vals.ndim == 0:
                vals = np.full(x_arr.shape, vals.item())
            elif vals.shape != x_arr.shape:
                raise ValueError(
                    f"Callable term returned shape {vals.shape}, expected {x_arr.shape}."
                )
        else:
            raise TypeError(f"Unsupported term type: {type(term)}")

        if np.any(np.isnan(vals)):
            raise ArithmeticError("Coupling matrix evaluation produced NaN values.")
        return vals

    def evaluate(
        self,
        x: Union[float, np.ndarray],
        verify_hermitian: bool = True
    ) -> np.ndarray:
        """
        Evaluates V(x) across scalar or 1D array coordinates.
        Returns shape (N, N) for scalar x, or (N, N, M) for 1D array x of length M.
        """
        is_scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        x_arr = np.atleast_1d(x).astype(np.float64)
        if x_arr.ndim != 1:
            raise ValueError("Spatial coordinates x must be scalar or 1D array.")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Spatial coordinates x must be finite.")

        n = self.num_channels
        m = len(x_arr)

        evaluated_blocks: List[List[np.ndarray]] = []
        has_complex = False
        for i in range(n):
            row_vals: List[np.ndarray] = []
            for j in range(n):
                arr = self.evaluate_term(self._terms[i][j], x_arr)
                if np.iscomplexobj(arr) and np.any(np.imag(arr) != 0.0):
                    has_complex = True
                row_vals.append(arr)
            evaluated_blocks.append(row_vals)

        out_dtype = np.complex128 if has_complex else np.float64
        v_mat = np.zeros((n, n, m), dtype=out_dtype)
        for i in range(n):
            for j in range(n):
                v_mat[i, j, :] = evaluated_blocks[i][j].astype(out_dtype)

        if verify_hermitian:
            self.validate_hermiticity(v_mat, tolerance=self.hermitian_tolerance)

        return v_mat[:, :, 0] if is_scalar else v_mat

    @staticmethod
    def validate_hermiticity(v_mat: np.ndarray, tolerance: float = 1e-10) -> None:
        """
        Verifies V(x) = V^dagger(x) for every spatial point x.
        Raises ValueError if any off-diagonal pair or diagonal entry violates Hermiticity.
        """
        if v_mat.ndim == 2:
            diff = v_mat - np.conjugate(v_mat.T)
        elif v_mat.ndim == 3:
            diff = v_mat - np.conjugate(np.swapaxes(v_mat, 0, 1))
        else:
            raise ValueError(f"Expected 2D or 3D potential array, received {v_mat.ndim}D.")

        # Ignore infinite diagonal walls when checking Hermitian symmetry
        finite_mask = np.isfinite(diff)
        if np.any(finite_mask):
            max_dev = float(np.max(np.abs(diff[finite_mask])))
            if max_dev > tolerance:
                raise ValueError(
                    f"Coupled potential matrix is not Hermitian: max |V - V^dagger| = {max_dev:.2e} > {tolerance:.2e}."
                )


class CoupledPotential:
    """
    High-level N-channel matrix-valued quantum potential V(x).
    Supports direct evaluation, uncoupled-limit extraction, and algebraic composition.
    """

    def __init__(
        self,
        coupling_matrix: CouplingMatrix,
        name: str = "CoupledPotential"
    ):
        if not isinstance(coupling_matrix, CouplingMatrix):
            raise TypeError(f"Expected CouplingMatrix instance. Received: {type(coupling_matrix)}")
        self.coupling_matrix = coupling_matrix
        self.num_channels = coupling_matrix.num_channels
        self.unit_system = coupling_matrix.unit_system
        self.name = name

    @classmethod
    def from_channels(
        cls,
        channel_potentials: Sequence[CouplingTerm],
        couplings: Optional[Dict[Tuple[int, int], CouplingTerm]] = None,
        unit_system: Optional[Literal["dimensionless", "si"]] = None,
        hermitian_tolerance: float = 1e-10,
        name: str = "CoupledPotential"
    ) -> "CoupledPotential":
        """
        Builds an N-channel CoupledPotential from diagonal `channel_potentials`
        and a dictionary of off-diagonal `couplings={(i, j): term}`.
        """
        if not channel_potentials:
            raise ValueError("channel_potentials must contain at least one channel.")

        inferred_unit: Literal["dimensionless", "si"] = "dimensionless"
        if unit_system is not None:
            inferred_unit = unit_system
        else:
            for pot in channel_potentials:
                if isinstance(pot, Potential):
                    inferred_unit = pot.unit_system
                    break

        cm = CouplingMatrix(
            num_channels=len(channel_potentials),
            unit_system=inferred_unit,
            hermitian_tolerance=hermitian_tolerance
        )
        for idx, pot in enumerate(channel_potentials):
            cm.set_diagonal(idx, pot)

        if couplings is not None:
            # If user explicitly supplied both (i, j) and (j, i), validate both without overwriting
            explicit_pairs = set(couplings.keys())
            for (i, j), term in couplings.items():
                has_transpose = (j, i) in explicit_pairs and i != j
                cm.set_coupling(i, j, term, auto_conjugate=(not has_transpose))

        return cls(coupling_matrix=cm, name=name)

    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluates V(x) returning shape (N, N) for scalar x or (N, N, M) for 1D array x."""
        return self.coupling_matrix.evaluate(x, verify_hermitian=True)

    def uncoupled_copy(self) -> "CoupledPotential":
        """Returns a new CoupledPotential with identical diagonal potentials and zero coupling."""
        cm = CouplingMatrix(
            num_channels=self.num_channels,
            unit_system=self.unit_system,
            hermitian_tolerance=self.coupling_matrix.hermitian_tolerance
        )
        for i in range(self.num_channels):
            cm.set_diagonal(i, self.coupling_matrix.get_term(i, i))
        return CoupledPotential(coupling_matrix=cm, name=f"{self.name}_uncoupled")

    def is_complex(self, x_probe: np.ndarray) -> bool:
        """Checks whether V(x) contains non-zero imaginary off-diagonal couplings on `x_probe`."""
        v_mat = self(x_probe)
        return bool(np.iscomplexobj(v_mat) and np.any(np.imag(v_mat) != 0.0))

    def has_nonzero_coupling(self, x_probe: np.ndarray, tolerance: float = 1e-14) -> bool:
        """Returns True if any off-diagonal element V_{i,j}(x) (i != j) is non-zero."""
        if self.num_channels < 2:
            return False
        v_mat = self(x_probe)
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                if i != j and np.any(np.abs(v_mat[i, j]) > tolerance):
                    return True
        return False

    def coupling_summary(self, x_probe: np.ndarray) -> Dict[str, Any]:
        """Summarizes channel count, complex status, and maximum coupling magnitudes."""
        v_mat = self(x_probe)
        max_off_diag = 0.0
        coupled_pairs: List[List[int]] = []
        for i in range(self.num_channels):
            for j in range(i + 1, self.num_channels):
                mag = float(np.max(np.abs(v_mat[i, j])))
                max_off_diag = max(max_off_diag, mag)
                if mag > 1e-14:
                    coupled_pairs.append([i, j])
        return {
            "num_channels": self.num_channels,
            "unit_system": self.unit_system,
            "is_complex": self.is_complex(x_probe),
            "has_nonzero_coupling": max_off_diag > 1e-14,
            "max_off_diagonal_magnitude": max_off_diag,
            "coupled_channel_pairs": coupled_pairs
        }

    def __add__(self, other: "CoupledPotential") -> "CoupledPotentialSum":
        if not isinstance(other, CoupledPotential):
            return NotImplemented
        return CoupledPotentialSum([self, other])

    def __sub__(self, other: "CoupledPotential") -> "CoupledPotentialSum":
        if not isinstance(other, CoupledPotential):
            return NotImplemented
        return CoupledPotentialSum([self, ScaledCoupledPotential(other, -1.0)])

    def __mul__(self, scalar: Union[int, float]) -> "ScaledCoupledPotential":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, np.number)):
            return NotImplemented
        return ScaledCoupledPotential(self, float(scalar))

    def __rmul__(self, scalar: Union[int, float]) -> "ScaledCoupledPotential":
        return self.__mul__(scalar)

    def __neg__(self) -> "ScaledCoupledPotential":
        return ScaledCoupledPotential(self, -1.0)


class CoupledPotentialSum(CoupledPotential):
    """Algebraic sum of compatible N-channel CoupledPotential instances."""

    def __init__(self, potentials: Sequence[CoupledPotential]):
        if not potentials:
            raise ValueError("CoupledPotentialSum requires at least one CoupledPotential.")
        base_n = potentials[0].num_channels
        base_unit = potentials[0].unit_system
        for p in potentials:
            if not isinstance(p, CoupledPotential):
                raise TypeError("All operands must be CoupledPotential instances.")
            if p.num_channels != base_n:
                raise ValueError(
                    f"Channel count mismatch in CoupledPotentialSum: {p.num_channels} != {base_n}."
                )
            if p.unit_system != base_unit:
                raise ValueError(
                    f"Unit mismatch in CoupledPotentialSum: '{p.unit_system}' != '{base_unit}'."
                )

        super().__init__(
            coupling_matrix=CouplingMatrix(num_channels=base_n, unit_system=base_unit),
            name="CoupledPotentialSum"
        )
        self.potentials: List[CoupledPotential] = list(potentials)

    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        total = self.potentials[0](x).copy()
        for p in self.potentials[1:]:
            total = total + p(x)
        CouplingMatrix.validate_hermiticity(total, tolerance=self.coupling_matrix.hermitian_tolerance)
        return total


class ScaledCoupledPotential(CoupledPotential):
    """Real scalar multiple of a CoupledPotential (preserves Hermiticity)."""

    def __init__(self, potential: CoupledPotential, scale: float):
        if not isinstance(potential, CoupledPotential):
            raise TypeError("Expected a CoupledPotential instance.")
        if isinstance(scale, bool) or not math.isfinite(scale):
            raise ValueError("Real finite scale factor required to preserve Hermiticity.")

        super().__init__(
            coupling_matrix=CouplingMatrix(
                num_channels=potential.num_channels,
                unit_system=potential.unit_system
            ),
            name=f"Scaled({potential.name})"
        )
        self.potential = potential
        self.scale = float(scale)

    def __call__(self, x: Union[float, np.ndarray]) -> np.ndarray:
        return self.scale * self.potential(x)


class TimeDependentCoupledPotential:
    """
    Represents a time- and position-dependent N x N Hermitian potential matrix V(x, t).
    Supports static CoupledPotential wrapping as well as time-dependent diagonal or
    off-diagonal coupling terms f(x, t) or TimeDependentPotential instances.
    """

    def __init__(
        self,
        num_channels: int,
        unit_system: Literal["dimensionless", "si"] = "dimensionless",
        hermitian_tolerance: float = 1e-10
    ):
        if not isinstance(num_channels, (int, np.integer)) or num_channels < 1:
            raise ValueError("num_channels must be >= 1.")
        if unit_system not in ("dimensionless", "si"):
            raise ValueError("unit_system must be 'dimensionless' or 'si'.")
        self.num_channels = int(num_channels)
        self.unit_system = unit_system
        self.hermitian_tolerance = float(hermitian_tolerance)
        self._terms: List[List[Tuple[Any, bool]]] = [
            [(0.0, False) for _ in range(self.num_channels)]
            for _ in range(self.num_channels)
        ]

    @classmethod
    def from_static(cls, coupled_potential: CoupledPotential) -> "TimeDependentCoupledPotential":
        """Wraps a static CoupledPotential into the time-dependent interface."""
        td = cls(
            num_channels=coupled_potential.num_channels,
            unit_system=coupled_potential.unit_system,
            hermitian_tolerance=coupled_potential.coupling_matrix.hermitian_tolerance
        )
        for i in range(coupled_potential.num_channels):
            for j in range(coupled_potential.num_channels):
                td._terms[i][j] = (coupled_potential.coupling_matrix.get_term(i, j), False)
        return td

    def set_term(
        self,
        i: int,
        j: int,
        term: TimeDependentCouplingTerm,
        auto_conjugate: bool = True
    ) -> None:
        """Sets entry (i, j) to a static or time-dependent term."""
        if not (0 <= i < self.num_channels and 0 <= j < self.num_channels):
            raise IndexError("Channel index out of range.")
        if isinstance(term, (Potential, TimeDependentPotential)) and term.unit_system != self.unit_system:
            raise ValueError(f"Unit mismatch: '{term.unit_system}' != '{self.unit_system}'.")

        self._terms[i][j] = (term, False)
        if auto_conjugate and i != j:
            self._terms[j][i] = (term, True)

    def _eval_single(self, term: Any, conjugate: bool, x_arr: np.ndarray, t: float) -> np.ndarray:
        if isinstance(term, TimeDependentPotential):
            vals = np.asarray(term(x_arr, t))
        elif isinstance(term, (Potential, _ConjugatePotentialTerm, int, float, complex, np.number)):
            vals = CouplingMatrix.evaluate_term(term, x_arr)
        elif callable(term):
            try:
                vals = np.asarray(term(x_arr, t))
            except TypeError:
                vals = np.asarray(term(x_arr))
            if vals.ndim == 0:
                vals = np.full(x_arr.shape, vals.item())
        else:
            raise TypeError(f"Unsupported time-dependent coupling term: {type(term)}")

        return np.conjugate(vals) if conjugate else vals

    def evaluate(self, x: Union[float, np.ndarray], t: float) -> np.ndarray:
        """Evaluates V(x, t) returning (N, N) for scalar x or (N, N, M) for 1D array x."""
        if isinstance(t, bool) or not math.isfinite(t):
            raise ValueError(f"Time t must be finite. Received: {t}")
        is_scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        x_arr = np.atleast_1d(x).astype(np.float64)

        n = self.num_channels
        m = len(x_arr)
        blocks: List[List[np.ndarray]] = []
        has_complex = False
        for i in range(n):
            row: List[np.ndarray] = []
            for j in range(n):
                term, conj = self._terms[i][j]
                arr = self._eval_single(term, conj, x_arr, float(t))
                if np.iscomplexobj(arr) and np.any(np.imag(arr) != 0.0):
                    has_complex = True
                row.append(arr)
            blocks.append(row)

        out_dtype = np.complex128 if has_complex else np.float64
        v_mat = np.zeros((n, n, m), dtype=out_dtype)
        for i in range(n):
            for j in range(n):
                v_mat[i, j, :] = blocks[i][j].astype(out_dtype)

        CouplingMatrix.validate_hermiticity(v_mat, tolerance=self.hermitian_tolerance)
        return v_mat[:, :, 0] if is_scalar else v_mat

    def at_time(self, t: float) -> CoupledPotential:
        """Freezes V(x, t) at instant t into a static CoupledPotential snapshot."""
        if isinstance(t, bool) or not math.isfinite(t):
            raise ValueError(f"Time t must be finite. Received: {t}")
        t_fixed = float(t)
        cm = CouplingMatrix(
            num_channels=self.num_channels,
            unit_system=self.unit_system,
            hermitian_tolerance=self.hermitian_tolerance
        )
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                term, conj = self._terms[i][j]
                cm.set_coupling(
                    i,
                    j,
                    lambda x_in, _t=term, _c=conj, _tf=t_fixed: self._eval_single(_t, _c, x_in, _tf),
                    auto_conjugate=False
                )
        return CoupledPotential(coupling_matrix=cm, name=f"CoupledPotential(t={t_fixed})")
