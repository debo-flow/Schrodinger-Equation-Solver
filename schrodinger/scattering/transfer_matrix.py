import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
from schrodinger.potentials.base import Potential
from schrodinger.potentials.barrier import BarrierPotential
from schrodinger.potentials.step import StepPotential
from schrodinger.potentials.models import FiniteSquareWell, FreeParticle
from .transmission import TransmissionCalculator
from .reflection import ReflectionCalculator


@dataclass
class PotentialSlab:
    """Represents a constant-potential interval [x_start, x_end] with V(x) = potential."""
    x_start: float
    x_end: float
    potential: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x_start) and math.isfinite(self.x_end) and math.isfinite(self.potential)):
            raise ValueError("PotentialSlab coordinates and potential must be finite.")
        if self.x_end < self.x_start:
            raise ValueError(f"Invalid slab interval [{self.x_start}, {self.x_end}].")

    @property
    def width(self) -> float:
        return float(self.x_end - self.x_start)


@dataclass
class TransferMatrixStabilityReport:
    """Numerical stability and boundary-matching diagnostics for a transfer matrix."""
    determinant: complex
    expected_determinant: complex
    determinant_error: float
    max_evanescent_exponent: float
    condition_number: float
    boundary_matching_error: float
    is_stable: bool
    warnings: List[str]


@dataclass
class TransferMatrixResult:
    """Complete output of a TransferMatrixEngine evaluation."""
    matrix: np.ndarray
    r_amplitude: complex
    t_amplitude: complex
    reflection: float
    transmission: float
    probability_conservation_error: float
    slab_amplitudes: List[Tuple[complex, complex]]
    stability: TransferMatrixStabilityReport


class TransferMatrixEngine:
    """
    Modular 1D Transfer Matrix Method (TMM) engine for piecewise-constant potentials.
    Enforces continuity of psi and d(psi)/dx across every slab interface and validates
    determinant invariance, evanescent exponent bounds, and boundary matching.
    """

    MAX_EVANESCENT_ARG = 300.0

    def __init__(self, mass: float = 1.0, hbar: float = 1.0):
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        self.mass = float(mass)
        self.hbar = float(hbar)

    def wavevector(self, energy: float, potential: float) -> complex:
        """
        Computes complex wavevector k = sqrt(2m(E - V)) / hbar with smooth
        threshold regularization when |E - V| is near zero.
        """
        diff = energy - potential
        eps = 1e-14 * max(1.0, abs(energy), abs(potential))
        if abs(diff) < eps:
            diff = eps
        if diff > 0.0:
            return complex(math.sqrt(2.0 * self.mass * diff) / self.hbar, 0.0)
        return complex(0.0, math.sqrt(2.0 * self.mass * (-diff)) / self.hbar)

    @staticmethod
    def interface_matrix(k_left: complex, k_right: complex) -> np.ndarray:
        """
        Constructs 2x2 matching matrix I_{L->R} such that:
            [A_L, B_L]^T = I_{L->R} [A_R, B_R]^T
        satisfying psi_L = psi_R and psi'_L = psi'_R.
        """
        if abs(k_left) == 0.0:
            raise ZeroDivisionError("Left wavevector cannot be zero in interface matching.")
        inv = 0.5 / k_left
        kp = k_left + k_right
        km = k_left - k_right
        return np.array(
            [
                [inv * kp, inv * km],
                [inv * km, inv * kp]
            ],
            dtype=np.complex128
        )

    @staticmethod
    def propagation_matrix(k: complex, width: float) -> np.ndarray:
        """
        Constructs 2x2 propagation matrix P across a slab of width d:
            [A(x_start), B(x_start)]^T = P [A(x_end), B(x_end)]^T
        """
        if width == 0.0:
            return np.eye(2, dtype=np.complex128)
        phase = -1j * k * width
        if np.real(phase) > 700.0 or np.real(-phase) > 700.0:
            raise OverflowError(
                f"Exponential overflow in propagation matrix (Re(-i*k*d)={np.real(phase):.1f})."
            )
        return np.array(
            [
                [np.exp(phase), 0.0 + 0.0j],
                [0.0 + 0.0j, np.exp(-phase)]
            ],
            dtype=np.complex128
        )

    @staticmethod
    def discretize_potential(
        potential: Potential,
        x_left: float,
        x_right: float,
        num_slabs: int = 200
    ) -> Tuple[List[PotentialSlab], float, float]:
        """
        Converts an arbitrary Potential into contiguous piecewise-constant slabs
        over [x_left, x_right], extracting exact geometry where available.
        Returns (slabs, v_left, v_right).
        """
        if x_right <= x_left:
            raise ValueError(f"x_right ({x_right}) must exceed x_left ({x_left}).")
        if num_slabs < 1:
            raise ValueError("num_slabs must be >= 1.")

        if isinstance(potential, BarrierPotential):
            b_left = potential.center - 0.5 * potential.width
            b_right = potential.center + 0.5 * potential.width
            slabs = [PotentialSlab(b_left, b_right, potential.height)]
            return slabs, potential.baseline, potential.baseline

        if isinstance(potential, FiniteSquareWell):
            w_left = potential.center - 0.5 * potential.width
            w_right = potential.center + 0.5 * potential.width
            slabs = [PotentialSlab(w_left, w_right, -potential.depth)]
            return slabs, 0.0, 0.0

        if isinstance(potential, StepPotential):
            x0 = potential.step_position
            slabs = [
                PotentialSlab(x0 - 1.0, x0, potential.left_height),
                PotentialSlab(x0, x0 + 1.0, potential.right_height)
            ]
            return slabs, potential.left_height, potential.right_height

        if isinstance(potential, FreeParticle):
            return [PotentialSlab(x_left, x_right, 0.0)], 0.0, 0.0

        edges = np.linspace(x_left, x_right, num_slabs + 1)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        v_mids = potential(midpoints)
        v_endpoints = potential(np.array([x_left, x_right], dtype=np.float64))

        if not (np.all(np.isfinite(v_mids)) and np.all(np.isfinite(v_endpoints))):
            raise ValueError("Potential must be finite across the scattering domain.")

        slabs = [
            PotentialSlab(float(edges[i]), float(edges[i + 1]), float(v_mids[i]))
            for i in range(num_slabs)
        ]
        return slabs, float(v_endpoints[0]), float(v_endpoints[1])

    def _validate_boundary_matching(
        self,
        slabs: Sequence[PotentialSlab],
        k_slabs: Sequence[complex],
        k_left: complex,
        k_right: complex,
        r_amp: complex,
        t_amp: complex,
        slab_amplitudes: Sequence[Tuple[complex, complex]]
    ) -> float:
        """
        Verifies continuity of psi and psi' at every slab interface from left to right.
        Returns the maximum absolute jump across all interfaces.
        """
        max_jump = 0.0
        # Left asymptotic region evaluated at start of first slab
        psi_prev = 1.0 + r_amp
        dpsi_prev = 1j * k_left * (1.0 - r_amp)

        for slab, k_j, (a_j, b_j) in zip(slabs, k_slabs, slab_amplitudes):
            # Entry of slab j (x = x_start)
            psi_in = a_j + b_j
            dpsi_in = 1j * k_j * (a_j - b_j)
            max_jump = max(max_jump, abs(psi_prev - psi_in), abs(dpsi_prev - dpsi_in))

            # Exit of slab j (x = x_end)
            e_pos = np.exp(1j * k_j * slab.width)
            e_neg = np.exp(-1j * k_j * slab.width)
            psi_prev = a_j * e_pos + b_j * e_neg
            dpsi_prev = 1j * k_j * (a_j * e_pos - b_j * e_neg)

        # Right asymptotic region evaluated at exit of last slab
        psi_right = t_amp
        dpsi_right = 1j * k_right * t_amp
        max_jump = max(max_jump, abs(psi_prev - psi_right), abs(dpsi_prev - dpsi_right))
        return float(max_jump)

    def solve(
        self,
        energy: float,
        slabs: Sequence[PotentialSlab],
        v_left: float = 0.0,
        v_right: float = 0.0
    ) -> TransferMatrixResult:
        """
        Solves 1D scattering across a sequence of contiguous PotentialSlabs
        for incident energy `energy`.
        """
        if energy <= v_left:
            raise ValueError(
                f"Incident energy ({energy}) must exceed left asymptotic potential ({v_left})."
            )
        if not slabs:
            raise ValueError("At least one PotentialSlab is required.")

        for idx in range(1, len(slabs)):
            if not math.isclose(slabs[idx].x_start, slabs[idx - 1].x_end, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"Slabs must be contiguous: slab {idx - 1} ends at {slabs[idx - 1].x_end}, "
                    f"but slab {idx} starts at {slabs[idx].x_start}."
                )

        k_left = self.wavevector(energy, v_left)
        k_right = self.wavevector(energy, v_right)
        k_slabs = [self.wavevector(energy, s.potential) for s in slabs]

        warnings: List[str] = []
        max_evan_arg = 0.0
        for s, k_s in zip(slabs, k_slabs):
            if np.imag(k_s) > 0.0:
                arg = float(np.imag(k_s) * s.width)
                max_evan_arg = max(max_evan_arg, arg)
        if max_evan_arg > self.MAX_EVANESCENT_ARG:
            warnings.append(
                f"Large evanescent argument kappa*d = {max_evan_arg:.1f} may cause floating-point saturation."
            )

        # Build step matrices from left to right and propagate backward for slab amplitudes
        step_matrices: List[np.ndarray] = []
        k_prev = k_left
        for slab, k_curr in zip(slabs, k_slabs):
            i_mat = self.interface_matrix(k_prev, k_curr)
            p_mat = self.propagation_matrix(k_curr, slab.width)
            step_matrices.append(i_mat @ p_mat)
            k_prev = k_curr
        i_final = self.interface_matrix(k_prev, k_right)

        total_m = np.eye(2, dtype=np.complex128)
        for sm in step_matrices:
            total_m = total_m @ sm
        total_m = total_m @ i_final

        if abs(total_m[0, 0]) == 0.0 or not np.all(np.isfinite(total_m)):
            raise ArithmeticError("Transfer matrix became singular or non-finite.")

        t_amp = complex(1.0 / total_m[0, 0])
        r_amp = complex(total_m[1, 0] / total_m[0, 0])

        # Reconstruct internal slab amplitudes [A_j, B_j] referenced at each slab's x_start
        vec = np.array([t_amp, 0.0 + 0.0j], dtype=np.complex128)
        vec = i_final @ vec
        slab_amps_rev: List[Tuple[complex, complex]] = []
        for slab, k_curr in zip(reversed(slabs), reversed(k_slabs)):
            p_mat = self.propagation_matrix(k_curr, slab.width)
            vec = p_mat @ vec
            slab_amps_rev.append((complex(vec[0]), complex(vec[1])))
            # Advance across left interface of this slab
            idx = len(slabs) - len(slab_amps_rev)
            k_before = k_left if idx == 0 else k_slabs[idx - 1]
            vec = self.interface_matrix(k_before, k_curr) @ vec
        slab_amplitudes = list(reversed(slab_amps_rev))

        # Flux-normalized reflection & transmission probabilities
        reflection = ReflectionCalculator.from_amplitude(r_amp)
        transmission = TransmissionCalculator.from_amplitude(
            t_amp, energy=energy, v_left=v_left, v_right=v_right, mass=self.mass, hbar=self.hbar
        )
        cons_err = abs((reflection + transmission) - 1.0)

        # Stability diagnostics
        det_m = complex(np.linalg.det(total_m))
        expected_det = k_right / k_left
        det_err = float(abs(det_m - expected_det))
        cond_num = float(np.linalg.cond(total_m))
        match_err = self._validate_boundary_matching(
            slabs, k_slabs, k_left, k_right, r_amp, t_amp, slab_amplitudes
        )

        if det_err > 1e-6:
            warnings.append(f"Transfer matrix determinant deviation ({det_err:.2e}) exceeds 1e-6.")
        if match_err > 1e-6:
            warnings.append(f"Interface boundary matching error ({match_err:.2e}) exceeds 1e-6.")
        if cons_err > 1e-6:
            warnings.append(f"Probability conservation error |R + T - 1| = {cons_err:.2e}.")

        is_stable = (det_err <= 1e-5) and (match_err <= 1e-5) and (cons_err <= 1e-5)
        stability = TransferMatrixStabilityReport(
            determinant=det_m,
            expected_determinant=expected_det,
            determinant_error=det_err,
            max_evanescent_exponent=max_evan_arg,
            condition_number=cond_num,
            boundary_matching_error=match_err,
            is_stable=is_stable,
            warnings=warnings
        )

        return TransferMatrixResult(
            matrix=total_m,
            r_amplitude=r_amp,
            t_amplitude=t_amp,
            reflection=reflection,
            transmission=transmission,
            probability_conservation_error=cons_err,
            slab_amplitudes=slab_amplitudes,
            stability=stability
        )
