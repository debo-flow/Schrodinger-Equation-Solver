import json
import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional, Sequence
import numpy as np
from scipy import integrate
from schrodinger.verification.error_analysis import ErrorAnalyzer


@dataclass
class NormConservationReport:
    """Summary of probability norm conservation across time propagation."""
    initial_norm: float
    final_norm: float
    min_norm: float
    max_norm: float
    max_norm_deviation: float
    is_conserved: bool


@dataclass
class TimeStepConvergencePoint:
    """Wavefunction L2 error recorded for a given time-step size dt."""
    dt: float
    l2_error: float
    max_abs_error: float
    final_norm_deviation: float


@dataclass
class TimeStepConvergenceReport:
    """Empirical temporal order estimation across refined time steps."""
    points: List[TimeStepConvergencePoint]
    estimated_temporal_order: Optional[float]
    is_monotonically_converging: bool
    converged: bool


@dataclass
class EvolutionValidationReport:
    """Structured validation report for a time-dependent simulation run."""
    integrator_name: str
    num_steps: int
    norm_report: NormConservationReport
    initial_energy: Optional[float]
    final_energy: Optional[float]
    max_energy_drift: Optional[float]
    all_finite: bool
    warnings: List[str]
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class EvolutionValidator:
    """
    Validates norm conservation, energy stability, temporal convergence order,
    and analytical time-evolution benchmarks.
    """

    @staticmethod
    def validate_norm_conservation(
        norms: Sequence[float],
        tolerance: float = 1e-4
    ) -> NormConservationReport:
        arr = np.asarray(norms, dtype=np.float64)
        if len(arr) == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("norms must be a non-empty sequence of finite floats.")

        deviations = np.abs(arr - 1.0)
        max_dev = float(np.max(deviations))
        return NormConservationReport(
            initial_norm=float(arr[0]),
            final_norm=float(arr[-1]),
            min_norm=float(np.min(arr)),
            max_norm=float(np.max(arr)),
            max_norm_deviation=max_dev,
            is_conserved=(max_dev <= tolerance)
        )

    @staticmethod
    def analytical_stationary_state(
        psi_initial: np.ndarray,
        energy: float,
        t: float,
        hbar: float = 1.0
    ) -> np.ndarray:
        """Exact time evolution of a stationary eigenstate: psi(x, t) = psi(x, 0) * exp(-i*E*t/hbar)."""
        return np.asarray(psi_initial, dtype=np.complex128) * np.exp(-1j * energy * t / hbar)

    @staticmethod
    def analytical_free_gaussian_wavepacket(
        x: np.ndarray,
        t: float,
        x0: float = 0.0,
        sigma: float = 0.5,
        k0: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> np.ndarray:
        """
        Exact analytical dispersion and propagation of a 1D free-particle Gaussian wavepacket:
            psi(x, t) = (2*pi*sigma^2)^(-1/4) * gamma(t)^(-1/2)
                        * exp(-(x - x0 - v_g*t)^2 / (4*sigma^2*gamma(t)) + i*k0*(x - x0) - i*hbar*k0^2*t/(2m))
        where gamma(t) = 1 + i * hbar * t / (2 * m * sigma^2) and v_g = hbar * k0 / m.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        gamma = 1.0 + (1j * hbar * t) / (2.0 * mass * (sigma ** 2))
        v_group = (hbar * k0) / mass
        disp = x_arr - x0 - v_group * t

        prefactor = ((2.0 * math.pi * (sigma ** 2)) ** (-0.25)) / np.sqrt(gamma)
        exponent = (
            -(disp ** 2) / (4.0 * (sigma ** 2) * gamma)
            + 1j * k0 * (x_arr - x0)
            - 1j * ((hbar * (k0 ** 2)) / (2.0 * mass)) * t
        )
        return prefactor * np.exp(exponent)

    @staticmethod
    def analytical_coherent_oscillator_state(
        x: np.ndarray,
        t: float,
        alpha: complex,
        mass: float = 1.0,
        omega: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0
    ) -> np.ndarray:
        """
        Exact analytical time evolution of a Glauber coherent state |alpha(t)> = |alpha * exp(-i*omega*t)>
        in a 1D Quantum Harmonic Oscillator.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        xi0 = math.sqrt(hbar / (mass * omega))
        alpha_t = alpha * np.exp(-1j * omega * t)

        x_t = center + math.sqrt(2.0) * xi0 * float(np.real(alpha_t))
        p_t = math.sqrt(2.0) * (hbar / xi0) * float(np.imag(alpha_t))
        k_t = p_t / hbar
        phase_t = -0.5 * omega * t - 0.5 * (x_t - center) * k_t

        prefactor = (math.pi * (xi0 ** 2)) ** (-0.25)
        return prefactor * np.exp(-((x_arr - x_t) ** 2) / (2.0 * (xi0 ** 2)) + 1j * k_t * (x_arr - center) + 1j * phase_t)

    @staticmethod
    def analyze_timestep_convergence(
        propagate_fn: Callable[[float], np.ndarray],
        dt_values: Sequence[float],
        reference_psi: np.ndarray,
        x: np.ndarray,
        min_expected_order: float = 1.5
    ) -> TimeStepConvergenceReport:
        """
        Evaluates temporal convergence order p in Error ~ C * dt^p across decreasing dt values.
        """
        sorted_dts = sorted([float(d) for d in dt_values], reverse=True)
        if len(sorted_dts) < 2:
            raise ValueError("Time-step convergence analysis requires at least 2 distinct dt values.")

        points: List[TimeStepConvergencePoint] = []
        for dt in sorted_dts:
            psi_num = propagate_fn(dt)
            l2_err = ErrorAnalyzer.l2_wavefunction_error(psi_num, reference_psi, x)
            max_err = ErrorAnalyzer.max_absolute_error(psi_num, reference_psi)
            norm_val = float(integrate.simpson(y=np.abs(psi_num) ** 2, x=x))
            points.append(
                TimeStepConvergencePoint(
                    dt=dt,
                    l2_error=l2_err,
                    max_abs_error=max_err,
                    final_norm_deviation=abs(norm_val - 1.0)
                )
            )

        dts = np.array([p.dt for p in points], dtype=np.float64)
        errs = np.array([p.l2_error for p in points], dtype=np.float64)
        valid = (dts > 0) & (errs > 1e-14) & np.isfinite(errs)

        est_order: Optional[float] = None
        if np.sum(valid) >= 2:
            slope, _ = np.polyfit(np.log(dts[valid]), np.log(errs[valid]), 1)
            est_order = float(slope)

        is_monotonic = all(errs[i] <= errs[i - 1] * (1.0 + 1e-9) for i in range(1, len(errs)))
        converged = is_monotonic and (est_order is not None and est_order >= min_expected_order)

        return TimeStepConvergenceReport(
            points=points,
            estimated_temporal_order=est_order,
            is_monotonically_converging=is_monotonic,
            converged=converged
        )

    @classmethod
    def validate_run(
        cls,
        integrator_name: str,
        num_steps: int,
        norms: Sequence[float],
        final_psi: np.ndarray,
        is_time_independent: bool = True,
        energies: Optional[Sequence[float]] = None,
        norm_tolerance: float = 1e-4,
        energy_drift_tolerance: float = 1e-3,
        existing_warnings: Optional[Sequence[str]] = None
    ) -> EvolutionValidationReport:
        """Compiles norm, finiteness, and energy conservation diagnostics into a report."""
        warnings: List[str] = list(existing_warnings) if existing_warnings else []
        all_finite = bool(np.all(np.isfinite(final_psi)) and np.all(np.isfinite(norms)))
        if not all_finite:
            warnings.append("Non-finite values (NaN/Inf) detected during time propagation.")

        norm_report = cls.validate_norm_conservation(norms, tolerance=norm_tolerance)
        if not norm_report.is_conserved:
            warnings.append(
                f"Norm conservation deviation {norm_report.max_norm_deviation:.2e} "
                f"exceeds tolerance {norm_tolerance:.2e}."
            )

        init_e: Optional[float] = None
        fin_e: Optional[float] = None
        max_drift: Optional[float] = None

        if energies is not None and len(energies) > 0:
            e_arr = np.asarray(energies, dtype=np.float64)
            init_e = float(e_arr[0])
            fin_e = float(e_arr[-1])
            max_drift = float(np.max(np.abs(e_arr - init_e)))
            if is_time_independent and max_drift > energy_drift_tolerance:
                warnings.append(
                    f"Time-independent energy drift {max_drift:.2e} exceeds tolerance {energy_drift_tolerance:.2e}."
                )

        is_valid = all_finite and norm_report.is_conserved and (len(warnings) == 0)
        return EvolutionValidationReport(
            integrator_name=integrator_name,
            num_steps=num_steps,
            norm_report=norm_report,
            initial_energy=init_e,
            final_energy=fin_e,
            max_energy_drift=max_drift,
            all_finite=all_finite,
            warnings=warnings,
            is_valid=is_valid
        )
