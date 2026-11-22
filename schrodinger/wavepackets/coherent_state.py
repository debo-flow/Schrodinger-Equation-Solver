import math
from dataclasses import dataclass
from typing import Dict
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.numerical.finite_difference import FiniteDifferenceEngine
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod
from schrodinger.observables.position_operator import PositionOperator
from schrodinger.observables.momentum_operator import MomentumOperator
from schrodinger.observables.observable_engine import ObservableEngine
from schrodinger.observables.uncertainty import UncertaintyAnalyzer


@dataclass
class CoherentStateProperties:
    """Exact analytical properties of a 1D Quantum Harmonic Oscillator coherent state |alpha>."""
    alpha: complex
    mass: float
    omega: float
    hbar: float
    center: float
    oscillator_length: float
    expectation_x: float
    expectation_p: float
    uncertainty_x: float
    uncertainty_p: float
    uncertainty_product: float
    mean_excitation_number: float
    expected_energy: float


@dataclass
class CoherentStateValidation:
    """Comparison of numerically discretized coherent state against exact analytical expectations."""
    expectation_x_error: float
    expectation_p_error: float
    uncertainty_x_error: float
    uncertainty_p_error: float
    uncertainty_product: float
    heisenberg_minimum_error: float
    is_valid: bool


class CoherentStateGenerator:
    """
    Generates and validates Glauber coherent states |alpha> of a 1D Harmonic Oscillator:
        psi_alpha(x) = (m*omega/(pi*hbar))^(1/4) * exp(-(x - x0)^2 / (2*xi0^2) + i*k0*(x - x_c) + i*phi0)
    """

    @staticmethod
    def analytical_properties(
        alpha: complex,
        mass: float = 1.0,
        omega: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0
    ) -> CoherentStateProperties:
        if mass <= 0 or omega <= 0 or hbar <= 0:
            raise ValueError("mass, omega, and hbar must be strictly positive.")
        if not (np.isfinite(alpha) and math.isfinite(center)):
            raise ValueError("alpha and center must be finite.")

        alpha_c = complex(alpha)
        xi0 = math.sqrt(hbar / (mass * omega))
        exp_x = center + math.sqrt(2.0) * xi0 * float(np.real(alpha_c))
        exp_p = math.sqrt(2.0) * (hbar / xi0) * float(np.imag(alpha_c))
        delta_x = xi0 / math.sqrt(2.0)
        delta_p = hbar / (math.sqrt(2.0) * xi0)
        mean_n = float(abs(alpha_c) ** 2)
        exp_e = float(hbar * omega * (mean_n + 0.5))

        return CoherentStateProperties(
            alpha=alpha_c,
            mass=float(mass),
            omega=float(omega),
            hbar=float(hbar),
            center=float(center),
            oscillator_length=xi0,
            expectation_x=exp_x,
            expectation_p=exp_p,
            uncertainty_x=delta_x,
            uncertainty_p=delta_p,
            uncertainty_product=0.5 * hbar,
            mean_excitation_number=mean_n,
            expected_energy=exp_e
        )

    @staticmethod
    def fock_occupation_probabilities(alpha: complex, max_n: int = 10) -> Dict[int, float]:
        """Returns exact Poissonian number-state probabilities P(n) = exp(-|alpha|^2) * |alpha|^(2n) / n!."""
        if max_n < 0:
            raise ValueError("max_n must be >= 0.")
        mean_n = float(abs(alpha) ** 2)
        if mean_n == 0.0:
            return {n: (1.0 if n == 0 else 0.0) for n in range(max_n + 1)}

        log_mean = math.log(mean_n)
        return {
            n: float(math.exp(-mean_n + n * log_mean - math.lgamma(n + 1)))
            for n in range(max_n + 1)
        }

    @classmethod
    def generate(
        cls,
        grid: SpatialGrid,
        alpha: complex,
        mass: float = 1.0,
        omega: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0,
        enforce_dirichlet_boundaries: bool = False,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Wavefunction:
        """Constructs a continuously normalized Harmonic Oscillator coherent state on `grid`."""
        props = cls.analytical_properties(alpha=alpha, mass=mass, omega=omega, hbar=hbar, center=center)
        k0 = props.expectation_p / props.hbar

        nyquist_k = math.pi / grid.dx
        if abs(k0) >= nyquist_k:
            raise ValueError(
                f"Coherent state carrier wavevector |k0|={abs(k0):.3f} exceeds Nyquist limit {nyquist_k:.3f}."
            )

        x = grid.grid.astype(np.float64)
        xi0 = props.oscillator_length
        x0 = props.expectation_x
        prefactor = (math.pi * (xi0 ** 2)) ** (-0.25)
        phase0 = -0.5 * (x0 - props.center) * k0
        psi_raw = prefactor * np.exp(-((x - x0) ** 2) / (2.0 * (xi0 ** 2)) + 1j * k0 * (x - props.center) + 1j * phase0)

        if enforce_dirichlet_boundaries and grid.boundary_condition == BoundaryCondition.DIRICHLET:
            psi_raw = psi_raw.copy()
            psi_raw[0] = 0.0 + 0.0j
            psi_raw[-1] = 0.0 + 0.0j

        psi_norm, _ = NormalizationEngine.normalize(psi_raw, grid.grid, method=integration_method)
        return Wavefunction(x=grid.grid, psi=psi_norm, energy=props.expected_energy)

    @classmethod
    def validate_coherent_state(
        cls,
        wf: Wavefunction,
        grid: SpatialGrid,
        alpha: complex,
        mass: float = 1.0,
        omega: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0,
        tolerance: float = 1e-2
    ) -> CoherentStateValidation:
        """Validates numerical <x>, <p>, Delta_x, Delta_p, and Delta_x*Delta_p = hbar/2."""
        props = cls.analytical_properties(alpha=alpha, mass=mass, omega=omega, hbar=hbar, center=center)
        fd = FiniteDifferenceEngine(grid)

        x_op = PositionOperator.build_x(grid)
        x2_op = PositionOperator.build_x_squared(grid)
        p_op = MomentumOperator.build_p(fd, hbar=hbar, accuracy_order=4)
        p2_op = MomentumOperator.build_p_squared(fd, hbar=hbar, accuracy_order=4)

        num_x = ObservableEngine.expectation(x_op, wf, is_hermitian=True)
        num_p = ObservableEngine.expectation(p_op, wf, is_hermitian=True)
        num_dx = UncertaintyAnalyzer.calculate_uncertainty(x_op, x2_op, wf)
        num_dp = UncertaintyAnalyzer.calculate_uncertainty(p_op, p2_op, wf)
        product = num_dx * num_dp

        err_x = abs(num_x - props.expectation_x)
        err_p = abs(num_p - props.expectation_p)
        err_dx = abs(num_dx - props.uncertainty_x)
        err_dp = abs(num_dp - props.uncertainty_p)
        err_heis = abs(product - props.uncertainty_product)

        is_valid = max(err_x, err_p, err_dx, err_dp, err_heis) <= tolerance
        return CoherentStateValidation(
            expectation_x_error=float(err_x),
            expectation_p_error=float(err_p),
            uncertainty_x_error=float(err_dx),
            uncertainty_p_error=float(err_dp),
            uncertainty_product=float(product),
            heisenberg_minimum_error=float(err_heis),
            is_valid=is_valid
        )
