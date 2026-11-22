import math
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod


@dataclass
class GaussianWavepacketSpec:
    """Configuration and analytical parameters for a 1D Gaussian wavepacket."""
    x0: float = 0.0
    sigma: float = 0.5
    k0: float = 0.0
    mass: float = 1.0
    hbar: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.x0):
            raise ValueError(f"Initial position x0 must be finite. Received: {self.x0}")
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError(f"Spatial width sigma must be finite and strictly positive. Received: {self.sigma}")
        if not math.isfinite(self.k0):
            raise ValueError(f"Central wavenumber k0 must be finite. Received: {self.k0}")
        if not math.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError(f"Particle mass must be strictly positive. Received: {self.mass}")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise ValueError(f"Reduced Planck constant hbar must be strictly positive. Received: {self.hbar}")

    @property
    def analytical_normalization(self) -> float:
        """Returns exact continuous normalization constant A = (2 * pi * sigma^2)^(-1/4)."""
        return float((2.0 * math.pi * (self.sigma ** 2)) ** (-0.25))

    @property
    def analytical_momentum(self) -> float:
        """Returns exact momentum expectation <p> = hbar * k0."""
        return float(self.hbar * self.k0)

    @property
    def analytical_momentum_uncertainty(self) -> float:
        """Returns exact momentum standard deviation Delta_p = hbar / (2 * sigma)."""
        return float(self.hbar / (2.0 * self.sigma))

    @property
    def analytical_free_energy(self) -> float:
        """Returns exact free-particle kinetic energy <E> = hbar^2 / (2m) * (k0^2 + 1/(4*sigma^2))."""
        return float(((self.hbar ** 2) / (2.0 * self.mass)) * ((self.k0 ** 2) + 1.0 / (4.0 * (self.sigma ** 2))))


class GaussianWavepacketGenerator:
    """
    Generates normalized 1D Gaussian wavepackets:
        psi(x, 0) = A * exp(-(x - x0)^2 / (4 * sigma^2)) * exp(i * k0 * x)
    with numerical normalization verification and analytical spreading formulas.
    """

    @staticmethod
    def evaluate_raw(grid: SpatialGrid, spec: GaussianWavepacketSpec) -> np.ndarray:
        """Evaluates the unclipped analytical Gaussian wavepacket on `grid`."""
        x = grid.grid.astype(np.float64)
        envelope = np.exp(-((x - spec.x0) ** 2) / (4.0 * (spec.sigma ** 2)))
        carrier = np.exp(1j * spec.k0 * x)
        return spec.analytical_normalization * envelope * carrier

    @classmethod
    def generate(
        cls,
        grid: SpatialGrid,
        x0: float = 0.0,
        sigma: float = 0.5,
        k0: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        enforce_dirichlet_boundaries: bool = False,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Wavefunction:
        """
        Constructs a continuously normalized Gaussian wavepacket Wavefunction on `grid`.
        """
        spec = GaussianWavepacketSpec(x0=x0, sigma=sigma, k0=k0, mass=mass, hbar=hbar)

        # Nyquist spatial frequency check: |k0| must be strictly below pi / dx
        nyquist_k = math.pi / grid.dx
        if abs(spec.k0) >= nyquist_k:
            raise ValueError(
                f"Central wavenumber |k0|={abs(spec.k0):.3f} exceeds grid Nyquist limit "
                f"pi/dx={nyquist_k:.3f}. Increase grid resolution."
            )

        psi_raw = cls.evaluate_raw(grid, spec)
        if enforce_dirichlet_boundaries and grid.boundary_condition == BoundaryCondition.DIRICHLET:
            psi_raw = psi_raw.copy()
            psi_raw[0] = 0.0 + 0.0j
            psi_raw[-1] = 0.0 + 0.0j

        psi_norm, _ = NormalizationEngine.normalize(psi_raw, grid.grid, method=integration_method)
        return Wavefunction(x=grid.grid, psi=psi_norm, energy=spec.analytical_free_energy)

    @staticmethod
    def analytical_spreading_width(sigma0: float, t: float, mass: float = 1.0, hbar: float = 1.0) -> float:
        """
        Exact free-particle Gaussian spatial width at time t:
            sigma(t) = sigma0 * sqrt(1 + (hbar * t / (2 * m * sigma0^2))^2)
        """
        if sigma0 <= 0 or mass <= 0 or hbar <= 0:
            raise ValueError("sigma0, mass, and hbar must be strictly positive.")
        tau = (2.0 * mass * (sigma0 ** 2)) / hbar
        return float(sigma0 * math.sqrt(1.0 + (t / tau) ** 2))

    @staticmethod
    def analytical_free_evolution(
        x: np.ndarray,
        t: float,
        x0: float = 0.0,
        sigma: float = 0.5,
        k0: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> np.ndarray:
        """
        Exact free-particle time evolution matching psi(x, 0) = A * exp(-(x-x0)^2/(4*sigma^2)) * exp(i*k0*x).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        gamma = 1.0 + (1j * hbar * t) / (2.0 * mass * (sigma ** 2))
        v_group = (hbar * k0) / mass
        disp = x_arr - x0 - v_group * t

        prefactor = ((2.0 * math.pi * (sigma ** 2)) ** (-0.25)) / np.sqrt(gamma)
        exponent = (
            -(disp ** 2) / (4.0 * (sigma ** 2) * gamma)
            + 1j * k0 * x_arr
            - 1j * ((hbar * (k0 ** 2)) / (2.0 * mass)) * t
        )
        return prefactor * np.exp(exponent)

    @staticmethod
    def expected_properties(spec: GaussianWavepacketSpec) -> Dict[str, Any]:
        """Returns dictionary of exact analytical expectations at t = 0."""
        return {
            "expectation_x": spec.x0,
            "uncertainty_x": spec.sigma,
            "expectation_p": spec.analytical_momentum,
            "uncertainty_p": spec.analytical_momentum_uncertainty,
            "uncertainty_product": 0.5 * spec.hbar,
            "free_particle_energy": spec.analytical_free_energy
        }
