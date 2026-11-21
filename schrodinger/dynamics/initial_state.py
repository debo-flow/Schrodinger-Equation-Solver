import math
from typing import Sequence
import numpy as np
from schrodinger.numerical.grid import SpatialGrid, BoundaryCondition
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod


class InitialStateBuilder:
    """
    Constructs validated, continuously normalized initial quantum states psi(x, 0)
    on a SpatialGrid.
    """

    @staticmethod
    def _finalize_wavefunction(
        grid: SpatialGrid,
        psi_raw: np.ndarray,
        energy: float = 0.0,
        enforce_dirichlet_boundaries: bool = False,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Wavefunction:
        psi_c = np.asarray(psi_raw, dtype=np.complex128).copy()
        if psi_c.shape != (grid.points,):
            raise ValueError(f"Expected wavefunction shape {(grid.points,)}, received {psi_c.shape}.")
        if not np.all(np.isfinite(psi_c)):
            raise ValueError("Initial wavefunction contains NaN or Inf values.")

        if enforce_dirichlet_boundaries and grid.boundary_condition == BoundaryCondition.DIRICHLET:
            psi_c[0] = 0.0 + 0.0j
            psi_c[-1] = 0.0 + 0.0j

        psi_norm, _ = NormalizationEngine.normalize(psi_c, grid.grid, method=integration_method)
        return Wavefunction(x=grid.grid, psi=psi_norm, energy=float(energy))

    @classmethod
    def gaussian_wavepacket(
        cls,
        grid: SpatialGrid,
        x0: float = 0.0,
        sigma: float = 0.5,
        k0: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        enforce_dirichlet_boundaries: bool = False
    ) -> Wavefunction:
        """
        Constructs a normalized Gaussian wavepacket:
            psi(x, 0) = (2 * pi * sigma^2)^(-1/4) * exp(-(x - x0)^2 / (4 * sigma^2) + i * k0 * (x - x0))
        """
        if not math.isfinite(x0):
            raise ValueError("Center position x0 must be finite.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Wavepacket width sigma must be strictly positive. Received: {sigma}")
        if not math.isfinite(k0):
            raise ValueError("Carrier wavevector k0 must be finite.")
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")

        x = grid.grid
        prefactor = (2.0 * math.pi * (sigma ** 2)) ** (-0.25)
        envelope = np.exp(-((x - x0) ** 2) / (4.0 * (sigma ** 2)))
        phase = np.exp(1j * k0 * (x - x0))
        psi = prefactor * envelope * phase

        # Approximate free kinetic energy expectation: <E> = hbar^2 / (2m) * (k0^2 + 1/(4*sigma^2))
        est_energy = ((hbar ** 2) / (2.0 * mass)) * ((k0 ** 2) + 1.0 / (4.0 * (sigma ** 2)))
        return cls._finalize_wavefunction(
            grid,
            psi,
            energy=est_energy,
            enforce_dirichlet_boundaries=enforce_dirichlet_boundaries
        )

    @classmethod
    def coherent_state(
        cls,
        grid: SpatialGrid,
        alpha: complex,
        mass: float = 1.0,
        omega: float = 1.0,
        hbar: float = 1.0,
        center: float = 0.0
    ) -> Wavefunction:
        """
        Constructs a normalized Glauber coherent state |alpha> of a harmonic oscillator
        with mass m, frequency omega, and equilibrium center x_c.
        """
        if mass <= 0 or omega <= 0 or hbar <= 0:
            raise ValueError("mass, omega, and hbar must be strictly positive.")
        if not (np.isfinite(alpha) and math.isfinite(center)):
            raise ValueError("alpha and center must be finite.")

        xi0 = math.sqrt(hbar / (mass * omega))
        x0 = center + math.sqrt(2.0) * xi0 * float(np.real(alpha))
        p0 = math.sqrt(2.0) * (hbar / xi0) * float(np.imag(alpha))
        k0 = p0 / hbar

        x = grid.grid
        prefactor = (math.pi * (xi0 ** 2)) ** (-0.25)
        phase0 = -0.5 * (x0 - center) * k0
        psi = prefactor * np.exp(-((x - x0) ** 2) / (2.0 * (xi0 ** 2)) + 1j * k0 * (x - center) + 1j * phase0)

        energy = hbar * omega * (abs(alpha) ** 2 + 0.5)
        return cls._finalize_wavefunction(grid, psi, energy=energy)

    @classmethod
    def superposition_state(
        cls,
        eigenstates: Sequence[Wavefunction],
        coefficients: Sequence[complex]
    ) -> Wavefunction:
        """
        Constructs a normalized linear superposition psi(x, 0) = sum_n c_n * psi_n(x).
        """
        if not eigenstates:
            raise ValueError("eigenstates must not be empty.")
        if len(eigenstates) != len(coefficients):
            raise ValueError(
                f"Number of eigenstates ({len(eigenstates)}) must match coefficients ({len(coefficients)})."
            )

        x_ref = eigenstates[0].x
        psi_sum = np.zeros_like(x_ref, dtype=np.complex128)
        weight_sum = 0.0
        weighted_energy = 0.0

        for wf, coeff in zip(eigenstates, coefficients):
            if wf.x.shape != x_ref.shape or not np.allclose(wf.x, x_ref):
                raise ValueError("All eigenstates in superposition must share the same spatial grid.")
            if not np.isfinite(coeff):
                raise ValueError("Superposition coefficients must be finite.")
            c = complex(coeff)
            psi_sum += c * wf.psi
            mag2 = abs(c) ** 2
            weight_sum += mag2
            weighted_energy += mag2 * wf.energy

        if weight_sum <= 0.0:
            raise ValueError("At least one superposition coefficient must be non-zero.")

        psi_norm, _ = NormalizationEngine.normalize(psi_sum, x_ref, method=IntegrationMethod.SIMPSON)
        return Wavefunction(x=x_ref, psi=psi_norm, energy=weighted_energy / weight_sum)

    @classmethod
    def from_array(
        cls,
        grid: SpatialGrid,
        psi: np.ndarray,
        energy: float = 0.0,
        enforce_dirichlet_boundaries: bool = False
    ) -> Wavefunction:
        """Creates a validated, normalized Wavefunction from a raw complex/real array."""
        return cls._finalize_wavefunction(
            grid,
            psi,
            energy=energy,
            enforce_dirichlet_boundaries=enforce_dirichlet_boundaries
        )
