from typing import List, Optional, Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.potentials.base import Potential
from schrodinger.potentials.time_dependent import TimeDependentPotential
from .time_integrators import TimeIntegrator, IntegratorMetadata


class SplitOperatorIntegrator(TimeIntegrator):
    """
    Second-order unitary Split-Operator Fourier Transform (Strang splitting) integrator:
        psi(t + dt) = exp(-i * V(x, t + dt/2) * dt / (2 * hbar))
                      * IFFT[ exp(-i * hbar * k^2 * dt / (2 * m))
                              * FFT( exp(-i * V(x, t + dt/2) * dt / (2 * hbar)) * psi(t) ) ]

    Assumptions & Constraints:
        - Assumes periodic boundary conditions on [x_min, x_max) via FFT.
        - Exact kinetic propagation in momentum space; O(dt^2) splitting error.
        - Strictly unitary and unconditionally stable in O(N log N) operations per step.
        - Requires finite potential values V(x, t) across the entire grid.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        potential: Union[Potential, TimeDependentPotential],
        mass: float = 1.0,
        hbar: float = 1.0
    ):
        super().__init__(grid=grid, potential=potential, mass=mass, hbar=hbar)

        # Discrete Fourier wavenumbers k = 2 * pi * freq
        self._k = 2.0 * np.pi * np.fft.fftfreq(grid.points, d=grid.dx)
        self._kinetic_eigenvalues = ((self.hbar ** 2) * (self._k ** 2)) / (2.0 * self.mass)

        self._static_v: Optional[np.ndarray] = None
        if not self.is_time_dependent:
            self._static_v = self._eval_potential_full(0.0)

        self._cached_dt: Optional[float] = None
        self._cached_u_kinetic: Optional[np.ndarray] = None
        self._cached_u_pot_half: Optional[np.ndarray] = None

    @property
    def metadata(self) -> IntegratorMetadata:
        return IntegratorMetadata(
            name="Split-Operator FFT",
            temporal_order=2,
            is_unconditionally_unitary=True,
            supports_time_dependent_potential=True,
            boundary_assumption="Periodic (discrete Fourier basis on [x_min, x_max))",
            constraints=[
                "Assumes periodic domain; wavepackets reaching edges wrap around.",
                "Requires finite potential values across all grid points (no infinite walls).",
                "Maximum representable wavevector is bounded by Nyquist limit pi / dx."
            ]
        )

    def _eval_potential_full(self, t: float) -> np.ndarray:
        v = np.asarray(self._td_potential(self.grid.grid, t), dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError(
                "SplitOperatorIntegrator requires finite potential values across the entire grid."
            )
        return v

    def validate_stability(self, dt: float) -> List[str]:
        warnings = super().validate_stability(dt)
        v0 = self._static_v if self._static_v is not None else self._eval_potential_full(0.0)
        max_phase = (float(np.max(np.abs(v0))) * dt) / self.hbar
        if max_phase > np.pi:
            warnings.append(
                f"Large potential phase rotation per step ({max_phase:.2f} rad > pi). "
                "Consider reducing dt for splitting accuracy."
            )
        return warnings

    def step(self, psi: np.ndarray, t: float, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be strictly positive.")

        if self._cached_u_kinetic is None or self._cached_dt != dt:
            self._cached_u_kinetic = np.exp(-1j * self._kinetic_eigenvalues * dt / self.hbar)
            if self._static_v is not None:
                self._cached_u_pot_half = np.exp(-0.5j * self._static_v * dt / self.hbar)
            else:
                self._cached_u_pot_half = None
            self._cached_dt = dt

        if self._cached_u_pot_half is not None:
            u_v_half = self._cached_u_pot_half
        else:
            v_mid = self._eval_potential_full(t + 0.5 * dt)
            u_v_half = np.exp(-0.5j * v_mid * dt / self.hbar)

        psi_half = u_v_half * psi
        psi_k = np.fft.fft(psi_half)
        psi_k = self._cached_u_kinetic * psi_k
        psi_after_t = np.fft.ifft(psi_k)
        return u_v_half * psi_after_t
