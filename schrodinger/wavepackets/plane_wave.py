import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.wavefunctions.wavefunction import Wavefunction
from schrodinger.wavefunctions.normalization import NormalizationEngine, IntegrationMethod


@dataclass
class PlaneWaveSpec:
    """Physical parameters for a 1D plane-wave momentum eigenstate."""
    k: float
    mass: float = 1.0
    hbar: float = 1.0
    taper_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.k):
            raise ValueError(f"Wavevector k must be finite. Received: {self.k}")
        if not math.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError(f"mass must be strictly positive. Received: {self.mass}")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise ValueError(f"hbar must be strictly positive. Received: {self.hbar}")
        if not (0.0 <= self.taper_fraction < 1.0):
            raise ValueError(f"taper_fraction must be in [0.0, 1.0). Received: {self.taper_fraction}")

    @property
    def momentum(self) -> float:
        """Exact momentum eigenvalue p = hbar * k."""
        return float(self.hbar * self.k)

    @property
    def kinetic_energy(self) -> float:
        """Exact free-particle dispersion energy E = hbar^2 * k^2 / (2 * m)."""
        return float(((self.hbar ** 2) * (self.k ** 2)) / (2.0 * self.mass))

    @property
    def wavelength(self) -> Optional[float]:
        """De Broglie wavelength lambda = 2 * pi / |k| (None if k == 0)."""
        if self.k == 0.0:
            return None
        return float(2.0 * math.pi / abs(self.k))


class PlaneWaveGenerator:
    """
    Constructs box-normalized 1D plane-wave states:
        psi_k(x) = (1 / sqrt(L)) * exp(i * k * x)
    with optional periodic mode quantization k_n = 2 * pi * n / L and optional
    smooth Tukey apodization for Dirichlet boundary compatibility.
    """

    @staticmethod
    def periodic_wavevector(grid: SpatialGrid, mode_index: int) -> float:
        """Computes periodic box wavevector k_n = 2 * pi * n / L."""
        if not isinstance(mode_index, (int, np.integer)):
            raise ValueError(f"mode_index must be an integer. Received: {mode_index}")
        length = grid.x_max - grid.x_min
        return float(2.0 * math.pi * int(mode_index) / length)

    @staticmethod
    def _tukey_window(n_points: int, alpha: float) -> np.ndarray:
        """Generates a symmetric Tukey (cosine-tapered) window of length n_points."""
        if alpha <= 0.0:
            return np.ones(n_points, dtype=np.float64)
        idx = np.linspace(0.0, 1.0, n_points)
        win = np.ones(n_points, dtype=np.float64)
        left = idx < 0.5 * alpha
        right = idx > (1.0 - 0.5 * alpha)
        win[left] = 0.5 * (1.0 - np.cos(2.0 * math.pi * idx[left] / alpha))
        win[right] = 0.5 * (1.0 - np.cos(2.0 * math.pi * (1.0 - idx[right]) / alpha))
        return win

    @classmethod
    def generate(
        cls,
        grid: SpatialGrid,
        k: Optional[float] = None,
        mode_index: Optional[int] = None,
        mass: float = 1.0,
        hbar: float = 1.0,
        taper_fraction: float = 0.0,
        integration_method: IntegrationMethod = IntegrationMethod.SIMPSON
    ) -> Wavefunction:
        """
        Generates a continuously normalized plane-wave Wavefunction on `grid`.
        Specify either continuous wavevector `k` or integer periodic `mode_index`.
        """
        if (k is None) == (mode_index is None):
            raise ValueError("Specify exactly one of `k` or `mode_index`.")

        k_eff = cls.periodic_wavevector(grid, mode_index) if mode_index is not None else float(k)  # type: ignore[arg-type]
        spec = PlaneWaveSpec(k=k_eff, mass=mass, hbar=hbar, taper_fraction=taper_fraction)

        nyquist_k = math.pi / grid.dx
        if abs(spec.k) >= nyquist_k:
            raise ValueError(
                f"Wavevector |k|={abs(spec.k):.3f} exceeds grid Nyquist limit pi/dx={nyquist_k:.3f}."
            )

        x = grid.grid.astype(np.float64)
        length = grid.x_max - grid.x_min
        psi_raw = (1.0 / math.sqrt(length)) * np.exp(1j * spec.k * x)

        if spec.taper_fraction > 0.0:
            psi_raw = psi_raw * cls._tukey_window(grid.points, spec.taper_fraction)

        psi_norm, _ = NormalizationEngine.normalize(psi_raw, x, method=integration_method)
        return Wavefunction(x=grid.grid, psi=psi_norm, energy=spec.kinetic_energy)

    @staticmethod
    def is_periodic_on_grid(wf: Wavefunction, tolerance: float = 1e-6) -> bool:
        """Checks whether psi(x_min) == psi(x_max) within tolerance."""
        return bool(abs(wf.psi[0] - wf.psi[-1]) <= tolerance)
