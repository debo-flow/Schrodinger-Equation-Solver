import math
from dataclasses import dataclass
from typing import Union
import numpy as np
from schrodinger.numerical.grid import SpatialGrid
from schrodinger.wavefunctions.wavefunction import Wavefunction


@dataclass
class MomentumRepresentation:
    """
    Momentum-space representation of a 1D quantum state under the unitary Fourier convention:
        psi_tilde(k) = (1 / sqrt(2 * pi)) * integral psi(x) * exp(-i * k * x) dx
        phi_tilde(p) = psi_tilde(p / hbar) / sqrt(hbar),   p = hbar * k
    satisfying Parseval normalization:
        sum |phi_tilde(p)|^2 * dp = sum |psi_tilde(k)|^2 * dk = sum |psi(x)|^2 * dx = 1.
    """
    k: np.ndarray
    p: np.ndarray
    dk: float
    dp: float
    psi_k: np.ndarray
    psi_p: np.ndarray
    k_probability_density: np.ndarray
    p_probability_density: np.ndarray
    fourier_norm: float
    expectation_p: float
    uncertainty_p: float
    hbar: float
    x_min: float


class MomentumSpaceEngine:
    """
    Performs unitary discrete Fourier transformations between position space psi(x)
    and momentum space phi_tilde(p), preserving Parseval normalization and physical units.
    """

    @staticmethod
    def to_momentum_space(
        state: Union[Wavefunction, np.ndarray],
        grid: SpatialGrid,
        hbar: float = 1.0
    ) -> MomentumRepresentation:
        """Transforms position wavefunction psi(x) into centered momentum space."""
        if hbar <= 0:
            raise ValueError("hbar must be strictly positive.")

        psi_x = state.psi if isinstance(state, Wavefunction) else np.asarray(state, dtype=np.complex128)
        if psi_x.shape != (grid.points,):
            raise ValueError(f"Wavefunction shape {psi_x.shape} does not match grid {(grid.points,)}.")
        if not np.all(np.isfinite(psi_x)):
            raise ValueError("Wavefunction contains non-finite values.")

        n = grid.points
        dx = grid.dx
        dk = 2.0 * math.pi / (n * dx)
        dp = hbar * dk

        # Centered wavenumber & momentum grids in [-pi/dx, pi/dx)
        k_unshifted = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
        k_shifted = np.fft.fftshift(k_unshifted)
        p_shifted = hbar * k_shifted

        # Continuous-limit Fourier transform with exact x_min phase origin
        fft_vals = np.fft.fft(psi_x)
        psi_k_unshifted = (dx / math.sqrt(2.0 * math.pi)) * np.exp(-1j * k_unshifted * grid.x_min) * fft_vals
        psi_k = np.fft.fftshift(psi_k_unshifted)
        psi_p = psi_k / math.sqrt(hbar)

        k_density = np.abs(psi_k) ** 2
        p_density = np.abs(psi_p) ** 2

        norm_p = float(np.sum(p_density) * dp)
        if norm_p <= 0.0:
            raise ValueError("Momentum-space norm is zero.")

        exp_p = float(np.sum(p_shifted * p_density) * dp / norm_p)
        exp_p2 = float(np.sum((p_shifted ** 2) * p_density) * dp / norm_p)
        var_p = max(0.0, exp_p2 - exp_p ** 2)
        delta_p = float(np.sqrt(var_p))

        return MomentumRepresentation(
            k=k_shifted,
            p=p_shifted,
            dk=float(dk),
            dp=float(dp),
            psi_k=psi_k,
            psi_p=psi_p,
            k_probability_density=k_density,
            p_probability_density=p_density,
            fourier_norm=norm_p,
            expectation_p=exp_p,
            uncertainty_p=delta_p,
            hbar=float(hbar),
            x_min=grid.x_min
        )

    @staticmethod
    def to_position_space(momentum_rep: MomentumRepresentation, grid: SpatialGrid) -> np.ndarray:
        """Inverts a MomentumRepresentation back to position space psi(x)."""
        n = grid.points
        dx = grid.dx
        if momentum_rep.psi_k.shape != (n,):
            raise ValueError("MomentumRepresentation size does not match SpatialGrid.")

        k_unshifted = np.fft.ifftshift(momentum_rep.k)
        psi_k_unshifted = np.fft.ifftshift(momentum_rep.psi_k)

        fft_vals = psi_k_unshifted * (math.sqrt(2.0 * math.pi) / dx) * np.exp(1j * k_unshifted * grid.x_min)
        return np.fft.ifft(fft_vals)
