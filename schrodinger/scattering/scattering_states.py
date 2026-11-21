from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from .transmission import TransmissionCalculator
from .transfer_matrix import PotentialSlab


@dataclass
class ScatteringState:
    """
    Represents a stationary 1D scattering state across spatial grid `x`,
    including incident, reflected, and transmitted asymptotic waves and
    the stationary probability current density j(x).
    """
    x: np.ndarray
    psi: np.ndarray
    dpsi_dx: np.ndarray
    incident_wave: np.ndarray
    reflected_wave: np.ndarray
    transmitted_wave: np.ndarray
    energy: float
    r_amplitude: complex
    t_amplitude: complex
    reflection: float
    transmission: float
    k_left: float
    k_right: complex
    x_left: float
    x_right: float
    mass: float = 1.0
    hbar: float = 1.0

    @property
    def probability_density(self) -> np.ndarray:
        """Returns |psi(x)|^2."""
        return np.abs(self.psi) ** 2

    @property
    def probability_current(self) -> np.ndarray:
        """
        Computes stationary probability current density:
            j(x) = (hbar / m) * Im(psi^*(x) * d(psi)/dx)
        """
        return (self.hbar / self.mass) * np.imag(np.conj(self.psi) * self.dpsi_dx)

    @property
    def incident_flux(self) -> float:
        """Returns incident probability current j_inc = hbar * k_L / m."""
        return float((self.hbar * self.k_left) / self.mass)

    @property
    def current_conservation_error(self) -> float:
        """
        Measures maximum relative deviation of j(x) from the asymptotic net flux
        j_net = j_inc * (1 - R) across the spatial domain.
        """
        j_arr = self.probability_current
        j_expected = self.incident_flux * (1.0 - self.reflection)
        scale = max(self.incident_flux, 1e-15)
        return float(np.max(np.abs(j_arr - j_expected)) / scale)


class ScatteringStateBuilder:
    """
    Constructs complete ScatteringState objects from asymptotic amplitudes
    and either piecewise-constant slabs or direct ODE integration profiles.
    """

    @staticmethod
    def build_asymptotic_waves(
        x: np.ndarray,
        x_left: float,
        x_right: float,
        k_left: float,
        k_right: complex,
        r_amplitude: complex,
        t_amplitude: complex
    ) -> tuple:
        """
        Evaluates incident, reflected, and transmitted wave components on `x`:
            - psi_inc(x)   = exp(i * k_L * (x - x_L))
            - psi_ref(x)   = r * exp(-i * k_L * (x - x_L))
            - psi_trans(x) = t * exp(i * k_R * (x - x_R))
        """
        x_arr = np.asarray(x, dtype=np.float64)
        psi_inc = np.exp(1j * k_left * (x_arr - x_left))
        psi_ref = r_amplitude * np.exp(-1j * k_left * (x_arr - x_left))
        psi_trans = t_amplitude * np.exp(1j * k_right * (x_arr - x_right))
        return psi_inc, psi_ref, psi_trans

    @classmethod
    def from_slabs(
        cls,
        x: np.ndarray,
        energy: float,
        slabs: Sequence[PotentialSlab],
        slab_amplitudes: Sequence[tuple],
        r_amplitude: complex,
        t_amplitude: complex,
        reflection: float,
        transmission: float,
        v_left: float = 0.0,
        v_right: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> ScatteringState:
        """
        Reconstructs exact piecewise-analytic psi(x) and d(psi)/dx across left asymptotic,
        interior slab, and right asymptotic regions.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        x_left = slabs[0].x_start
        x_right = slabs[-1].x_end

        k_left_c = TransmissionCalculator.wavevector(energy, v_left, mass=mass, hbar=hbar)
        k_left = float(np.real(k_left_c))
        k_right = TransmissionCalculator.wavevector(energy, v_right, mass=mass, hbar=hbar)

        psi_inc, psi_ref, psi_trans = cls.build_asymptotic_waves(
            x_arr, x_left, x_right, k_left, k_right, r_amplitude, t_amplitude
        )

        psi = np.zeros_like(x_arr, dtype=np.complex128)
        dpsi = np.zeros_like(x_arr, dtype=np.complex128)

        # Left asymptotic region (x < x_left)
        left_mask = x_arr < x_left
        if np.any(left_mask):
            psi[left_mask] = psi_inc[left_mask] + psi_ref[left_mask]
            dpsi[left_mask] = 1j * k_left * (psi_inc[left_mask] - psi_ref[left_mask])

        # Interior slabs
        for idx, (slab, (a_j, b_j)) in enumerate(zip(slabs, slab_amplitudes)):
            if idx == len(slabs) - 1:
                mask = (x_arr >= slab.x_start) & (x_arr <= slab.x_end)
            else:
                mask = (x_arr >= slab.x_start) & (x_arr < slab.x_end)

            if np.any(mask):
                k_j = TransmissionCalculator.wavevector(energy, slab.potential, mass=mass, hbar=hbar)
                dx_local = x_arr[mask] - slab.x_start
                term_pos = a_j * np.exp(1j * k_j * dx_local)
                term_neg = b_j * np.exp(-1j * k_j * dx_local)
                psi[mask] = term_pos + term_neg
                dpsi[mask] = 1j * k_j * (term_pos - term_neg)

        # Right asymptotic region (x > x_right)
        right_mask = x_arr > x_right
        if np.any(right_mask):
            psi[right_mask] = psi_trans[right_mask]
            dpsi[right_mask] = 1j * k_right * psi_trans[right_mask]

        return ScatteringState(
            x=x_arr,
            psi=psi,
            dpsi_dx=dpsi,
            incident_wave=psi_inc,
            reflected_wave=psi_ref,
            transmitted_wave=psi_trans,
            energy=float(energy),
            r_amplitude=r_amplitude,
            t_amplitude=t_amplitude,
            reflection=float(reflection),
            transmission=float(transmission),
            k_left=k_left,
            k_right=k_right,
            x_left=float(x_left),
            x_right=float(x_right),
            mass=float(mass),
            hbar=float(hbar)
        )
