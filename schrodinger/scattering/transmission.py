import math
from typing import Union
import numpy as np


class TransmissionCalculator:
    """
    Computes flux-normalized quantum transmission probabilities T from complex
    transmission amplitudes, as well as closed-form analytical benchmarks.
    """

    @staticmethod
    def wavevector(energy: float, potential: float, mass: float = 1.0, hbar: float = 1.0) -> complex:
        """
        Computes complex wavevector k = sqrt(2 * m * (E - V)) / hbar.
        Returns real positive k when E >= V, and purely imaginary i*kappa when E < V.
        """
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if not (math.isfinite(energy) and math.isfinite(potential)):
            raise ValueError("energy and potential must be finite.")

        diff = energy - potential
        if diff >= 0.0:
            return complex(math.sqrt(2.0 * mass * diff) / hbar, 0.0)
        return complex(0.0, math.sqrt(2.0 * mass * (-diff)) / hbar)

    @staticmethod
    def from_amplitude(
        t_amplitude: complex,
        energy: float,
        v_left: float = 0.0,
        v_right: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Calculates transmission probability:
            T = (k_R / k_L) * |t|^2   for E > max(V_L, V_R)
            T = 0.0                   for E <= V_R (evanescent right channel)
        """
        if energy <= v_left:
            raise ValueError(
                f"Incident energy ({energy}) must exceed left asymptotic potential ({v_left})."
            )
        if energy <= v_right:
            return 0.0

        k_left = math.sqrt(2.0 * mass * (energy - v_left)) / hbar
        k_right = math.sqrt(2.0 * mass * (energy - v_right)) / hbar
        t_prob = (k_right / k_left) * (abs(t_amplitude) ** 2)
        return float(t_prob)

    @staticmethod
    def analytical_rectangular_barrier(
        energy: float,
        barrier_height: float,
        barrier_width: float,
        baseline: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Exact analytical transmission probability for a rectangular barrier of
        height `barrier_height` above `baseline` and width `barrier_width`.
        """
        if barrier_width <= 0:
            raise ValueError("barrier_width must be strictly positive.")
        if barrier_height <= baseline:
            raise ValueError("barrier_height must exceed baseline.")
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if energy <= baseline:
            raise ValueError(f"Incident energy ({energy}) must exceed baseline ({baseline}).")

        e_rel = energy - baseline
        v0 = barrier_height - baseline

        if math.isclose(e_rel, v0, rel_tol=1e-13, abs_tol=1e-15):
            denom = 1.0 + (mass * v0 * (barrier_width ** 2)) / (2.0 * (hbar ** 2))
            return float(1.0 / denom)

        if e_rel < v0:
            kappa = math.sqrt(2.0 * mass * (v0 - e_rel)) / hbar
            arg = kappa * barrier_width
            if arg > 350.0:
                return 0.0
            sinh_val = math.sinh(arg)
            denom = 1.0 + ((v0 ** 2) * (sinh_val ** 2)) / (4.0 * e_rel * (v0 - e_rel))
            return float(1.0 / denom)

        q = math.sqrt(2.0 * mass * (e_rel - v0)) / hbar
        sin_val = math.sin(q * barrier_width)
        denom = 1.0 + ((v0 ** 2) * (sin_val ** 2)) / (4.0 * e_rel * (e_rel - v0))
        return float(1.0 / denom)

    @staticmethod
    def analytical_potential_step(
        energy: float,
        left_height: float,
        right_height: float,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Exact analytical transmission probability for a potential step:
            T = 4 * k_L * k_R / (k_L + k_R)^2  for E > max(V_L, V_R)
            T = 0.0                            for V_L < E <= V_R
        """
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if energy <= left_height:
            raise ValueError(f"Incident energy ({energy}) must exceed left_height ({left_height}).")
        if energy <= right_height:
            return 0.0

        k_l = math.sqrt(2.0 * mass * (energy - left_height)) / hbar
        k_r = math.sqrt(2.0 * mass * (energy - right_height)) / hbar
        return float((4.0 * k_l * k_r) / ((k_l + k_r) ** 2))

    @staticmethod
    def analytical_finite_square_well(
        energy: float,
        depth: float,
        width: float,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Exact analytical transmission probability across a finite square well
        of depth V_0 > 0 (V = -V_0 inside, 0 outside) for E > 0:
            T = [1 + V_0^2 * sin^2(q * L) / (4 * E * (E + V_0))]^(-1)
        """
        if depth <= 0 or width <= 0:
            raise ValueError("depth and width must be strictly positive.")
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if energy <= 0.0:
            raise ValueError(f"Scattering energy ({energy}) must be strictly positive above well baseline 0.")

        q = math.sqrt(2.0 * mass * (energy + depth)) / hbar
        sin_val = math.sin(q * width)
        denom = 1.0 + ((depth ** 2) * (sin_val ** 2)) / (4.0 * energy * (energy + depth))
        return float(1.0 / denom)

    @staticmethod
    def verify_probability_conservation(
        reflection: float,
        transmission: float,
        tolerance: float = 1e-6
    ) -> bool:
        """Verifies R >= 0, T >= 0, and |R + T - 1| <= tolerance."""
        if not (math.isfinite(reflection) and math.isfinite(transmission)):
            return False
        if reflection < -tolerance or transmission < -tolerance:
            return False
        return abs((reflection + transmission) - 1.0) <= tolerance

    @staticmethod
    def sweep_energies(
        energies: Union[np.ndarray, list],
        transmission_fn
    ) -> np.ndarray:
        """Evaluates a scalar transmission function over a 1D array of energies."""
        e_arr = np.asarray(energies, dtype=np.float64)
        return np.array([float(transmission_fn(float(e))) for e in e_arr], dtype=np.float64)
