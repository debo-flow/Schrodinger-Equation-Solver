import math
from typing import Union
import numpy as np
from .transmission import TransmissionCalculator


class ReflectionCalculator:
    """
    Computes quantum reflection probabilities R = |r|^2 from complex
    reflection amplitudes and closed-form analytical benchmarks.
    """

    @staticmethod
    def from_amplitude(r_amplitude: complex) -> float:
        """Calculates reflection probability R = |r|^2."""
        r_prob = abs(r_amplitude) ** 2
        if not math.isfinite(r_prob):
            raise ValueError("Non-finite reflection amplitude encountered.")
        return float(r_prob)

    @staticmethod
    def analytical_rectangular_barrier(
        energy: float,
        barrier_height: float,
        barrier_width: float,
        baseline: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """Exact analytical reflection probability R = 1 - T for a rectangular barrier."""
        t_val = TransmissionCalculator.analytical_rectangular_barrier(
            energy=energy,
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            baseline=baseline,
            mass=mass,
            hbar=hbar
        )
        return float(max(0.0, 1.0 - t_val))

    @staticmethod
    def analytical_potential_step(
        energy: float,
        left_height: float,
        right_height: float,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """
        Exact analytical reflection probability for a potential step:
            R = ((k_L - k_R) / (k_L + k_R))^2  for E > max(V_L, V_R)
            R = 1.0                            for V_L < E <= V_R
        """
        if mass <= 0 or hbar <= 0:
            raise ValueError("mass and hbar must be strictly positive.")
        if energy <= left_height:
            raise ValueError(f"Incident energy ({energy}) must exceed left_height ({left_height}).")
        if energy <= right_height:
            return 1.0

        k_l = math.sqrt(2.0 * mass * (energy - left_height)) / hbar
        k_r = math.sqrt(2.0 * mass * (energy - right_height)) / hbar
        return float(((k_l - k_r) / (k_l + k_r)) ** 2)

    @staticmethod
    def analytical_finite_square_well(
        energy: float,
        depth: float,
        width: float,
        mass: float = 1.0,
        hbar: float = 1.0
    ) -> float:
        """Exact analytical reflection probability R = 1 - T for a finite square well."""
        t_val = TransmissionCalculator.analytical_finite_square_well(
            energy=energy,
            depth=depth,
            width=width,
            mass=mass,
            hbar=hbar
        )
        return float(max(0.0, 1.0 - t_val))

    @staticmethod
    def sweep_energies(
        energies: Union[np.ndarray, list],
        reflection_fn
    ) -> np.ndarray:
        """Evaluates a scalar reflection function over a 1D array of energies."""
        e_arr = np.asarray(energies, dtype=np.float64)
        return np.array([float(reflection_fn(float(e))) for e in e_arr], dtype=np.float64)
