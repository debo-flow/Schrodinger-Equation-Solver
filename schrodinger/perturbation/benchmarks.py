import math
from typing import Dict, Any, Literal
import numpy as np
from schrodinger.potentials.base import Potential


class MonomialPotential(Potential):
    """
    Configurable monomial perturbation potential:
        V'(x) = coeff * (x - center)^power
    Supports linear (1), quadratic (2), cubic (3), quartic (4), and arbitrary integer powers.
    """

    def __init__(
        self,
        power: int,
        coeff: float = 1.0,
        center: float = 0.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(unit_system=unit_system)
        if not isinstance(power, (int, np.integer)) or power < 0:
            raise ValueError(f"power must be a non-negative integer. Received: {power}")
        if not math.isfinite(coeff) or not math.isfinite(center):
            raise ValueError("coeff and center must be finite real numbers.")
        self.power = int(power)
        self.coeff = float(coeff)
        self.center = float(center)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.coeff * ((x - self.center) ** self.power)


class HarmonicOscillatorPerturbationBenchmarks:
    """
    Analytical Rayleigh-Schrödinger perturbation formulas and matrix elements
    for the 1D Quantum Harmonic Oscillator (H_0 = p^2/(2m) + 1/2 m omega^2 x^2)
    subject to polynomial perturbations H' = c * x^p (p = 1, 2, 3, 4).
    """

    def __init__(self, mass: float = 1.0, omega: float = 1.0, hbar: float = 1.0):
        if mass <= 0 or omega <= 0 or hbar <= 0:
            raise ValueError("mass, omega, and hbar must be strictly positive.")
        self.mass = float(mass)
        self.omega = float(omega)
        self.hbar = float(hbar)
        self.length_scale = math.sqrt(self.hbar / (self.mass * self.omega))

    def unperturbed_energy(self, n: int) -> float:
        if n < 0:
            raise ValueError("Quantum number n must be >= 0.")
        return float(self.hbar * self.omega * (n + 0.5))

    def linear_perturbation(self, n: int, alpha: float) -> Dict[str, Any]:
        """
        H' = alpha * x.
        Exact closed-form shift: E_n = E_n^(0) - alpha^2 / (2 * m * omega^2).
        """
        e0 = self.unperturbed_energy(n)
        e1 = 0.0
        e2 = -(alpha ** 2) / (2.0 * self.mass * (self.omega ** 2))
        return {
            "model": "linear_perturbation",
            "n": n,
            "E0": e0,
            "E1": e1,
            "E2": float(e2),
            "E_approx_2nd": float(e0 + e1 + e2),
            "E_exact": float(e0 + e2),
            "notes": "Second-order perturbation theory is exact for a linear perturbation."
        }

    def quadratic_perturbation(self, n: int, gamma: float) -> Dict[str, Any]:
        """
        H' = gamma * x^2.
        Exact energy for omega^2 + 2*gamma/m > 0: E_n = hbar * sqrt(omega^2 + 2*gamma/m) * (n + 1/2).
        """
        e0 = self.unperturbed_energy(n)
        ell2 = self.length_scale ** 2
        e1 = gamma * ell2 * (n + 0.5)
        e2 = -((gamma ** 2) * (ell2 ** 2) / (2.0 * self.hbar * self.omega)) * (n + 0.5)

        omega_eff_sq = (self.omega ** 2) + (2.0 * gamma / self.mass)
        e_exact = (
            float(self.hbar * math.sqrt(omega_eff_sq) * (n + 0.5))
            if omega_eff_sq > 0.0
            else None
        )
        convergent_radius = 0.5 * self.mass * (self.omega ** 2)
        return {
            "model": "quadratic_perturbation",
            "n": n,
            "E0": e0,
            "E1": float(e1),
            "E2": float(e2),
            "E_approx_1st": float(e0 + e1),
            "E_approx_2nd": float(e0 + e1 + e2),
            "E_exact": e_exact,
            "convergence_radius": float(convergent_radius),
            "within_convergence_radius": bool(abs(gamma) < convergent_radius)
        }

    def quartic_perturbation(self, n: int, beta: float) -> Dict[str, Any]:
        """
        H' = beta * x^4 (Quartic Anharmonic Oscillator).
        Asymptotic series (zero radius of convergence); accurate for small beta > 0 and low n.
        """
        e0 = self.unperturbed_energy(n)
        ell4 = self.length_scale ** 4
        e1 = 0.75 * beta * ell4 * (2 * (n ** 2) + 2 * n + 1)
        e2 = -((beta ** 2) * (ell4 ** 2) / (8.0 * self.hbar * self.omega)) * (
            34 * (n ** 3) + 51 * (n ** 2) + 59 * n + 21
        )
        return {
            "model": "quartic_anharmonic_oscillator",
            "n": n,
            "E0": e0,
            "E1": float(e1),
            "E2": float(e2),
            "E_approx_1st": float(e0 + e1),
            "E_approx_2nd": float(e0 + e1 + e2),
            "is_asymptotic_series": True,
            "notes": (
                "The quartic perturbation expansion is asymptotic (diverges for any beta != 0 at high order); "
                "best accuracy is achieved for small beta > 0 and low quantum numbers."
            )
        }

    def cubic_perturbation(self, n: int, kappa: float) -> Dict[str, Any]:
        """
        H' = kappa * x^3.
        WARNING: Globally unbounded below as x -> +/- inf; perturbative levels describe
        metastable states localized in the harmonic well for small |kappa|.
        """
        e0 = self.unperturbed_energy(n)
        ell6 = self.length_scale ** 6
        e1 = 0.0
        e2 = -((kappa ** 2) * ell6 / (8.0 * self.hbar * self.omega)) * (
            30 * (n ** 2) + 30 * n + 11
        )
        # Classical barrier height of V(x) = 1/2 m omega^2 x^2 + kappa x^3 occurs at x_b = -m*omega^2/(3*kappa)
        barrier_height = (
            float(((self.mass * (self.omega ** 2)) ** 3) / (54.0 * (kappa ** 2)))
            if kappa != 0.0
            else float("inf")
        )
        return {
            "model": "cubic_perturbation",
            "n": n,
            "E0": e0,
            "E1": e1,
            "E2": float(e2),
            "E_approx_2nd": float(e0 + e2),
            "metastable_barrier_height": barrier_height,
            "is_globally_bounded": False,
            "stability_warning": (
                "V_0 + kappa*x^3 is unbounded below; perturbation theory only approximates "
                "metastable states well below the local barrier height."
            )
        }

    def x_matrix_element(self, m: int, n: int) -> float:
        """Exact analytical matrix element <m | x | n> for the 1D Harmonic Oscillator."""
        if m == n + 1:
            return float((self.length_scale / math.sqrt(2.0)) * math.sqrt(n + 1))
        if m == n - 1:
            return float((self.length_scale / math.sqrt(2.0)) * math.sqrt(n))
        return 0.0

    def x4_diagonal_element(self, n: int) -> float:
        """Exact analytical diagonal element <n | x^4 | n>."""
        return float(0.75 * (self.length_scale ** 4) * (2 * (n ** 2) + 2 * n + 1))


class InfiniteWellPerturbationBenchmark:
    """
    Analytical perturbation benchmarks for a 1D Infinite Square Well on [0, L].
    """

    def __init__(self, width: float = 1.0, mass: float = 1.0, hbar: float = 1.0):
        if width <= 0 or mass <= 0 or hbar <= 0:
            raise ValueError("width, mass, and hbar must be strictly positive.")
        self.width = float(width)
        self.mass = float(mass)
        self.hbar = float(hbar)

    def unperturbed_energy(self, n: int) -> float:
        if n < 1:
            raise ValueError("Infinite square well quantum number n must be >= 1.")
        return float((n ** 2) * (math.pi ** 2) * (self.hbar ** 2) / (2.0 * self.mass * (self.width ** 2)))

    def linear_tilt_first_order(self, n: int, slope: float) -> float:
        """
        Exact first-order shift for H' = slope * x on [0, L]:
            <n | slope * x | n> = slope * L / 2
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        return float(0.5 * slope * self.width)

    def centered_linear_tilt_first_order(self, n: int, slope: float) -> float:
        """
        Exact first-order shift for H' = slope * (x - L/2) on [0, L] is identically 0 by parity.
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        return 0.0

    def half_well_step_first_order(self, n: int, step_height: float) -> float:
        """
        Exact first-order shift for H' = step_height on [0, L/2] and 0 on (L/2, L]:
            E_n^(1) = step_height / 2
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        return float(0.5 * step_height)
