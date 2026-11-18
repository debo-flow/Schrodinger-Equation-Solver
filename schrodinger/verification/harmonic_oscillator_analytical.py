import math
from typing import Dict, Any, List, Literal
import numpy as np
from scipy.special import eval_hermite
import sympy as sp
from .analytical_solver import AnalyticalSolution, AnalyticalSolver


class HarmonicOscillatorAnalytical(AnalyticalSolution):
    """
    Exact analytical solution for the 1D Quantum Harmonic Oscillator.
    
    V(x) = 0.5 * m * omega^2 * (x - center)^2
    E_n = hbar * omega * (n + 1/2),  n = 0, 1, 2, ...
    psi_n(x) = (m*omega/(pi*hbar))^(1/4) * (1/sqrt(2^n * n!)) * H_n(xi) * exp(-xi^2 / 2)
    """

    def __init__(
        self,
        omega: float = 1.0,
        center: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if omega <= 0:
            raise ValueError(f"Angular frequency omega must be strictly positive. Received: {omega}")

        self.omega = float(omega)
        self.center = float(center)
        self.alpha = math.sqrt((self.mass * self.omega) / self.hbar)

    @property
    def name(self) -> str:
        return "HarmonicOscillator"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "omega": self.omega,
            "center": self.center,
            "mass": self.mass,
            "hbar": self.hbar,
            "alpha": self.alpha,
            "unit_system": self.unit_system
        }

    def _validate_quantum_number(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"Quantum number n must be an integer >= 0. Received: {n}")

    def quantum_numbers(self, num_states: int) -> List[int]:
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive.")
        return list(range(0, num_states))

    def energy(self, n: int) -> float:
        self._validate_quantum_number(n)
        return float(self.hbar * self.omega * (n + 0.5))

    def normalization_constant(self, n: int) -> float:
        """
        Computes N_n = (alpha / sqrt(pi))^(1/2) / sqrt(2^n * n!) using log-gamma
        to prevent overflow for high quantum numbers.
        """
        self._validate_quantum_number(n)
        log_prefactor = 0.25 * math.log((self.mass * self.omega) / (math.pi * self.hbar))
        log_denom = 0.5 * (n * math.log(2.0) + math.lgamma(n + 1))
        return float(math.exp(log_prefactor - log_denom))

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        self._validate_quantum_number(n)
        x_arr = np.asarray(x, dtype=np.float64)
        xi = self.alpha * (x_arr - self.center)

        norm = self.normalization_constant(n)
        hermite_vals = eval_hermite(n, xi)
        gaussian = np.exp(-0.5 * (xi ** 2))

        psi = norm * hermite_vals * gaussian
        if not np.all(np.isfinite(psi)):
            raise OverflowError(
                f"Numerical overflow encountered evaluating Harmonic Oscillator state n={n}."
            )
        return psi

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Verifies asymptotic decay of the wavefunction at the finite grid boundaries.
        """
        self._validate_quantum_number(n)
        endpoints = np.array([x[0], x[-1]], dtype=np.float64)
        vals = self.wavefunction(n, endpoints)
        return bool(np.all(np.abs(vals) <= tolerance))

    @staticmethod
    def symbolic_energy() -> sp.Expr:
        """Returns the SymPy symbolic expression for E_n."""
        n = sp.symbols("n", integer=True, nonnegative=True)
        hbar, omega = sp.symbols("hbar omega", positive=True)
        return hbar * omega * (n + sp.Rational(1, 2))

    @staticmethod
    def symbolic_wavefunction(n: int) -> sp.Expr:
        """Returns the SymPy symbolic expression for psi_n(x) centered at 0."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer.")
        x = sp.symbols("x", real=True)
        m, omega, hbar = sp.symbols("m omega hbar", positive=True)
        alpha = sp.sqrt(m * omega / hbar)
        xi = alpha * x
        norm = (m * omega / (sp.pi * hbar)) ** sp.Rational(1, 4) / sp.sqrt(2 ** n * sp.factorial(n))
        return norm * sp.hermite(n, xi) * sp.exp(-xi ** 2 / 2)


AnalyticalSolver.register("harmonic_oscillator", HarmonicOscillatorAnalytical)
