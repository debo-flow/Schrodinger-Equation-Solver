import math
from typing import Dict, Any, List, Literal
import numpy as np
from scipy.special import eval_gegenbauer
import sympy as sp
from .analytical_solver import AnalyticalSolution, AnalyticalSolver


class PoschlTellerAnalytical(AnalyticalSolution):
    """
    Exact closed-form bound-state analytical solution for the symmetric Pöschl–Teller well:
        V(x) = -V_0 * sech^2(alpha * (x - center))
        E_n = -(hbar^2 * alpha^2 / 2m) * (lambda - 1 - n)^2
    where lambda = 1/2 + sqrt(1/4 + 2*m*V_0 / (hbar^2 * alpha^2)) and 0 <= n < lambda - 1.
    """

    def __init__(
        self,
        depth: float = 1.0,
        alpha: float = 1.0,
        center: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if depth <= 0:
            raise ValueError("depth (V_0) must be strictly positive.")
        if alpha <= 0:
            raise ValueError("alpha must be strictly positive.")

        self.depth = float(depth)
        self.alpha = float(alpha)
        self.center = float(center)
        self.lam = 0.5 + math.sqrt(0.25 + (2.0 * self.mass * self.depth) / ((self.hbar ** 2) * (self.alpha ** 2)))
        self.num_bound_states = max(0, int(math.ceil(self.lam - 1.0)))

    @property
    def name(self) -> str:
        return "PoschlTellerPotential"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "alpha": self.alpha,
            "center": self.center,
            "mass": self.mass,
            "hbar": self.hbar,
            "lambda": self.lam,
            "num_bound_states": self.num_bound_states,
            "unit_system": self.unit_system
        }

    def _validate_quantum_number(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"Quantum number n must be an integer >= 0. Received: {n}")
        if n >= self.num_bound_states:
            raise ValueError(
                f"State n={n} is unbound for Pöschl–Teller potential (num_bound_states={self.num_bound_states})."
            )

    def quantum_numbers(self, num_states: int) -> List[int]:
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive.")
        if num_states > self.num_bound_states:
            raise ValueError(
                f"Requested {num_states} bound states, but only {self.num_bound_states} exist."
            )
        return list(range(0, num_states))

    def energy(self, n: int) -> float:
        self._validate_quantum_number(n)
        mu = self.lam - 1.0 - n
        return float(-((self.hbar ** 2) * (self.alpha ** 2) / (2.0 * self.mass)) * (mu ** 2))

    def normalization_constant(self, n: int) -> float:
        self._validate_quantum_number(n)
        mu = self.lam - 1.0 - n
        log_integral = (
            math.log(math.pi)
            - 2.0 * mu * math.log(2.0)
            + math.lgamma(n + 2.0 * mu + 1.0)
            - math.lgamma(n + 1.0)
            - math.log(mu)
            - 2.0 * math.lgamma(mu + 0.5)
        )
        return float(math.exp(0.5 * (math.log(self.alpha) - log_integral)))

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        self._validate_quantum_number(n)
        x_arr = np.asarray(x, dtype=np.float64)
        mu = self.lam - 1.0 - n
        norm = self.normalization_constant(n)

        y = self.alpha * (x_arr - self.center)
        abs_y = np.abs(y)
        # Overflow-safe ln(cosh(y)) = |y| + ln(1 + exp(-2|y|)) - ln(2)
        log_cosh = abs_y + np.log1p(np.exp(-2.0 * abs_y)) - math.log(2.0)
        sech_mu = np.exp(-mu * log_cosh)

        u = np.tanh(y)
        gegenbauer_vals = eval_gegenbauer(n, mu + 0.5, u)
        return norm * sech_mu * gegenbauer_vals

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        self._validate_quantum_number(n)
        endpoints = np.array([x[0], x[-1]], dtype=np.float64)
        vals = self.wavefunction(n, endpoints)
        return bool(np.all(np.abs(vals) <= tolerance))

    @staticmethod
    def symbolic_energy() -> sp.Expr:
        n = sp.symbols("n", integer=True, nonnegative=True)
        lam, alpha, hbar, m = sp.symbols("lambda alpha hbar m", positive=True)
        return -(hbar ** 2 * alpha ** 2 / (2 * m)) * (lam - 1 - n) ** 2


AnalyticalSolver.register("poschl_teller", PoschlTellerAnalytical)
