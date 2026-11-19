import math
from typing import Dict, Any, List, Literal
import numpy as np
from scipy.special import eval_genlaguerre
import sympy as sp
from .analytical_solver import AnalyticalSolution, AnalyticalSolver


class MorseAnalytical(AnalyticalSolution):
    """
    Exact closed-form bound-state analytical solution for the 1D Morse Potential:
        V(x) = D_e * (1 - exp(-a * (x - x_e)))^2
        E_n = D_e - (hbar^2 * a^2 / 2m) * (lambda - n - 1/2)^2
    where lambda = sqrt(2 * m * D_e) / (a * hbar) and 0 <= n < lambda - 1/2.
    """

    def __init__(
        self,
        dissociation_energy: float = 10.0,
        width_parameter: float = 1.0,
        equilibrium_position: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if dissociation_energy <= 0:
            raise ValueError("dissociation_energy (D_e) must be strictly positive.")
        if width_parameter <= 0:
            raise ValueError("width_parameter (a) must be strictly positive.")

        self.dissociation_energy = float(dissociation_energy)
        self.width_parameter = float(width_parameter)
        self.equilibrium_position = float(equilibrium_position)
        self.lam = math.sqrt(2.0 * self.mass * self.dissociation_energy) / (self.width_parameter * self.hbar)
        self.num_bound_states = max(0, int(math.ceil(self.lam - 0.5)))

    @property
    def name(self) -> str:
        return "MorsePotential"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "dissociation_energy": self.dissociation_energy,
            "width_parameter": self.width_parameter,
            "equilibrium_position": self.equilibrium_position,
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
                f"State n={n} is unbound for Morse potential (num_bound_states={self.num_bound_states})."
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
        prefactor = (self.hbar ** 2) * (self.width_parameter ** 2) / (2.0 * self.mass)
        return float(self.dissociation_energy - prefactor * ((self.lam - n - 0.5) ** 2))

    def normalization_constant(self, n: int) -> float:
        self._validate_quantum_number(n)
        s = 2.0 * self.lam - 2.0 * n - 1.0
        log_norm_sq = (
            math.log(self.width_parameter * s)
            + math.lgamma(n + 1)
            - math.lgamma(n + s + 1)
        )
        return float(math.exp(0.5 * log_norm_sq))

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        self._validate_quantum_number(n)
        x_arr = np.asarray(x, dtype=np.float64)
        s = 2.0 * self.lam - 2.0 * n - 1.0
        norm = self.normalization_constant(n)

        arg = np.clip(-self.width_parameter * (x_arr - self.equilibrium_position), -350.0, 350.0)
        z = 2.0 * self.lam * np.exp(arg)

        # Evaluate envelope z^(s/2) * exp(-z/2) in log-space to avoid inf * 0 = NaN
        psi = np.zeros_like(x_arr, dtype=np.float64)
        positive_z = z > 0.0
        if np.any(positive_z):
            z_pos = z[positive_z]
            log_env = 0.5 * s * np.log(z_pos) - 0.5 * z_pos
            active = log_env > -700.0
            if np.any(active):
                z_act = z_pos[active]
                env_act = np.exp(log_env[active])
                laguerre_vals = eval_genlaguerre(n, s, z_act)
                active_indices = np.where(positive_z)[0][active]
                psi[active_indices] = norm * env_act * laguerre_vals

        return psi

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-6) -> bool:
        self._validate_quantum_number(n)
        endpoints = np.array([x[0], x[-1]], dtype=np.float64)
        vals = self.wavefunction(n, endpoints)
        return bool(np.all(np.abs(vals) <= tolerance))

    @staticmethod
    def symbolic_energy() -> sp.Expr:
        n = sp.symbols("n", integer=True, nonnegative=True)
        D_e, a, hbar, m = sp.symbols("D_e a hbar m", positive=True)
        lam = sp.sqrt(2 * m * D_e) / (a * hbar)
        return D_e - (hbar ** 2 * a ** 2 / (2 * m)) * (lam - n - sp.Rational(1, 2)) ** 2


AnalyticalSolver.register("morse", MorseAnalytical)
