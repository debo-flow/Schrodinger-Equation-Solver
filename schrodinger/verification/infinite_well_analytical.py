import math
from typing import Dict, Any, List, Literal
import numpy as np
import sympy as sp
from .analytical_solver import AnalyticalSolution, AnalyticalSolver


class InfiniteSquareWellAnalytical(AnalyticalSolution):
    """
    Exact analytical solution for the 1D Infinite Square Well on [x_min, x_min + width].
    
    E_n = n^2 * pi^2 * hbar^2 / (2 * m * L^2),  n = 1, 2, 3, ...
    psi_n(x) = sqrt(2 / L) * sin(n * pi * (x - x_min) / L)
    """

    def __init__(
        self,
        width: float = 1.0,
        x_min: float = 0.0,
        mass: float = 1.0,
        hbar: float = 1.0,
        unit_system: Literal["dimensionless", "si"] = "dimensionless"
    ):
        super().__init__(mass=mass, hbar=hbar, unit_system=unit_system)
        if width <= 0:
            raise ValueError(f"Well width must be strictly positive. Received: {width}")

        self.width = float(width)
        self.x_min = float(x_min)
        self.x_max = self.x_min + self.width
        self.center = self.x_min + 0.5 * self.width

    @property
    def name(self) -> str:
        return "InfiniteSquareWell"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "center": self.center,
            "mass": self.mass,
            "hbar": self.hbar,
            "unit_system": self.unit_system
        }

    def _validate_quantum_number(self, n: int) -> None:
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Quantum number n must be an integer >= 1. Received: {n}")

    def quantum_numbers(self, num_states: int) -> List[int]:
        if num_states <= 0:
            raise ValueError("num_states must be strictly positive.")
        return list(range(1, num_states + 1))

    def energy(self, n: int) -> float:
        self._validate_quantum_number(n)
        return float((n ** 2) * (math.pi ** 2) * (self.hbar ** 2) / (2.0 * self.mass * (self.width ** 2)))

    def normalization_constant(self, n: int) -> float:
        self._validate_quantum_number(n)
        return float(math.sqrt(2.0 / self.width))

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        self._validate_quantum_number(n)
        x_arr = np.asarray(x, dtype=np.float64)
        psi = np.zeros_like(x_arr, dtype=np.float64)

        inside = (x_arr >= self.x_min) & (x_arr <= self.x_max)
        norm = self.normalization_constant(n)
        psi[inside] = norm * np.sin(n * math.pi * (x_arr[inside] - self.x_min) / self.width)

        # Clean floating-point sin(n*pi) residue at exact endpoints
        at_boundary = np.isclose(x_arr, self.x_min, atol=1e-15) | np.isclose(x_arr, self.x_max, atol=1e-15)
        psi[at_boundary] = 0.0
        return psi

    def verify_boundary_conditions(self, n: int, x: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Verifies Dirichlet condition psi(x_min) = psi(x_max) = 0."""
        self._validate_quantum_number(n)
        endpoints = np.array([self.x_min, self.x_max], dtype=np.float64)
        boundary_vals = self.wavefunction(n, endpoints)
        grid_vals = self.wavefunction(n, np.array([x[0], x[-1]], dtype=np.float64))
        return bool(np.all(np.abs(boundary_vals) <= tolerance) and np.all(np.abs(grid_vals) <= tolerance))

    @staticmethod
    def symbolic_energy() -> sp.Expr:
        """Returns the SymPy symbolic expression for E_n."""
        n, hbar, m, L = sp.symbols("n hbar m L", positive=True)
        return (n ** 2 * sp.pi ** 2 * hbar ** 2) / (2 * m * L ** 2)

    @staticmethod
    def symbolic_wavefunction() -> sp.Expr:
        """Returns the SymPy symbolic expression for psi_n(x) on [0, L]."""
        x, L = sp.symbols("x L", real=True, positive=True)
        n = sp.symbols("n", integer=True, positive=True)
        return sp.sqrt(2 / L) * sp.sin(n * sp.pi * x / L)


AnalyticalSolver.register("infinite_square_well", InfiniteSquareWellAnalytical)
